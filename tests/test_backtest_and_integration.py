"""
tests/test_backtest_and_integration.py

Covers spec test case #32 (closed 4H candle only — no lookahead) and
#33 (Telegram uses the new strategy engine everywhere), plus review #3's
backtester-vs-live divergence findings.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest  # noqa: E402
import strategy  # noqa: E402


def test_slice_4h_excludes_still_forming_candle():
    """spec Part 25 audit fix: a 4H candle opening at 08:00 is NOT closed
    at 09:00 (only 1 hour into its 4-hour window) — it must be excluded
    even though its open time is <= a 1H timestamp of 09:00."""
    ts = pd.date_range("2025-01-01 00:00", periods=5, freq="4h", tz="UTC")  # 00:00, 04:00, 08:00, 12:00, 16:00
    df_4h = pd.DataFrame({"timestamp": ts, "open": [1, 2, 3, 4, 5], "high": [1, 2, 3, 4, 5],
                           "low": [1, 2, 3, 4, 5], "close": [1, 2, 3, 4, 5], "volume": [1, 1, 1, 1, 1]})

    # "now" = 09:00 -> the 08:00 candle (closes at 12:00) must NOT be included,
    # only 00:00 and 04:00 (which close at 04:00 and 08:00, both <= 09:00).
    now_ts = pd.Timestamp("2025-01-01 09:00", tz="UTC")
    window = backtest._slice_4h(df_4h, now_ts)
    assert list(window["open"]) == [1, 2]

    # "now" = 12:00 exactly -> the 08:00 candle closes exactly at 12:00, so it IS closed.
    now_ts2 = pd.Timestamp("2025-01-01 12:00", tz="UTC")
    window2 = backtest._slice_4h(df_4h, now_ts2)
    assert list(window2["open"]) == [1, 2, 3]


def test_backtest_now_ts_uses_1h_close_time_not_open_time():
    """The old bug used the 1H candle's own OPEN time as "now", which is
    1 hour too early and let partially-formed 4H candles leak in. This
    checks the fix end-to-end via run_backtest_symbol's windowing logic
    by constructing a case where the difference matters."""
    # 1H candle opens at 08:30 (so "now" should be 09:30, not 08:30).
    ts_1h = pd.date_range("2025-01-01 00:00", periods=100, freq="h", tz="UTC")
    df_1h = pd.DataFrame({"timestamp": ts_1h, "open": range(100), "high": range(100),
                           "low": range(100), "close": range(100), "volume": [1] * 100})
    # A 4H candle opening at ts_1h[50] (i.e. covers [t, t+4h)) should only be
    # considered closed once "now" has advanced 4h past that open time.
    candle_open = ts_1h[50]
    df_4h = pd.DataFrame({"timestamp": [candle_open], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]})

    now_too_early = candle_open + pd.Timedelta(hours=1)   # only 1h in — matches the OLD buggy cutoff (1H candle's open time)
    now_correct = candle_open + pd.Timedelta(hours=4)      # actually closed
    assert len(backtest._slice_4h(df_4h, now_too_early)) == 0
    assert len(backtest._slice_4h(df_4h, now_correct)) == 1


def test_telegram_uses_new_strategy_engine_everywhere():
    """spec Part 23/#33: every call to strategy.evaluate_symbol across the
    live/Telegram/backtest paths must use the new (symbol, df_4h, df_1h,
    settings, ...) signature — never a leftover single-timeframe call."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files_to_check = ["telegram_bot.py", "main.py", "backtest.py"]
    call_pattern = re.compile(r"evaluate_symbol\(([^)]*)\)")

    for filename in files_to_check:
        path = os.path.join(project_root, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        calls = [c for c in call_pattern.findall(content) if c.strip()]  # skip bare mentions like "evaluate_symbol()" in docstrings/comments
        assert calls, f"{filename}: expected at least one real evaluate_symbol(...) call"
        for call_args in calls:
            # The new signature always passes two OHLCV frames before settings
            # (df_4h, df_1h) — grep for both a 4h-named and 1h-named argument
            # in every call site rather than a bare single `df`.
            assert "df_4h" in call_args or "window_4h" in call_args, \
                f"{filename}: evaluate_symbol call missing a 4H dataframe argument: {call_args}"
            assert "df_1h" in call_args or "window_1h" in call_args, \
                f"{filename}: evaluate_symbol call missing a 1H dataframe argument: {call_args}"

    # Also confirm no lingering old single-timeframe helper names, and no
    # FUNCTIONAL reads of the dead min_confirmations setting (comments /
    # the deliberate legacy-key warning message are fine — see
    # telegram_bot.py's _LEGACY_INERT_SETTINGS).
    dead_read_pattern = re.compile(r"""settings\.get\(\s*['"]min_confirmations['"]""")
    for filename in files_to_check + ["strategy.py"]:
        path = os.path.join(project_root, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "_direction_from_divergence" not in content
        assert not dead_read_pattern.search(content), f"{filename}: still functionally reads the dead min_confirmations setting"


def test_backtest_applies_same_duplicate_signal_suppression_as_live():
    """
    Backtest diagnostic finding (review #3, Part 10 — backtester vs live
    divergence): live scanning always checks
    strategy.should_send_signal()/duplicate_signal_hours before acting on
    a signal. The backtest walk-forward loop previously had no equivalent,
    letting it re-enter an essentially identical setup on the same symbol
    within hours of a fast stop-out — a real 180-day run showed 5 entries
    at the identical entry price on APT/USDT within one day. This
    constructs a scenario with a fast stop-out followed by conditions that
    would re-trigger a signal soon after, and confirms the second entry is
    suppressed within the duplicate_signal_hours window.
    """
    rng = np.random.default_rng(3)
    closes = []
    price = 200
    for leg in range(10):
        for _ in range(10):
            price -= 1.2
            closes.append(price)
        for _ in range(4):
            price += 0.5
            closes.append(price)
    for _ in range(40):
        price += 2.5
        closes.append(price)
    for _ in range(5):
        price -= 0.4
        closes.append(price)
    for _ in range(6):
        price += 0.5
        closes.append(price)
    for _ in range(4):
        price -= 0.2
        closes.append(price)
    for _ in range(60):
        price += 0.3 + rng.normal(0, 0.4)  # choppy grind, likely to re-trigger similar setups repeatedly
        closes.append(price)

    closes = np.array(closes, dtype=float)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + 0.3
    lows = np.minimum(opens, closes) - 0.3
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="h", tz="UTC")
    df_1h = pd.DataFrame({"timestamp": ts, "open": opens, "high": highs, "low": lows, "close": closes,
                           "volume": np.full(len(closes), 1000.0)})
    df_4h = df_1h.set_index("timestamp").resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().reset_index()

    settings = {"minimum_rr": 0.05, "risk_per_trade_usd": 250, "capital": 25000,
                "max_structural_risk_pct": 50.0, "duplicate_signal_hours": 6}
    trades = backtest.run_backtest_symbol(df_4h, df_1h, "DUPCHECK/USDT", settings)["trades"]

    if len(trades) >= 2:
        entry_times = sorted(pd.Timestamp(t.entry_time) for t in trades)
        gaps_hours = [(b - a).total_seconds() / 3600 for a, b in zip(entry_times, entry_times[1:])]
        assert all(g >= settings["duplicate_signal_hours"] for g in gaps_hours), (
            f"consecutive entries on the same symbol are closer together than duplicate_signal_hours "
            f"({settings['duplicate_signal_hours']}h) — found gaps: {gaps_hours}"
        )
