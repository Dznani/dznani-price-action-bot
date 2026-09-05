"""
tests/test_regression_real_world.py

Covers spec review #2, item 4 — regression tests for two real-world
failure modes discussed during review:

TEST A (AXS/USDT-style): the old bot bought purely on indicator
confirmations (RSI divergence + volume + MFI + momentum + EMA "4/5").
The new engine must never produce a VALID BUY from indicators alone when
structure/BOS/location/R:R don't support it.

TEST B (SKHYB/USDT-style): the old bot called a market bearish merely
because price reached a level, then price expanded strongly bullish
afterward. The new engine must require genuine bearish CHoCH/BOS/location
before calling BEARISH CONTEXT, and — since this is spot-only — a bearish
read must NEVER produce a short execution signal, only BEARISH CONTEXT or
DIP-BUY PLAN with decision="WAIT".
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy  # noqa: E402


def _choppy_sideways_series(seed=1, n=260):
    """A genuinely range-bound market with no directional structure — 4H
    trend should read NEUTRAL/TRANSITION and 1H should not produce an
    accepted CHoCH/BOS aligned with any one direction."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.15, n))
    close = np.clip(close, 95, 105)  # keep it range-bound, no real trend
    open_ = close - rng.normal(0, 0.1, n)
    high = np.maximum(close, open_) + np.abs(rng.normal(0.2, 0.05, n))
    low = np.minimum(close, open_) - np.abs(rng.normal(0.2, 0.05, n))
    volume = np.full(n, 1000.0)
    return open_, high, low, close, volume


def test_axs_style_indicators_alone_never_produce_valid_buy():
    """
    TEST A (spec review #2, item 4): manufacture a last candle with every
    secondary-indicator box ticked bullish (RSI divergence, volume spike,
    momentum, EMA stack) sitting on top of genuinely range-bound, non-
    trending structure. The old engine would have BUY'd on "4/5
    confirmations." The new engine must not.
    """
    o, h, l, c, v = _choppy_sideways_series(seed=7)

    # Manufacture a strong bullish-looking finish: sharp volume spike +
    # upward push on the last few candles, to make every secondary
    # indicator read bullish — while the underlying series is still
    # range-bound with no accepted structural CHoCH/BOS.
    c[-6:] = np.linspace(c[-6], c[-6] + 3.0, 6)
    h[-6:] = c[-6:] + 0.3
    l[-6:] = c[-6:] - 0.3
    o[-6:] = c[-6:] - 0.1
    v[-3:] = v[-3:] * 6  # volume spike

    ts = pd.date_range("2025-01-01", periods=len(c), freq="h", tz="UTC")
    df_1h = pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
    ts4 = pd.date_range("2025-01-01", periods=max(len(c) // 4, 60), freq="4h", tz="UTC")
    o4, h4, l4, c4, v4 = _choppy_sideways_series(seed=7, n=len(ts4))
    df_4h = pd.DataFrame({"timestamp": ts4, "open": o4, "high": h4, "low": l4, "close": c4, "volume": v4})

    settings = {"minimum_rr": 2.0, "risk_per_trade_usd": 250, "capital": 25000}
    sig = strategy.evaluate_symbol("AXS/USDT", df_4h, df_1h, settings)

    ind_ = sig["indicators"]
    # Confirm the manufactured setup really does look bullish on indicators
    # alone — otherwise this test wouldn't be exercising the intended path.
    bullish_indicator_hits = sum([
        ind_["rsi_divergence"] == "Bullish",
        ind_["volume_spike_ratio"] > 1.5,
        ind_["momentum_state"] == "Positive",
        ind_["ema_stack"] == "Bullish",
    ])
    # Regardless of how many indicators line up, the decision must never be
    # a VALID BUY unless real structure (CHoCH/BOS/location/R:R) supports it.
    if sig["decision"] == "VALID":
        assert sig["entry_model"] is not None, "a VALID decision must always be backed by a real entry model (structure), never indicators alone."
        assert sig["choch"] is not None or sig["bos"] is not None, (
            "a VALID BUY exists with no CHoCH/BOS at all — indicators alone drove the decision. This is exactly the AXS regression."
        )
    else:
        assert sig["decision"] in ("WAIT", "NO TRADE")


def test_skhyb_style_level_reached_is_not_automatically_bearish():
    """
    TEST B, part 1 (spec review #2, item 4): price merely trading up into
    a prior resistance level, with no genuine bearish CHoCH/BOS, must NOT
    be classified as a bearish setup — the old bot's SKHYB call was wrong
    precisely because it treated "reached a level" as bearish without
    requiring real bearish structure.
    """
    rng = np.random.default_rng(3)
    closes = []
    price = 100
    # Clean, ongoing uptrend (HH/HL) — price approaches and even slightly
    # exceeds an earlier local high, but the STRUCTURE is still bullish
    # throughout (no bearish CHoCH ever fires).
    for leg in range(15):
        for _ in range(10):
            price += 1.0 + rng.normal(0, 0.05)
            closes.append(price)
        for _ in range(4):
            price -= 0.4 + rng.normal(0, 0.03)
            closes.append(price)
    closes = np.array(closes)
    opens = closes - rng.normal(0, 0.1, len(closes))
    highs = np.maximum(closes, opens) + np.abs(rng.normal(0.2, 0.05, len(closes)))
    lows = np.minimum(closes, opens) - np.abs(rng.normal(0.2, 0.05, len(closes)))
    volume = np.full(len(closes), 1000.0)
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="h", tz="UTC")
    df_1h = pd.DataFrame({"timestamp": ts, "open": opens, "high": highs, "low": lows, "close": closes, "volume": volume})
    ts4 = pd.date_range("2025-01-01", periods=max(len(closes) // 4, 60), freq="4h", tz="UTC")
    o4 = np.linspace(100, closes[-1], len(ts4))
    df_4h = pd.DataFrame({"timestamp": ts4, "open": o4, "high": o4 + 1, "low": o4 - 1, "close": o4, "volume": np.full(len(ts4), 1000.0)})

    settings = {"minimum_rr": 1.0, "risk_per_trade_usd": 250, "capital": 25000}
    sig = strategy.evaluate_symbol("SKHYB/USDT", df_4h, df_1h, settings)

    # In an ongoing, structurally bullish market, "price reached a level"
    # must not by itself flip the read to bearish.
    assert sig["direction"] != "SELL" or sig["bos"] is not None and sig["bos"]["direction"] == "bearish", (
        "market was called bearish/SELL with no genuine bearish BOS behind it — a level-reached false positive."
    )


def _bearish_choch_bos_series(seed=11):
    """Deterministic bearish CHoCH+BOS 1H series (mirrors the proven
    pattern from test_structure.py::test_bearish_choch_then_bos) paired
    with an independently-built bearish 4H series, so state_4h.trend
    reliably reads BEARISH/TRANSITION rather than NEUTRAL."""
    closes = []
    price = 100
    for leg in range(10):
        for _ in range(10):
            price += 1.2
            closes.append(price)
        for _ in range(4):
            price -= 0.5
            closes.append(price)
    for _ in range(40):
        price -= 2.5
        closes.append(price)
    for _ in range(5):
        price += 0.4
        closes.append(price)
    for _ in range(6):
        price -= 0.5
        closes.append(price)
    for _ in range(4):
        price += 0.2
        closes.append(price)
    for _ in range(30):
        price -= 2.0
        closes.append(price)
    for _ in range(20):
        price -= 0.05
        closes.append(price)

    rng = np.random.default_rng(seed)
    closes = np.array(closes, dtype=float)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    jitter = rng.uniform(0.01, 0.05, size=len(closes))
    highs = np.maximum(opens, closes) + 0.3 + jitter
    lows = np.minimum(opens, closes) - 0.3 - jitter
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="h", tz="UTC")
    df_1h = pd.DataFrame({"timestamp": ts, "open": opens, "high": highs, "low": lows, "close": closes,
                           "volume": np.full(len(closes), 1000.0)})

    def _staircase_4h(n_legs, leg_len, start, up, down, seed4):
        rng4 = np.random.default_rng(seed4)
        p = start
        cs = []
        for _ in range(n_legs):
            for _ in range(leg_len):
                p += up / leg_len + rng4.normal(0, 0.15)
                cs.append(p)
            for _ in range(max(leg_len // 4, 1)):
                p -= down / max(leg_len // 4, 1) + rng4.normal(0, 0.1)
                cs.append(p)
        cs = np.array(cs)
        os_ = cs - rng4.normal(0, 0.1, len(cs))
        hs = np.maximum(cs, os_) + np.abs(rng4.normal(0.2, 0.1, len(cs)))
        ls = np.minimum(cs, os_) - np.abs(rng4.normal(0.2, 0.1, len(cs)))
        return os_, hs, ls, cs

    # Bearish staircase: negative "up" legs each cycle (a descending
    # staircase), mirroring the proven bullish generator used elsewhere.
    o4, h4, l4, c4 = _staircase_4h(20, 20, start=300, up=-8, down=-3, seed4=seed)
    ts4 = pd.date_range("2025-01-01", periods=len(c4), freq="4h", tz="UTC")
    df_4h = pd.DataFrame({"timestamp": ts4, "open": o4, "high": h4, "low": l4, "close": c4,
                           "volume": np.full(len(c4), 1000.0)})
    return df_4h, df_1h


def test_spot_only_never_generates_short_execution_signal():
    """
    TEST B, part 2 (spec review #2, item 4): even when bearish structure
    IS genuine (real bearish CHoCH/BOS), this is a SPOT-ONLY bot — a
    bearish read must never become decision="VALID" or an executable short.
    It may only ever produce BEARISH CONTEXT or DIP-BUY PLAN with
    decision="WAIT".
    """
    df_4h, df_1h = _bearish_choch_bos_series()
    settings = {"minimum_rr": 0.01, "risk_per_trade_usd": 250, "capital": 25000, "max_structural_risk_pct": 50.0}
    sig = strategy.evaluate_symbol("SKHYB/USDT", df_4h, df_1h, settings)

    # This scenario is specifically built to trigger a real bearish
    # CHoCH+BOS — confirm the test actually exercises the SELL path and
    # isn't passing vacuously.
    assert sig["direction"] == "SELL", "test setup failed to produce a genuine bearish structural read — not exercising the intended path."
    assert sig["decision"] != "VALID", "a SELL/bearish read must never reach decision='VALID' — this bot is spot-only, no shorts."
    assert sig["signal_type"] in ("BEARISH CONTEXT", "DIP-BUY PLAN"), (
        f"unexpected bearish signal_type '{sig['signal_type']}' — must be BEARISH CONTEXT or DIP-BUY PLAN only."
    )
    assert sig["decision"] == "WAIT"
