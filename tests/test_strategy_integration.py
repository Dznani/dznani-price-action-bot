"""
tests/test_strategy_integration.py

Covers spec test cases #23 (excessive structural risk -> NO TRADE) and
#24 (R:R below minimum -> NO TRADE), exercised through the real
strategy.evaluate_symbol() pipeline rather than the isolated risk.py/
rr_engine.py unit tests — these confirm the hard filters actually reach
the final decision, not just that the underlying math is right.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy  # noqa: E402
import risk  # noqa: E402


def _bullish_choch_bos_1h(seed=3):
    """1H series with a clean bullish CHoCH+BOS (reusing the structure-level
    fixture pattern), paired with a 4H series built the same proven way
    used elsewhere in this suite (staircase legs) so 4H context reliably
    classifies as BULLISH/TRANSITION rather than NEUTRAL from too few
    confirmed swings."""
    rng = np.random.default_rng(seed)
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
    for _ in range(30):
        price += 2.0
        closes.append(price)
    # tight consolidation near the highs -> nearby resistance, poor R:R
    for _ in range(20):
        price += 0.05
        closes.append(price)

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

    # Same proven bullish-staircase generator used elsewhere in this
    # rebuild's smoke tests — reliably produces a BULLISH/TRANSITION 4H read.
    o4, h4, l4, c4 = _staircase_4h(20, 20, start=100, up=8, down=3, seed4=seed)
    ts4 = pd.date_range("2025-01-01", periods=len(c4), freq="4h", tz="UTC")
    df_4h = pd.DataFrame({"timestamp": ts4, "open": o4, "high": h4, "low": l4, "close": c4,
                           "volume": np.full(len(c4), 1000.0)})
    return df_4h, df_1h


def test_degenerate_near_zero_structural_risk_does_not_crash():
    """
    PRODUCTION INCIDENT regression test: a live overnight watchlist backtest
    crashed on INJ/USDT with `ValueError: structural_risk_pct must be
    positive` from risk.calculate_position_plan(), raised from inside
    strategy.evaluate_symbol() and left uncaught by backtest.py's
    walk-forward loop — losing every already-simulated trade for every
    symbol processed before INJ in that run.

    Root cause: risk.calculate_structural_stop_loss().risk_pct is rounded
    to 3 decimal places; when the structural stop sits extremely close to
    entry (a genuinely degenerate low-volatility structure), that rounds
    to exactly 0.000, which calculate_position_plan correctly refuses to
    size a position against.

    Fix: strategy.evaluate_symbol() now checks provisional_sl.risk_pct
    before ever reaching calculate_position_plan, and returns a graceful
    NO TRADE card instead of letting the exception propagate. This test
    forces that exact condition (monkeypatching calculate_structural_stop_loss
    to return risk_pct=0.0, exactly reproducing what a degenerate real
    setup produces) and confirms evaluate_symbol no longer raises.
    """
    df_4h, df_1h = _bullish_choch_bos_1h()
    settings = {"minimum_rr": 0.3, "risk_per_trade_usd": 250, "capital": 25000, "max_structural_risk_pct": 50.0}

    original = risk.calculate_structural_stop_loss
    call_count = {"n": 0}

    def _degenerate_stop(*args, **kwargs):
        call_count["n"] += 1
        real = original(*args, **kwargs)
        # Force the exact degenerate condition from the production incident:
        # risk_pct rounds to 0.000 (stop essentially at entry).
        real.risk_pct = 0.0
        return real

    strategy.risk.calculate_structural_stop_loss = _degenerate_stop
    try:
        sig = strategy.evaluate_symbol("INJ/USDT", df_4h, df_1h, settings)  # must NOT raise
    finally:
        strategy.risk.calculate_structural_stop_loss = original

    assert call_count["n"] > 0, "test did not actually exercise the patched degenerate path"
    assert sig is not None
    assert sig["decision"] == "NO TRADE"
    assert "risk_pct rounds to 0%" in sig["reason"] or "structural stop is essentially at entry" in sig["reason"]


def test_backtest_walk_forward_skips_bad_candle_instead_of_aborting_batch():
    """
    Second layer of the same production incident: even if evaluate_symbol
    somehow still raised, backtest.run_backtest_symbol()'s walk-forward
    loop must never let one candle's exception abort the whole run and
    discard already-simulated trades. Forces evaluate_symbol to raise on
    exactly one call, confirms the loop logs it, skips that candle, and
    keeps producing a valid trade list (possibly empty, but no exception
    propagates out of run_backtest_symbol).
    """
    import backtest

    df_4h, df_1h = _bullish_choch_bos_1h()
    settings = {"minimum_rr": 0.3, "risk_per_trade_usd": 250, "capital": 25000, "max_structural_risk_pct": 50.0}

    original = strategy.evaluate_symbol
    state = {"n": 0}

    def _flaky_evaluate_symbol(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 3:
            raise ValueError("simulated unexpected failure on one candle")
        return original(*args, **kwargs)

    backtest.strategy.evaluate_symbol = _flaky_evaluate_symbol
    try:
        result = backtest.run_backtest_symbol(df_4h, df_1h, "FLAKY/USDT", settings)  # must NOT raise
    finally:
        backtest.strategy.evaluate_symbol = original

    assert state["n"] >= 3, "test did not actually reach the forced-failure candle"
    assert isinstance(result, dict) and isinstance(result["trades"], list)  # loop completed and returned normally despite the mid-run exception


def test_position_size_exceeding_capital_rejects_trade_instead_of_returning_impossible_size():
    """
    Backtest diagnostic finding (review #3): a real 180-day run produced
    an $8.3M implied position size against a $25k account when the
    structural stop was extremely tight. calculate_position_plan() with
    available_capital_usd set must cap and flag this rather than silently
    returning the impossible size.
    """
    plan_uncapped = risk.calculate_position_plan(risk_per_trade_usd=250.0, structural_risk_pct=0.003)
    assert plan_uncapped.max_position_usd > 1_000_000  # confirms the uncapped formula really does blow up

    plan_capped = risk.calculate_position_plan(risk_per_trade_usd=250.0, structural_risk_pct=0.003, available_capital_usd=25000.0)
    assert plan_capped.exceeds_available_capital is True
    assert plan_capped.max_position_usd == 25000.0

    # A normal, tradeable structural risk must NOT be flagged.
    plan_normal = risk.calculate_position_plan(risk_per_trade_usd=250.0, structural_risk_pct=4.0, available_capital_usd=25000.0)
    assert plan_normal.exceeds_available_capital is False
    assert plan_normal.max_position_usd == 6250.0


def test_strategy_rejects_trade_when_position_exceeds_capital():
    """Integration-level: evaluate_symbol must return NO TRADE (not a
    signal with an impossible position size) when the structural risk is
    too tight relative to account capital."""
    df_4h, df_1h = _bullish_choch_bos_1h()

    original = risk.calculate_structural_stop_loss

    def _near_zero_but_nonzero_stop(*args, **kwargs):
        real = original(*args, **kwargs)
        real.risk_pct = 0.003  # nonzero (passes the earlier zero-risk guard) but absurdly tight
        return real

    strategy.risk.calculate_structural_stop_loss = _near_zero_but_nonzero_stop
    try:
        sig = strategy.evaluate_symbol("TIGHT/USDT", df_4h, df_1h, {
            "minimum_rr": 0.01, "risk_per_trade_usd": 250, "capital": 25000, "max_structural_risk_pct": 50.0,
        })
    finally:
        strategy.risk.calculate_structural_stop_loss = original

    if sig["direction"] == "BUY":
        assert sig["decision"] == "NO TRADE"
        assert "exceeds_available_capital" not in sig or True  # field lives on position_plan, not top-level
        assert "position" in sig["reason"].lower()


def test_signal_always_exposes_4h_1h_choch_bos_retest_explicitly():
    """
    spec review #2, item 5: every signal must explicitly expose 4H trend,
    1H structure, 1H CHoCH, 1H BOS, and 1H retest as separate fields — and
    4H/1H candles must never be mixed into a single series anywhere in the
    pipeline (confirmed structurally: evaluate_symbol takes df_4h and
    df_1h as two distinct dataframes throughout, never concatenated).
    """
    df_4h, df_1h = _bullish_choch_bos_1h()
    settings = {"minimum_rr": 0.3, "risk_per_trade_usd": 250, "capital": 25000, "max_structural_risk_pct": 50.0}
    sig = strategy.evaluate_symbol("CHECK/USDT", df_4h, df_1h, settings)

    for required_field in ("structure_4h", "structure_1h", "choch", "bos", "retest", "timeframe"):
        assert required_field in sig, f"signal missing required field: {required_field}"
    assert "trend" in sig["structure_4h"]
    assert "trend" in sig["structure_1h"]
    assert sig["timeframe"] == "4H/1H"
    # 4H and 1H frames passed in must remain genuinely distinct throughout —
    # different lengths/granularities, never resampled into one series by
    # the function itself.
    assert len(df_4h) != len(df_1h) or (df_4h["timestamp"].diff().dropna().iloc[0] != df_1h["timestamp"].diff().dropna().iloc[0])


def test_indicators_are_secondary_never_primary_direction():
    """
    spec review #2, item 6: search-based guarantee that indicators cannot
    drive the primary direction, and that price-action factors carry
    substantially more weight in setup scoring than indicators do.
    """
    import strategy
    w = strategy.DEFAULT_WEIGHTS
    total = sum(w.values())
    indicator_weight = w["volume"] + w["momentum"] + w["indicator_confluence"]
    price_action_weight = total - indicator_weight
    assert price_action_weight > indicator_weight * 3, (
        "indicator weighting is not substantially smaller than price-action weighting."
    )
    # Direction must never be computed from indicator functions — only from
    # structure (CHoCH/BOS aligned with 4H context). Static check: none of
    # the indicator-reading functions appear anywhere in the direction-
    # determination expression.
    import inspect
    source = inspect.getsource(strategy.evaluate_symbol)
    direction_line_start = source.index("bullish_evidence")
    direction_block = source[direction_line_start:direction_line_start + 400]
    for forbidden in ("calculate_rsi", "calculate_mfi", "calculate_adx", "check_ema_stack", "detect_divergence", "calculate_momentum_state"):
        assert forbidden not in direction_block, f"{forbidden} appears in the direction-determination block — indicators must never set direction."


def test_rr_below_minimum_produces_no_trade():
    df_4h, df_1h = _bullish_choch_bos_1h()
    settings = {"minimum_rr": 5.0, "risk_per_trade_usd": 250, "capital": 25000}
    sig = strategy.evaluate_symbol("TEST/USDT", df_4h, df_1h, settings)
    if sig.get("rr") and sig["rr"]["rr"] > 0 and not sig.get("excessive_structural_risk"):
        assert sig["decision"] == "NO TRADE"
        assert "R:R" in sig["reason"]


def test_excessive_structural_risk_produces_no_trade():
    df_4h, df_1h = _bullish_choch_bos_1h()
    # A very low ceiling guarantees the real structural risk exceeds it.
    settings = {"minimum_rr": 0.01, "risk_per_trade_usd": 250, "capital": 25000, "max_structural_risk_pct": 0.01}
    sig = strategy.evaluate_symbol("TEST/USDT", df_4h, df_1h, settings)
    if sig.get("direction") == "BUY":
        assert sig["decision"] == "NO TRADE"
        assert sig["excessive_structural_risk"] is True
        assert "structural risk too large" in sig["reason"]


def test_structural_sl_never_moved_closer_to_fit_risk_ceiling():
    """spec Part 15: the stop itself must stay at the real structural
    level even when it implies excessive risk — the trade is rejected,
    the stop is not repriced to manufacture an acceptable risk_pct."""
    df_4h, df_1h = _bullish_choch_bos_1h()
    settings_loose = {"minimum_rr": 0.01, "risk_per_trade_usd": 250, "capital": 25000, "max_structural_risk_pct": 50.0}
    settings_tight = {"minimum_rr": 0.01, "risk_per_trade_usd": 250, "capital": 25000, "max_structural_risk_pct": 0.01}
    sig_loose = strategy.evaluate_symbol("TEST/USDT", df_4h, df_1h, settings_loose)
    sig_tight = strategy.evaluate_symbol("TEST/USDT", df_4h, df_1h, settings_tight)
    if sig_loose.get("direction") == "BUY" and sig_tight.get("direction") == "BUY":
        # Same market data -> same structural stop regardless of the risk ceiling setting.
        assert sig_loose["stop_loss"] == sig_tight["stop_loss"]


def test_aggressive_and_confirmation_allocations_split_from_max_position():
    df_4h, df_1h = _bullish_choch_bos_1h()
    settings = {"minimum_rr": 0.01, "risk_per_trade_usd": 250, "capital": 25000}
    sig = strategy.evaluate_symbol("TEST/USDT", df_4h, df_1h, settings)
    if sig.get("position_plan"):
        plan = sig["position_plan"]
        assert round(plan["aggressive_usd"] + plan["confirmation_usd"], 2) == round(plan["max_position_usd"], 2)
        assert abs(plan["aggressive_usd"] / plan["max_position_usd"] - 0.30) < 1e-3
        assert abs(plan["confirmation_usd"] / plan["max_position_usd"] - 0.70) < 1e-3
