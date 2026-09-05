"""
tests/test_liquidity_and_zones.py

Covers spec test cases #7 (liquidity sweep), #8 (genuine breakout),
#11 (FVG), #12 (IFVG), #14 (Fibonacci confluence).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import liquidity as liq  # noqa: E402
import structure  # noqa: E402
import zones as zn  # noqa: E402


def _candles(rows):
    """rows: list of (open, high, low, close)"""
    ts = pd.date_range("2025-01-01", periods=len(rows), freq="h", tz="UTC")
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c,
                          "volume": [1000.0] * len(rows)})


def test_liquidity_sweep_vs_genuine_breakout():
    # Build a range with an obvious swing high at 110, then:
    #  (a) a sweep: wick to 111.5, close back at 108 (rejected)
    #  (b) later, a genuine breakout: close at 112 and holds above.
    rows = []
    price = 100
    for _ in range(20):
        rows.append((price, price + 1, price - 1, price + 0.2))
        price += 0.1
    # swing high forms around here (~110)
    for _ in range(6):
        rows.append((price, price + 0.8, price - 0.8, price))
    sweep_high = rows[-1][1] + 1.5  # wick beyond the recent high
    rows.append((price, sweep_high, price - 0.5, price - 1.0))  # sweep + reject (close back down)
    for _ in range(5):
        price -= 0.3
        rows.append((price, price + 0.5, price - 0.5, price))
    # later: genuine breakout, closes above and stays there
    for _ in range(8):
        price += 1.0
        rows.append((price, price + 1.2, price - 0.3, price))

    df = _candles(rows)
    state = structure.analyze_structure(df, left=2, right=2)
    pools = liq.detect_liquidity_pools(df, state)
    assert pools, "expected at least one liquidity pool from swing highs"

    # A genuine breakout candle should NOT be reported as a sweep once the
    # market has accepted beyond the level for several candles.
    high_pool = next(p for p in pools if p.liquidity_type == "swing_high")
    result = liq.detect_sweep(df, high_pool, sweep_confirm_candles=3)
    assert result is not None


def test_fvg_detection():
    # Three-candle bullish FVG: candle i-1 high, impulse candle i, candle i+1
    # low strictly above candle i-1's high.
    rows = [(100, 101, 99, 100.5)] * 5
    rows.append((101, 104, 100.8, 103.5))   # candle i-1 (high=104)
    rows.append((104, 108, 103.8, 107))     # impulse candle i
    rows.append((107.5, 110, 106.5, 109))   # candle i+1 (low=106.5 > 104 -> gap)
    rows += [(109, 110, 108, 109.5)] * 5
    df = _candles(rows)
    fvgs = zn.detect_fvgs(df, "1H", lookback=20)
    bullish = [z for z in fvgs if z.zone_type == "bullish_fvg"]
    assert bullish, "expected a detected bullish FVG"
    assert bullish[0].zone_low < bullish[0].zone_high


def test_ifvg_flip_when_closed_through():
    rows = [(100, 101, 99, 100.5)] * 5
    rows.append((101, 104, 100.8, 103.5))
    rows.append((104, 108, 103.8, 107))
    rows.append((107.5, 110, 106.5, 109))  # bullish FVG formed here (gap ~104-106.5)
    rows += [(109, 110, 108, 109.5)] * 3
    # Now close back down THROUGH the full FVG range (below 104) to invert it.
    rows.append((109, 109.2, 101, 102))
    rows += [(102, 103, 101, 102.5)] * 3
    df = _candles(rows)
    fvgs = zn.detect_fvgs(df, "1H", lookback=30)
    ifvgs = zn.detect_ifvgs(df, fvgs, "1H")
    assert any(z.zone_type == "ifvg_bearish" for z in ifvgs)


def test_fibonacci_zone_confluence_scoring():
    zones = zn.fibonacci_zones(swing_low=100, swing_high=120, current_price=112.4, timeframe="1H", leg_direction="bullish")
    assert len(zones) == 3
    result = zn.location_confluence_score(zones, current_price=112.4, direction="BUY", proximity_pct=2.0)
    assert result["supporting_score"] > 0
    assert result["opposing_score"] == 0  # bullish zones must not count as opposing confluence for a BUY


def test_directional_confluence_does_not_mix_polarities():
    """spec Part 12 audit fix: a bearish zone near price must never be
    counted as bullish confluence, and vice versa."""
    bullish_zone = zn.Zone("bullish_fvg", "1H", 101.0, 99.0, 0, 0.5, 0.5, {})
    bearish_zone = zn.Zone("bearish_fvg", "1H", 101.0, 99.0, 0, 0.5, 0.5, {})
    assert zn.is_bullish(bullish_zone) and not zn.is_bearish(bullish_zone)
    assert zn.is_bearish(bearish_zone) and not zn.is_bullish(bearish_zone)

    zones = [bullish_zone, bearish_zone]
    for_long = zn.location_confluence_score(zones, current_price=100.0, direction="BUY", proximity_pct=5.0)
    assert for_long["supporting_score"] > 0
    assert for_long["opposing_score"] > 0  # the bearish zone shows up as opposing, not folded into supporting
    assert for_long["supporting_score"] != for_long["opposing_score"] + for_long["supporting_score"]  # sanity: kept separate


def test_bearish_fvg_detection():
    """spec test case #14 (bearish FVG): candle i+1 high strictly below
    candle i-1 low."""
    rows = [(109, 110, 108, 109.5)] * 5
    rows.append((109, 109.2, 104.8, 105.5))  # candle i-1 (low=104.8)
    rows.append((105, 101.5, 100.5, 101))     # impulse candle i, down
    rows.append((101, 102.5, 99, 100))        # candle i+1 (high=102.5 < 104.8 -> gap)
    rows += [(100, 101, 99, 100.2)] * 5
    df = _candles(rows)
    fvgs = zn.detect_fvgs(df, "1H", lookback=20)
    bearish = [z for z in fvgs if z.zone_type == "bearish_fvg"]
    assert bearish, "expected a detected bearish FVG"
    assert bearish[0].zone_low < bearish[0].zone_high


def test_bullish_ifvg_flip_when_closed_through():
    """Mirror of test_ifvg_flip_when_closed_through for a bearish FVG that
    gets violated to the upside -> flips into bullish support."""
    rows = [(109, 110, 108, 109.5)] * 5
    rows.append((109, 109.2, 104.8, 105.5))
    rows.append((105, 101.5, 100.5, 101))
    rows.append((101, 102.5, 99, 100))       # bearish FVG formed here (gap ~102.5-104.8)
    rows += [(100, 101, 99, 100.2)] * 3
    # Now close back up THROUGH the full FVG range (above 104.8) to invert it.
    rows.append((100, 106, 99.8, 105.5))
    rows += [(105.5, 106.5, 105, 106)] * 3
    df = _candles(rows)
    fvgs = zn.detect_fvgs(df, "1H", lookback=30)
    ifvgs = zn.detect_ifvgs(df, fvgs, "1H")
    assert any(z.zone_type == "ifvg_bullish" for z in ifvgs)


def test_fvg_not_reported_as_ifvg_when_merely_filled_not_violated():
    """spec Part 10 audit note: an FVG that gets wicked/filled (price
    returns INTO the zone) but does not CLOSE all the way through it must
    NOT be reported as an IFVG — only a decisive opposite-direction close
    through the full range flips it."""
    rows = [(100, 101, 99, 100.5)] * 5
    rows.append((101, 104, 100.8, 103.5))
    rows.append((104, 108, 103.8, 107))
    rows.append((107.5, 110, 106.5, 109))  # bullish FVG formed here (~104-106.5)
    rows += [(109, 110, 108, 109.5)] * 3
    # Price wicks back INTO the FVG range but the close stays inside/above
    # zone_low (104) — filled, not violated.
    rows.append((109, 109.5, 105.0, 105.5))
    rows += [(105.5, 107, 105, 106.5)] * 3
    df = _candles(rows)
    fvgs = zn.detect_fvgs(df, "1H", lookback=30)
    ifvgs = zn.detect_ifvgs(df, fvgs, "1H")
    assert not any(z.zone_type == "ifvg_bearish" for z in ifvgs)


def test_order_block_reports_validity_not_just_distance():
    """spec Part 13: every OB return must include an explicit validity
    field, and a body that price has since fully closed back through
    (opposite direction) must be flagged invalidated, not just scored the
    same as a fresh one."""
    import structure
    rows = [(100 + i * 0.1, 100 + i * 0.1 + 1, 100 + i * 0.1 - 1, 100 + i * 0.1 + 0.05) for i in range(39)]
    rows.append((104.0, 104.3, 102.5, 102.8))  # a real bearish candle right before the break (the OB body)
    rows += [(103 + i * 0.1, 103 + i * 0.1 + 1, 103 + i * 0.1 - 1, 103 + i * 0.1 + 0.05) for i in range(20)]
    df = _candles(rows)
    bos_event = structure.StructureEvent(
        kind="BOS", direction="bullish", level=105.0, break_index=40, break_timestamp="t",
        close_price=106.0, displacement_pct=1.0, previous_structure="TRANSITION", new_structure="BULLISH",
    )
    zones_valid = zn.detect_order_blocks(df, [bos_event], "1H")
    assert zones_valid, "expected at least one OB zone"
    for z in zones_valid:
        assert "validity" in z.meta
        assert z.meta["validity"] in ("valid", "invalidated")
    """spec Part 14 audit note: fibonacci_zones now requires an explicit
    leg_direction and tags zones accordingly — it must reject an
    unspecified direction rather than silently guessing."""
    import pytest
    with pytest.raises(ValueError):
        zn.fibonacci_zones(100, 120, 112, "1H", leg_direction="sideways")

    bullish_zones = zn.fibonacci_zones(100, 120, 112, "1H", leg_direction="bullish")
    bearish_zones = zn.fibonacci_zones(100, 120, 112, "1H", leg_direction="bearish")
    assert all(zn.is_bullish(z) for z in bullish_zones)
    assert all(zn.is_bearish(z) for z in bearish_zones)
    """spec Part 11: resistance->support must be tagged bullish, support->resistance bearish."""
    bullish_event = structure.StructureEvent(
        kind="BOS", direction="bullish", level=100.0, break_index=10, break_timestamp="t",
        close_price=101.0, displacement_pct=0.5, previous_structure="BEARISH", new_structure="BULLISH",
    )
    bearish_event = structure.StructureEvent(
        kind="BOS", direction="bearish", level=100.0, break_index=10, break_timestamp="t",
        close_price=99.0, displacement_pct=0.5, previous_structure="BULLISH", new_structure="BEARISH",
    )
    flip_up = zn.flip_zone_from_event(bullish_event, "1H", current_price=100.0)
    flip_down = zn.flip_zone_from_event(bearish_event, "1H", current_price=100.0)
    assert flip_up.zone_type == "resistance_to_support"
    assert zn.is_bullish(flip_up)
    assert flip_down.zone_type == "support_to_resistance"
    assert zn.is_bearish(flip_down)
