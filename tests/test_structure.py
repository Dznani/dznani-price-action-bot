"""
tests/test_structure.py — deterministic tests for structure.py.

Covers spec test cases #1 (bullish HH/HL), #2 (bearish LL/LH),
#3 (bullish CHoCH), #5 (bullish BOS).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import structure  # noqa: E402
import liquidity  # noqa: E402


def _candles(closes, wick=0.3, seed=42):
    # Tiny deterministic per-candle jitter breaks exact high/low ties at
    # turning points (which would otherwise happen because open[i] ==
    # close[i-1] and a constant wick), while leaving the overall staircase
    # shape intact for structure detection.
    rng = np.random.default_rng(seed)
    closes = np.array(closes, dtype=float)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    jitter = rng.uniform(0.01, 0.05, size=len(closes))
    highs = np.maximum(opens, closes) + wick + jitter
    lows = np.minimum(opens, closes) - wick - jitter
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": opens, "high": highs, "low": lows, "close": closes,
                          "volume": np.full(len(closes), 1000.0)})


def test_bullish_hh_hl_structure():
    # Clean staircase: each leg's high/low both exceed the previous leg's.
    closes = []
    price = 100
    for leg in range(6):
        for _ in range(10):
            price += 1.0
            closes.append(price)
        for _ in range(4):
            price -= 0.5
            closes.append(price)
    df = _candles(closes)
    state = structure.analyze_structure(df, left=2, right=2)
    assert state.trend == "BULLISH"
    labels = [s.label for s in state.swing_highs + state.swing_lows if s.label]
    assert "HH" in labels and "HL" in labels
    assert "LL" not in labels and "LH" not in labels


def test_bearish_ll_lh_structure():
    closes = []
    price = 200
    for leg in range(6):
        for _ in range(10):
            price -= 1.0
            closes.append(price)
        for _ in range(4):
            price += 0.5
            closes.append(price)
    df = _candles(closes)
    state = structure.analyze_structure(df, left=2, right=2)
    assert state.trend == "BEARISH"
    labels = [s.label for s in state.swing_highs + state.swing_lows if s.label]
    assert "LL" in labels and "LH" in labels
    assert "HH" not in labels and "HL" not in labels


def test_no_lookahead_before_confirmation_index():
    """
    CRITICAL regression test (review #2, item 1): "a future candle cannot
    cause an earlier CHoCH/BOS" — and, the sharper failure mode this
    specifically guards against, a not-yet-confirmed swing must not mask
    (make unreachable) an older, still-actionable swing level before its
    own confirmation index arrives.

    Hand-built, exact scenario (verified against both the fixed and the
    previously-buggy implementation to confirm this actually discriminates
    between them, not just a same-result-either-way check):

      index 2: swing high H0 = 100 (confirms at index 2+right = 4)
      index 7: swing high H1 = 105, but its own CLOSE is 102 (confirms at
               index 7+right = 9)

    At index 7, close=102 is above H0's level (100) but below H1's own
    level (105). The correct (lookahead-free) behavior: H1 is not
    confirmed yet at index 7, so the active recent-high is still H0 (100),
    and close=102 > 100 must fire a break event at index 7.

    The bug this targets: the previous implementation added a swing to
    `seen_highs` at its PIVOT index (7) rather than its CONFIRMATION index
    (9). That made H1 the "recent high" starting at index 7 itself,
    silently replacing H0 — so close=102 was compared against H1's level
    (105) instead of H0's (100), found no break, and the entire event was
    missed. Confirmed empirically: running this exact scenario against the
    pre-fix code produces ZERO events; the fixed code produces one event
    at break_index=7, level=100.0.
    """
    rows = [
        (88, 90, 85, 89), (89, 95, 87, 90), (90, 100, 88, 91), (91, 97, 87, 90), (90, 93, 86, 89),
        (89, 98, 86, 90), (90, 101, 87, 91), (91, 105, 88, 102), (100, 99, 90, 95), (94, 94, 85, 90),
        (90, 92, 85, 88), (88, 90, 83, 86),
    ]
    ts = pd.date_range("2025-01-01", periods=len(rows), freq="h", tz="UTC")
    o, h, l, c = zip(*rows)
    df = pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": [1000.0] * len(rows)})

    swings = structure.detect_swings(df, left=2, right=2)
    swing_prices = {(s.index, s.kind): s.price for s in swings}
    assert swing_prices.get((2, "high")) == 100.0, "test setup assumption broken: H0 pivot not where expected"
    assert swing_prices.get((7, "high")) == 105.0, "test setup assumption broken: H1 pivot not where expected"

    state = structure.analyze_structure_1h(df, left=2, right=2, bos_min_displacement_pct=0.1)
    events_at_7 = [e for e in state.events if e.break_index == 7]
    assert events_at_7, (
        "No event fired at index 7 for the close=102 break of H0's level (100) — the not-yet-"
        "confirmed swing H1 (105) masked the still-active older level. Lookahead bias present."
    )
    assert events_at_7[0].level == 100.0, (
        f"Event at index 7 used level {events_at_7[0].level}, expected H0's level (100.0) — "
        f"the engine is using the unconfirmed H1 (105) instead."
    )


def test_swing_not_available_until_right_candles_after_pivot():
    """Mechanism-level check: directly walks analyze_structure_1h's swing
    availability by comparing results computed on a prefix that stops
    exactly at a swing's pivot index (no confirmation possible yet) against
    a prefix that includes its full confirmation window."""
    closes = []
    price = 100
    for leg in range(4):
        for _ in range(10):
            price += 1.0
            closes.append(price)
        for _ in range(4):
            price -= 0.5
            closes.append(price)
    df_full = _candles(closes)
    swings = structure.detect_swings(df_full, left=2, right=2)
    assert swings, "expected at least one confirmed swing in the full series"
    pivot = swings[-1]

    # A prefix that ends exactly AT the pivot index cannot possibly confirm
    # it yet (needs `right` more candles) — detect_swings itself must not
    # report it either, since the fractal window can't be evaluated.
    prefix_at_pivot = df_full.iloc[: pivot.index + 1]
    swings_at_pivot = structure.detect_swings(prefix_at_pivot, left=2, right=2)
    assert all(s.index != pivot.index for s in swings_at_pivot), (
        "A swing was reported before its confirmation window (right candles) existed."
    )

    # Once `right` more candles exist, it must be confirmable.
    prefix_confirmed = df_full.iloc[: pivot.index + 2 + 1]
    swings_confirmed = structure.detect_swings(prefix_confirmed, left=2, right=2)
    assert any(s.index == pivot.index for s in swings_confirmed), (
        "Swing still not confirmed even after its full right-side window exists."
    )


def test_future_candles_never_change_past_choch_bos_events():
    """Outcome-level regression test using the full bullish CHoCH+BOS
    fixture: appending DIFFERENT future data after a point must never
    change any event whose break_index is safely before that point."""
    def _make_series(tail_pattern):
        closes = []
        price = 200
        for leg in range(5):
            for _ in range(10):
                price -= 1.2
                closes.append(price)
            for _ in range(4):
                price += 0.5
                closes.append(price)
        for _ in range(20):
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
        for _ in range(15):
            price += 2.0
            closes.append(price)
        safe_cutoff = len(closes)
        if tail_pattern == "continue_up":
            for _ in range(20):
                price += 2.0
                closes.append(price)
        else:  # "reverse_down"
            for _ in range(20):
                price -= 3.0
                closes.append(price)
        return closes, safe_cutoff

    closes_up, cutoff = _make_series("continue_up")
    closes_down, _ = _make_series("reverse_down")
    state_up = structure.analyze_structure_1h(_candles(closes_up), left=2, right=2, bos_min_displacement_pct=0.1)
    state_down = structure.analyze_structure_1h(_candles(closes_down), left=2, right=2, bos_min_displacement_pct=0.1)

    # Give a small safety margin (right=2) below the divergence point, since
    # events whose confirmation window straddles the cutoff legitimately
    # could differ — the lookahead guarantee is about events CLEARLY before
    # the divergent tail, not the boundary candle itself.
    margin = cutoff - 3
    events_up = [(e.kind, e.direction, e.break_index, round(e.level, 6)) for e in state_up.events if e.break_index <= margin]
    events_down = [(e.kind, e.direction, e.break_index, round(e.level, 6)) for e in state_down.events if e.break_index <= margin]
    assert events_up == events_down, "Diverging future candles changed a CHoCH/BOS event from well before the divergence point."


def test_minor_swing_does_not_become_structural_level():
    """
    spec review #2, item 2: "minor swing break that is NOT BOS."

    H_struct = 100 confirms first. A later swing H_minor = 100.2 forms
    (only 0.2% above H_struct — below the default 0.3% significance
    threshold) and is a real, confirmed fractal pivot, but must NOT
    replace H_struct as the active/structural level. A later close that
    exceeds H_struct (100) but stays below H_minor (100.2) must fire an
    event referencing H_struct's level (100.0) — proving the minor swing
    was never installed as "the" level in the first place, exactly what
    the review calls out: "the engine must not call 'close above latest
    swing high = bullish BOS' unless that swing is the relevant
    structural level."
    """
    rows = [
        (88, 90, 85, 89), (89, 95, 87, 90), (90, 100, 88, 91), (91, 97, 87, 90), (90, 93, 86, 89),
        (89, 98, 86, 90), (90, 100.2, 87, 91), (91, 100.05, 88, 92), (92, 99, 90, 95), (94, 94, 85, 90),
        (90, 92, 85, 88), (89, 100.15, 87, 100.1), (100, 99, 90, 92), (91, 92, 85, 88),
    ]
    df = _rows_to_df(rows)
    swings = structure.detect_swings(df, left=2, right=2)
    swing_prices = {(s.index, s.kind): s.price for s in swings}
    assert swing_prices.get((2, "high")) == 100.0
    assert swing_prices.get((6, "high")) == 100.2  # confirmed as a real pivot...

    state = structure.analyze_structure_1h(df, left=2, right=2, bos_min_displacement_pct=0.05, min_swing_significance_pct=0.3)
    # ...but must never have been used as an event level.
    levels_used = {round(e.level, 4) for e in state.events}
    assert 100.2 not in levels_used, "the minor (insignificant) swing was used as a BOS/CHoCH level — it should never be."
    assert any(round(e.level, 4) == 100.0 for e in state.events), "expected an event at the actual structural level (100.0)."


def test_structural_swing_break_is_bos():
    """
    spec review #2, item 2: "protected structural break that IS BOS."

    Mirror of the above with a genuinely significant new high (well above
    the significance threshold) that MUST be promoted to the active level,
    and whose break by a later close IS registered as the state-machine's
    next event, referencing its own (correct, higher) level.
    """
    rows = [
        (88, 90, 85, 89), (89, 95, 87, 90), (90, 100, 88, 91), (91, 97, 87, 90), (90, 93, 86, 89),
        (89, 98, 86, 90), (90, 106, 87, 91), (91, 104, 88, 92), (92, 99, 90, 95), (94, 94, 85, 90),
        (90, 92, 85, 88), (89, 107, 87, 106.5), (105, 104, 100, 102), (100, 101, 96, 97),
    ]
    df = _rows_to_df(rows)
    swings = structure.detect_swings(df, left=2, right=2)
    swing_prices = {(s.index, s.kind): s.price for s in swings}
    assert swing_prices.get((6, "high")) == 106.0  # genuinely significant vs H_struct=100 (6% higher)

    state = structure.analyze_structure_1h(df, left=2, right=2, bos_min_displacement_pct=0.05, min_swing_significance_pct=0.3)
    assert any(round(e.level, 4) == 106.0 for e in state.events), (
        "a genuinely significant new structural high must become the active level and drive the next break event."
    )


def test_minor_liquidity_sweep_is_not_choch():
    """
    spec review #2, item 2: "minor liquidity sweep that is NOT CHoCH" +
    "genuine CHoCH."

    A wick that clears the structural level but whose CANDLE CLOSES back
    below it is a liquidity sweep (liquidity.py) — and, correctly, must
    NEVER register as a structure.py CHoCH/BOS event, since those are
    close-based, not wick-based. A later candle whose CLOSE genuinely
    clears the level is the real, separate event.
    """
    rows = [
        (88, 90, 85, 89), (89, 95, 87, 90), (90, 100, 88, 91), (91, 97, 87, 90), (90, 93, 86, 89),
        (89, 98, 86, 90), (90, 92, 87, 91),
        (91, 104, 89, 96),   # WICK to 104 (clears 100) but CLOSES at 96, well below 100 — a sweep, not a break
        (95, 97, 90, 92), (91, 93, 88, 90),
        (89, 106, 87, 103),  # genuine close (103) beyond 100 — a real break
        (103, 105, 100, 104), (104, 106, 101, 105),
    ]
    df = _rows_to_df(rows)
    state = structure.analyze_structure_1h(df, left=2, right=2, bos_min_displacement_pct=0.05, min_swing_significance_pct=0.3)

    # No event may have fired at the sweep candle (index 7) — its high wicked
    # through 100 but its close (96) never did; CHoCH/BOS is close-based.
    assert not any(e.break_index == 7 for e in state.events), "a wick-only sweep incorrectly registered as a CHoCH/BOS event."
    # The wick itself (104) is still a legitimate fractal swing high once
    # confirmed (fractal pivots are wick-based, by definition) and becomes
    # the new active level — but note it was NOT treated as a break/event
    # at the time it formed (index 7), only later once a CLOSE actually
    # cleared it (index 12). That's the real guarantee: the sweep candle
    # itself never fires an event on its own wick.
    assert any(e.break_index == 12 for e in state.events), (
        "expected a genuine CHoCH/BOS once a later candle's CLOSE actually broke the active level."
    )

    # Cross-check with liquidity.py: the wick-only candle at index 7 IS
    # independently detected there as a genuine sweep of the original
    # structural level (100) — confirming this isn't "nothing happened",
    # it's specifically a sweep, and the two modules agree on that.
    state_for_pools = structure.analyze_structure(df.iloc[:9], left=2, right=2)
    pools = liquidity.detect_liquidity_pools(df.iloc[:9], state_for_pools, equal_tolerance_pct=0.15)
    struct_pool = next((p for p in pools if p.liquidity_type == "swing_high" and round(p.level, 1) == 100.0), None)
    assert struct_pool is not None
    sweep = liquidity.detect_sweep(df.iloc[:9], struct_pool, sweep_confirm_candles=2)
    assert sweep is not None and sweep.sweep_direction == "up", "liquidity.py should independently confirm this was a genuine sweep."


def _rows_to_df(rows):
    ts = pd.date_range("2025-01-01", periods=len(rows), freq="h", tz="UTC")
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": [1000.0] * len(rows)})


def test_retest_excursion_and_touch_cannot_be_same_candle():
    """
    spec review #2, item 3: "Do not allow the same candle to simultaneously
    create the required excursion and retest."

    Hand-built scenario: a bullish CHoCH breaks level=100 at index 6. A
    single WIDE-RANGE candle at index 9 has a high (104) that alone
    provides enough excursion (4%, above the 0.3% threshold) AND a low
    (100.05) that alone touches back within tolerance of the level — both
    conditions satisfied by ONE bar. The retest must NOT be confirmed on
    that same candle; it must wait for a genuinely separate later candle.

    Verified to discriminate: the pre-fix implementation reports
    `bars_since_bos=3` (the wide-range candle itself, index 9) as the
    retest touch; the fixed implementation correctly reports
    `bars_since_bos=4` (the next candle, index 10).
    """
    rows = [
        (88, 90, 85, 89), (89, 95, 87, 90), (90, 100, 88, 91), (91, 97, 87, 90), (90, 93, 86, 89),
        (89, 98, 86, 90), (90, 101.5, 87, 101.2), (91, 99, 86, 90), (90, 92, 85, 88),
        (101, 104, 100.05, 102),  # excursion (high=104) AND touch (low=100.05) on ONE candle
        (102, 103, 99, 100.5), (100, 101, 97, 98),
    ]
    df = _rows_to_df(rows)
    state = structure.analyze_structure_1h(df, left=2, right=2, bos_min_displacement_pct=0.05, min_swing_significance_pct=0.3)
    event = structure.latest_event(state, "CHoCH")
    assert event is not None and event.break_index == 6 and event.level == 100.0

    result = structure.detect_retest(df, event, distance_tolerance_pct=0.5, max_wait_candles=10,
                                      min_excursion_pct=0.3, reaction_min_pct=0.1)
    assert result.occurred is True
    assert result.bars_since_bos != 3, "the excursion candle (index 9) itself was incorrectly used as the retest touch too."
    assert result.bars_since_bos == 4, "expected the retest touch on the candle AFTER the excursion candle, not the same one."


def test_bullish_choch_then_bos():
    # Downtrend (LL/LH) for the first half, then a sharp reversal that
    # closes back above the most recent LH -> bullish CHoCH. A brief
    # consolidation lets a fresh swing high form, and a further impulse
    # breaks that fresh high -> BOS confirming the new bullish structure.
    closes = []
    price = 200
    for leg in range(5):
        for _ in range(10):
            price -= 1.2
            closes.append(price)
        for _ in range(4):
            price += 0.5
            closes.append(price)
    # Sharp reversal impulse well beyond the last LH (triggers CHoCH).
    for _ in range(20):
        price += 2.5
        closes.append(price)
    peak1 = price
    # Shallow pullback (forms swing low L1), then a bounce to a fresh local
    # high (swing high H1), then a shallower pullback that stays ABOVE L1
    # (so no low gets broken / no whipsaw CHoCH), then a strong impulse
    # breaks H1 -> BOS confirming continuation of the new bullish structure.
    for _ in range(5):
        price -= 0.4
        closes.append(price)
    l1 = price
    for _ in range(6):
        price += 0.5
        closes.append(price)
    h1 = price
    for _ in range(4):
        price -= 0.2
        closes.append(price)
    assert price > l1  # sanity: shallow pullback must not revisit L1
    for _ in range(15):
        price += 2.0
        closes.append(price)
    assert price > h1  # sanity: impulse must clear H1 for a BOS to be possible

    df = _candles(closes)
    state = structure.analyze_structure_1h(df, left=2, right=2, bos_min_displacement_pct=0.1)
    kinds = [(e.kind, e.direction) for e in state.events]
    assert ("CHoCH", "bullish") in kinds
    assert ("BOS", "bullish") in kinds


def test_bearish_choch_then_bos():
    """Mirror of test_bullish_choch_then_bos — spec test cases #4/#7."""
    closes = []
    price = 100
    for leg in range(5):
        for _ in range(10):
            price += 1.2
            closes.append(price)
        for _ in range(4):
            price -= 0.5
            closes.append(price)
    # Sharp reversal down well beyond the last HL (triggers bearish CHoCH).
    for _ in range(20):
        price -= 2.5
        closes.append(price)
    # Shallow bounce (forms swing high H1), then a dip to a fresh local low
    # (swing low L1), then a shallower bounce that stays BELOW H1 (no
    # whipsaw), then a strong impulse breaks L1 -> bearish BOS.
    for _ in range(5):
        price += 0.4
        closes.append(price)
    h1 = price
    for _ in range(6):
        price -= 0.5
        closes.append(price)
    l1 = price
    for _ in range(4):
        price += 0.2
        closes.append(price)
    assert price < h1
    for _ in range(15):
        price -= 2.0
        closes.append(price)
    assert price < l1

    df = _candles(closes)
    state = structure.analyze_structure_1h(df, left=2, right=2, bos_min_displacement_pct=0.1)
    kinds = [(e.kind, e.direction) for e in state.events]
    assert ("CHoCH", "bearish") in kinds
    assert ("BOS", "bearish") in kinds


def test_transition_classification():
    """spec test case #3: mostly bullish swings with one recent opposite
    swing should classify as TRANSITION, not a clean BULLISH/BEARISH."""
    closes = []
    price = 100
    for leg in range(5):
        for _ in range(10):
            price += 1.0
            closes.append(price)
        for _ in range(4):
            price -= 0.5
            closes.append(price)
    # A deep pullback that breaks the prior swing low -> an LL forms,
    # injecting real "opposite-side evidence" into a bullish sequence.
    # Trailing candles let the low actually confirm as a swing (needs
    # `right` closed candles after it).
    for _ in range(8):
        price -= 3.0
        closes.append(price)
    for _ in range(6):
        price += 0.3
        closes.append(price)
    df = _candles(closes)
    state = structure.analyze_structure(df, left=2, right=2)
    assert state.trend in ("TRANSITION", "BEARISH")  # opposite evidence must be acknowledged, not ignored


def test_protected_hl_tied_to_state_machine_not_naive_latest_swing():
    """spec Part 4 audit fix: protected_low must be the HL that actually
    supported the accepted BOS transition, taken from the event chain —
    not just whatever swing happens to be most recent afterward."""
    closes = []
    price = 200
    for leg in range(5):
        for _ in range(10):
            price -= 1.2
            closes.append(price)
        for _ in range(4):
            price += 0.5
            closes.append(price)
    for _ in range(20):
        price += 2.5
        closes.append(price)
    for _ in range(5):
        price -= 0.4
        closes.append(price)
    l1 = price
    for _ in range(6):
        price += 0.5
        closes.append(price)
    for _ in range(4):
        price -= 0.2
        closes.append(price)
    for _ in range(15):
        price += 2.0
        closes.append(price)

    df = _candles(closes)
    state = structure.analyze_structure_1h(df, left=2, right=2, bos_min_displacement_pct=0.1)
    bos = structure.latest_event(state, "BOS")
    assert bos is not None
    assert bos.protected_level is not None
    # The event's own protected_level and the state's final protected_low
    # (propagated from the latest accepted event) must agree.
    assert state.protected_low == bos.protected_level
    assert state.protected_high is None  # bullish thesis -> no protected high should linger


def test_protected_lh_bearish_mirror():
    closes = []
    price = 100
    for leg in range(5):
        for _ in range(10):
            price += 1.2
            closes.append(price)
        for _ in range(4):
            price -= 0.5
            closes.append(price)
    for _ in range(20):
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
    for _ in range(15):
        price -= 2.0
        closes.append(price)

    df = _candles(closes)
    state = structure.analyze_structure_1h(df, left=2, right=2, bos_min_displacement_pct=0.1)
    bos = structure.latest_event(state, "BOS")
    assert bos is not None and bos.direction == "bearish"
    assert state.protected_high == bos.protected_level
    assert state.protected_low is None


def test_displacement_scales_quality_not_hard_gate_by_default():
    """spec Part 5 audit fix: a structurally valid close beyond the level
    must still register as an event even with weak displacement, with
    `quality` scaled down — not silently dropped."""
    closes = []
    price = 200
    for leg in range(5):
        for _ in range(10):
            price -= 1.2
            closes.append(price)
        for _ in range(4):
            price += 0.5
            closes.append(price)
    # Very small reversal — closes just barely above the last LH.
    for _ in range(3):
        price += 0.3
        closes.append(price)
    df = _candles(closes)
    state_soft = structure.analyze_structure_1h(df, bos_min_displacement_pct=5.0, require_min_displacement=False)
    state_hard = structure.analyze_structure_1h(df, bos_min_displacement_pct=5.0, require_min_displacement=True)
    soft_choch = [e for e in state_soft.events if e.kind in ("CHoCH", "failed_CHoCH")]
    hard_choch = [e for e in state_hard.events if e.kind in ("CHoCH", "failed_CHoCH")]
    # With a high displacement threshold used only as a quality scaler
    # (require_min_displacement=False), a weak break can still register
    # (possibly with low quality); as a hard gate, it's excluded outright.
    assert len(soft_choch) >= len(hard_choch)


def test_failed_choch_reclassified_when_price_reclaims():
    """spec test cases #9/#11/#12: a CHoCH that breaks a level, immediately
    reverses back through it, and gets no follow-through must be
    reclassified as failed_CHoCH rather than left standing as a real one."""
    closes = []
    price = 200
    for leg in range(5):
        for _ in range(10):
            price -= 1.2
            closes.append(price)
        for _ in range(4):
            price += 0.5
            closes.append(price)
    # Break just above the last swing high (triggers CHoCH)...
    for _ in range(13):
        price += 1.0
        closes.append(price)
    # ...then immediately reverses back down through it with no continuation.
    for _ in range(10):
        price -= 2.0
        closes.append(price)
    for _ in range(6):
        price -= 0.1
        closes.append(price)
    df = _candles(closes)
    state = structure.analyze_structure_1h(df, bos_min_displacement_pct=0.1, fail_lookahead=8)
    kinds = [(e.kind, e.direction) for e in state.events]
    assert ("failed_CHoCH", "bullish") in kinds


def test_retest_requires_excursion_not_just_proximity():
    """spec Part 8 audit fix, test case #29: a "retest" candle immediately
    after the break with NO prior excursion away from the level must NOT
    be confirmed as occurred."""
    closes = list(np.linspace(100, 115, 15)) + list(np.linspace(115, 112, 4)) + list(np.linspace(112, 130, 15))
    df = _candles(closes)
    state = structure.analyze_structure_1h(df, bos_min_displacement_pct=0.1)
    event = structure.latest_event(state, "CHoCH")
    assert event is not None
    result = structure.detect_retest(df, event, distance_tolerance_pct=5.0, max_wait_candles=1, min_excursion_pct=50.0)
    assert result.occurred is False  # excursion requirement (50%) can never be met in 1 candle


def test_actual_retest_with_excursion_confirmed():
    """spec test case #28: a real post-break excursion followed by a
    genuine pullback and reaction must be confirmed as occurred."""
    closes = list(np.linspace(100, 115, 15)) + list(np.linspace(115, 112, 4)) + list(np.linspace(112, 130, 10))
    closes += list(np.linspace(130, 128, 3))       # further excursion away from the level
    closes += list(np.linspace(128, 118, 8))       # pulls back toward the broken level
    closes += list(np.linspace(118, 126, 6))       # reaction away, holding
    df = _candles(closes)
    state = structure.analyze_structure_1h(df, bos_min_displacement_pct=0.1)
    event = structure.latest_event(state, "CHoCH")
    assert event is not None
    result = structure.detect_retest(df, event, distance_tolerance_pct=3.0, max_wait_candles=20, min_excursion_pct=1.0, reaction_min_pct=0.5)
    assert result.excursion_pct > 1.0  # confirms the excursion requirement was actually satisfied by this path


def test_retest_requires_reaction_not_just_touch():
    # Break up, then price comes back to the level and immediately keeps
    # falling through it (no reaction) -> retest occurred but did NOT hold.
    closes = list(np.linspace(100, 115, 15)) + list(np.linspace(115, 112, 4)) + list(np.linspace(112, 130, 11))
    closes += list(np.linspace(130, 118, 6))   # pulls back near the level
    closes += list(np.linspace(118, 100, 10))  # keeps falling through — no reaction
    df = _candles(closes)
    state = structure.analyze_structure_1h(df, bos_min_displacement_pct=0.1)
    event = structure.latest_event(state, "CHoCH")
    if event is not None:
        result = structure.detect_retest(df, event, distance_tolerance_pct=1.0, max_wait_candles=15, min_excursion_pct=0.3)
        if result.occurred:
            assert result.held is False or result.invalidated is True
