"""
structure.py — deterministic market-structure engine for Dznani Signals Bot
(Price Action Edition).

Implements swing detection and structure classification for both the 4H
(context) and 1H (execution) timeframes, plus the 1H CHoCH/BOS engine.

DEFINITIONS (documented, deterministic — see spec sections 3/4/6/8)
---------------------------------------------------------------------
Swing high  : a candle whose high is the max of `left` candles before it
              and `right` candles after it (fractal pivot). It only becomes
              known once `right` candles have closed after it — i.e. a swing
              is never used before it is confirmable from already-closed
              candles. No lookahead: analyze_structure() only ever looks at
              df.iloc[:i+1] the caller passes in.
Swing low   : symmetric, using lows.
HH / HL / LH / LL : classified by comparing each new confirmed swing high
              to the previous confirmed swing high (HH if higher, LH if
              lower) and each new confirmed swing low to the previous
              confirmed swing low (HL if higher, LL if lower).
Protected high/low : the most recent swing point structurally protecting
              the current trend (the swing low an uptrend must hold above,
              or the swing high a downtrend must hold below).
CHoCH       : the first close beyond the swing point that was protecting
              the *opposite* trend while structure was still in that trend
              (e.g. bearish structure, price closes above the last LH ->
              bullish CHoCH). This is a possible transition signal only,
              NOT a confirmed reversal.
BOS         : a close beyond a valid swing high (bullish) / swing low
              (bearish) THAT IS ALREADY THE TREND'S OWN DIRECTION - i.e.
              continuation/confirmation once the new trend is underway, or
              the break that confirms a CHoCH was real ("CHoCH then BOS").
              Requires: a valid prior swing, a candle CLOSE beyond the
              level (not just a wick), and minimum displacement
              (bos_min_displacement_pct of settings) so noise doesn't
              count. Never evaluated on the still-open/last candle unless
              that candle is explicitly passed as closed by the caller.
Failed CHoCH/BOS : a CHoCH or BOS whose break candle closes back on the
              original side within `fail_lookahead` candles without any
              further confirmation swing forming beyond it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

Trend = Literal["BULLISH", "BEARISH", "TRANSITION", "NEUTRAL"]


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
@dataclass
class SwingPoint:
    index: int              # positional index into the df it was detected on
    timestamp: str
    price: float
    kind: str                # "high" | "low"
    label: Optional[str] = None  # "HH" | "HL" | "LH" | "LL" once classified
    is_structural: Optional[bool] = None  # set by analyze_structure_1h(): True = promoted to
                                            # the active protected/BOS-eligible level, False = a
                                            # minor/liquidity swing that never became "the" level.


@dataclass
class StructureEvent:
    kind: str                 # "CHoCH" | "BOS" | "failed_CHoCH" | "failed_BOS"
    direction: str             # "bullish" | "bearish"
    level: float               # price level that was broken
    break_index: int
    break_timestamp: str
    close_price: float
    displacement_pct: float
    previous_structure: str
    new_structure: str
    quality: float = 1.0       # 0-1, scales with displacement — a valid structural close still
                                # registers the event even with weak displacement (spec Part 5);
                                # displacement affects confidence/quality, not eligibility, unless
                                # require_min_displacement=True is explicitly set.
    protected_level: Optional[float] = None  # the OPPOSITE-side swing this event newly protects:
                                # for a bullish event, the candidate HL that must hold; for a
                                # bearish event, the candidate LH. This is what Part 4/6 require —
                                # the point that actually invalidates the resulting thesis, tied to
                                # the state-machine transition that created it, not just "latest swing".


@dataclass
class StructureState:
    trend: Trend
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    last_HH: Optional[SwingPoint]
    last_HL: Optional[SwingPoint]
    last_LH: Optional[SwingPoint]
    last_LL: Optional[SwingPoint]
    protected_high: Optional[float]
    protected_low: Optional[float]
    confidence: float
    events: List[StructureEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def _sp(s: Optional[SwingPoint]):
            return None if s is None else {"index": s.index, "price": s.price, "timestamp": s.timestamp, "label": s.label}

        return {
            "trend": self.trend,
            "swing_highs": [_sp(s) for s in self.swing_highs[-8:]],
            "swing_lows": [_sp(s) for s in self.swing_lows[-8:]],
            "last_HH": _sp(self.last_HH),
            "last_HL": _sp(self.last_HL),
            "last_LH": _sp(self.last_LH),
            "last_LL": _sp(self.last_LL),
            "protected_high": self.protected_high,
            "protected_low": self.protected_low,
            "confidence": round(self.confidence, 2),
            "events": [e.__dict__ for e in self.events[-6:]],
        }


# --------------------------------------------------------------------------- #
# Swing detection (fractal pivots, no lookahead beyond already-closed candles)
# --------------------------------------------------------------------------- #
def detect_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> List[SwingPoint]:
    """
    Fractal swing detection. A candle at position i is a swing high if its
    high is strictly the max over [i-left, i+right], and a swing low if its
    low is strictly the min over the same window. Only candles that have
    `right` fully-closed candles after them are eligible — the caller is
    expected to pass only closed candles (df excludes the still-forming bar).
    """
    highs = df["high"].values
    lows = df["low"].values
    ts = df["timestamp"]
    n = len(df)
    swings: List[SwingPoint] = []

    for i in range(left, n - right):
        window_h = highs[i - left : i + right + 1]
        window_l = lows[i - left : i + right + 1]

        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            swings.append(SwingPoint(index=i, timestamp=str(ts.iloc[i]), price=float(highs[i]), kind="high"))
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            swings.append(SwingPoint(index=i, timestamp=str(ts.iloc[i]), price=float(lows[i]), kind="low"))

    swings.sort(key=lambda s: s.index)
    return swings


# --------------------------------------------------------------------------- #
# HH/HL/LH/LL classification + trend state
# --------------------------------------------------------------------------- #
def _classify_swings(swings: List[SwingPoint]) -> None:
    """Mutates swings in place, labeling each relative to the previous
    swing of the same kind."""
    prev_high: Optional[SwingPoint] = None
    prev_low: Optional[SwingPoint] = None
    for s in swings:
        if s.kind == "high":
            if prev_high is not None:
                s.label = "HH" if s.price > prev_high.price else "LH"
            prev_high = s
        else:
            if prev_low is not None:
                s.label = "HL" if s.price > prev_low.price else "LL"
            prev_low = s


def analyze_structure(df: pd.DataFrame, left: int = 2, right: int = 2) -> StructureState:
    """
    4H-style structure engine (section 3). Classifies overall environment
    as BULLISH / BEARISH / TRANSITION / NEUTRAL from the most recent
    confirmed swings. Also usable standalone for the 1H swing set that
    structure_1h() builds CHoCH/BOS on top of.
    """
    swings = detect_swings(df, left, right)
    _classify_swings(swings)

    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]

    last_HH = next((s for s in reversed(highs) if s.label == "HH"), None)
    last_LH = next((s for s in reversed(highs) if s.label == "LH"), None)
    last_HL = next((s for s in reversed(lows) if s.label == "HL"), None)
    last_LL = next((s for s in reversed(lows) if s.label == "LL"), None)

    # Look at the last 4 labeled swings (2 highs + 2 lows worth of recency)
    recent_labels = [s.label for s in swings if s.label is not None][-4:]

    bullish_evidence = recent_labels.count("HH") + recent_labels.count("HL")
    bearish_evidence = recent_labels.count("LL") + recent_labels.count("LH")

    if len(recent_labels) < 2:
        trend: Trend = "NEUTRAL"
        confidence = 0.2
    elif bullish_evidence >= 3 and bearish_evidence == 0:
        trend = "BULLISH"
        confidence = 0.9
    elif bearish_evidence >= 3 and bullish_evidence == 0:
        trend = "BEARISH"
        confidence = 0.9
    elif bullish_evidence > bearish_evidence:
        trend = "BULLISH" if bearish_evidence == 0 else "TRANSITION"
        confidence = 0.6 if trend == "BULLISH" else 0.45
    elif bearish_evidence > bullish_evidence:
        trend = "BEARISH" if bullish_evidence == 0 else "TRANSITION"
        confidence = 0.6 if trend == "BEARISH" else 0.45
    else:
        trend = "TRANSITION" if (bullish_evidence and bearish_evidence) else "NEUTRAL"
        confidence = 0.35

    if trend == "BULLISH":
        protected_low = last_HL.price if last_HL else (lows[-1].price if lows else None)
        protected_high = None
    elif trend == "BEARISH":
        protected_high = last_LH.price if last_LH else (highs[-1].price if highs else None)
        protected_low = None
    else:
        protected_high = highs[-1].price if highs else None
        protected_low = lows[-1].price if lows else None

    return StructureState(
        trend=trend,
        swing_highs=highs,
        swing_lows=lows,
        last_HH=last_HH,
        last_HL=last_HL,
        last_LH=last_LH,
        last_LL=last_LL,
        protected_high=protected_high,
        protected_low=protected_low,
        confidence=confidence,
    )


# --------------------------------------------------------------------------- #
# 1H structure engine: adds CHoCH / BOS / failed-CHoCH / failed-BOS
# --------------------------------------------------------------------------- #
def analyze_structure_1h(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    bos_min_displacement_pct: float = 0.15,
    fail_lookahead: int = 6,
    require_min_displacement: bool = False,
    min_swing_significance_pct: float = 0.3,
) -> StructureState:
    """
    Builds on analyze_structure() and walks forward through swings in
    chronological order, replaying trend transitions to detect CHoCH and
    BOS events exactly as they would have appeared live (each event is
    evaluated only against candles at/after its break, never using future
    swings). See module docstring for definitions.

    FIXED (audit, spec Part 5): displacement now scales each event's
    `quality` (0-1) instead of gating whether the event is registered at
    all. A structurally valid close beyond the level is a CHoCH/BOS
    regardless of how big the move was — weak displacement should lower
    confidence, not silently make the engine blind to the break. Pass
    require_min_displacement=True to restore the old hard-filter behavior
    (event only registers if disp_pct >= bos_min_displacement_pct).

    FIXED (review #2, CRITICAL — lookahead bias): a fractal swing at pivot
    index i is not actually knowable until `right` further CLOSED candles
    have printed after it (that's the whole point of a fractal: you need
    the candles on both sides to confirm it's a local extreme). The
    previous version made a swing available to the CHoCH/BOS state machine
    the moment the loop reached `i` — i.e. at the exact candle where the
    swing sits, using information (the shape of the next `right` candles)
    that would not actually be known yet at that point in time. Fixed by
    only introducing a swing into `seen_highs`/`seen_lows` once the replay
    index reaches `s.index + right` (its confirmation index), never
    earlier. See test_structure.py::test_no_lookahead_before_confirmation_index.

    FIXED (review #2, item 2 — protected/structural vs minor swings): every
    fractal pivot used to be treated as "the" level for BOS/CHoCH purposes,
    including tiny 2-candle noise wiggles. That is exactly the anti-pattern
    the review calls out ("close above latest swing high = bullish BOS"
    without checking whether that swing is actually the relevant
    structural level). Fixed with an explicit minor-vs-structural
    classification at confirmation time: a newly confirmed swing only gets
    promoted to be the active "recent_high"/"recent_low" (the level BOS/
    CHoCH break tests are run against) if it differs from the CURRENT
    active level of the same kind by at least `min_swing_significance_pct`.
    A swing that fails this test is marked `is_structural=False`, stays
    visible in `state.swing_highs`/`swing_lows` (still useful to
    liquidity.py as a liquidity pool / minor swing) but never becomes "the"
    level — so a close beyond a minor swing's own price can never fire a
    BOS/CHoCH on its own; only a close beyond the actual structural level
    can. See test_structure.py::test_minor_swing_does_not_trigger_bos and
    test_structural_swing_break_is_bos.
    """
    state = analyze_structure(df, left, right)
    swings = sorted(state.swing_highs + state.swing_lows, key=lambda s: s.index)
    closes = df["close"].values
    n = len(df)

    # Walk candle-by-candle, maintaining a trend state machine and testing
    # each closed candle's close against the currently protected level. A
    # break in the trend's own direction while already in that trend is a
    # BOS (continuation); a break in the opposite direction is a CHoCH
    # (possible transition).
    events: List[StructureEvent] = []
    current_trend: Trend = "NEUTRAL"
    protected_high: Optional[float] = None
    protected_low: Optional[float] = None
    active_high: Optional[SwingPoint] = None  # the current STRUCTURAL recent high (BOS/CHoCH-eligible)
    active_low: Optional[SwingPoint] = None   # the current STRUCTURAL recent low
    # Swings become available at their CONFIRMATION index (pivot + right),
    # not their pivot index — this is the lookahead fix. Group by
    # confirmation index since more than one swing can confirm on the same
    # candle.
    by_confirmation_index: Dict[int, List[SwingPoint]] = {}
    for s in swings:
        by_confirmation_index.setdefault(s.index + right, []).append(s)
    already_broken: set = set()

    for i in range(n):
        for s in by_confirmation_index.get(i, []):
            if s.kind == "high":
                reference = active_high
                significant = reference is None or abs(s.price - reference.price) / reference.price * 100 >= min_swing_significance_pct
                s.is_structural = significant
                if significant:
                    active_high = s
            else:
                reference = active_low
                significant = reference is None or abs(s.price - reference.price) / reference.price * 100 >= min_swing_significance_pct
                s.is_structural = significant
                if significant:
                    active_low = s

        close = closes[i]

        # Determine current protected points from the active STRUCTURAL
        # swings only — a minor/insignificant swing never becomes "the"
        # level, so it can never be the thing a BOS/CHoCH break is
        # measured against.
        recent_high = active_high
        recent_low = active_low

        if recent_low is not None:
            level = recent_low.price
            key = ("low", recent_low.index)
            if close < level and key not in already_broken:
                disp_pct = abs(close - level) / level * 100
                if disp_pct >= bos_min_displacement_pct or not require_min_displacement:
                    kind = "BOS" if current_trend == "BEARISH" else "CHoCH"
                    events.append(StructureEvent(
                        kind=kind, direction="bearish", level=level, break_index=i,
                        break_timestamp=str(df["timestamp"].iloc[i]), close_price=float(close),
                        displacement_pct=round(disp_pct, 3),
                        previous_structure=current_trend, new_structure="BEARISH",
                        quality=round(min(1.0, disp_pct / bos_min_displacement_pct) if bos_min_displacement_pct > 0 else 1.0, 2),
                        protected_level=recent_high.price if recent_high else None,
                    ))
                    current_trend = "BEARISH"
                    protected_high = recent_high.price if recent_high else protected_high
                    already_broken.add(key)

        if recent_high is not None:
            level = recent_high.price
            key = ("high", recent_high.index)
            if close > level and key not in already_broken:
                disp_pct = abs(close - level) / level * 100
                if disp_pct >= bos_min_displacement_pct or not require_min_displacement:
                    kind = "BOS" if current_trend == "BULLISH" else "CHoCH"
                    events.append(StructureEvent(
                        kind=kind, direction="bullish", level=level, break_index=i,
                        break_timestamp=str(df["timestamp"].iloc[i]), close_price=float(close),
                        displacement_pct=round(disp_pct, 3),
                        previous_structure=current_trend, new_structure="BULLISH",
                        quality=round(min(1.0, disp_pct / bos_min_displacement_pct) if bos_min_displacement_pct > 0 else 1.0, 2),
                        protected_level=recent_low.price if recent_low else None,
                    ))
                    current_trend = "BULLISH"
                    protected_low = recent_low.price if recent_low else protected_low
                    already_broken.add(key)

    # Failed CHoCH/BOS: a CHoCH/BOS event with no follow-through event of
    # the same direction within fail_lookahead candles, and price closed
    # back beyond the broken level.
    final_events: List[StructureEvent] = []
    for idx, ev in enumerate(events):
        if ev.kind not in ("CHoCH", "BOS"):
            final_events.append(ev)
            continue
        window_end = min(ev.break_index + fail_lookahead, n - 1)
        reclaimed = False
        if ev.direction == "bullish":
            reclaimed = any(closes[j] < ev.level for j in range(ev.break_index + 1, window_end + 1))
        else:
            reclaimed = any(closes[j] > ev.level for j in range(ev.break_index + 1, window_end + 1))
        has_next_same_direction = any(
            e2.direction == ev.direction and e2.break_index > ev.break_index
            for e2 in events
        )
        if reclaimed and not has_next_same_direction:
            final_events.append(StructureEvent(
                kind=f"failed_{ev.kind}", direction=ev.direction, level=ev.level,
                break_index=ev.break_index, break_timestamp=ev.break_timestamp,
                close_price=ev.close_price, displacement_pct=ev.displacement_pct,
                previous_structure=ev.previous_structure, new_structure=ev.previous_structure,
                quality=ev.quality, protected_level=None,  # a failed event protects nothing — the prior structure's point still governs
            ))
        else:
            final_events.append(ev)

    state.events = final_events
    if final_events:
        last_confirmed = [e for e in final_events if e.kind in ("CHoCH", "BOS")]
        if last_confirmed:
            latest = last_confirmed[-1]
            state.trend = "BULLISH" if latest.direction == "bullish" else "BEARISH"
            state.confidence = round((0.85 if latest.kind == "BOS" else 0.6) * latest.quality, 2)
            # FIXED (audit, spec Part 4): tie protected_high/protected_low to the point the
            # state machine actually established at the latest accepted transition, instead of
            # leaving state.protected_* at the naive top-level swing classification computed by
            # analyze_structure() before any CHoCH/BOS replay happened.
            if latest.direction == "bullish" and latest.protected_level is not None:
                state.protected_low = latest.protected_level
                state.protected_high = None
            elif latest.direction == "bearish" and latest.protected_level is not None:
                state.protected_high = latest.protected_level
                state.protected_low = None
    return state


def latest_event(state: StructureState, kind: Optional[str] = None) -> Optional[StructureEvent]:
    events = state.events if kind is None else [e for e in state.events if e.kind == kind]
    return events[-1] if events else None


# --------------------------------------------------------------------------- #
# Retest engine (spec section 9)
# --------------------------------------------------------------------------- #
@dataclass
class RetestResult:
    occurred: bool
    held: bool
    distance_tolerance_pct: float
    reaction_confirmed: bool
    invalidated: bool
    retest_level: float
    direction: str
    bars_since_bos: Optional[int] = None
    excursion_pct: float = 0.0


def detect_retest(
    df: pd.DataFrame,
    event: StructureEvent,
    distance_tolerance_pct: float = 0.3,
    max_wait_candles: int = 12,
    reaction_min_pct: float = 0.2,
    min_excursion_pct: float = 0.3,
) -> RetestResult:
    """
    FIXED (audit, spec Part 8 — high priority): the previous version scanned
    for the first candle within `distance_tolerance_pct` of the broken
    level starting the candle right after the break. Since a break candle
    often closes only marginally beyond the level, this could call the very
    next candle a "retest" with zero actual post-BOS movement — exactly the
    bug the spec calls out ("distance < 0.5% is not sufficient").

    Correct sequence now enforced:
      BOS/CHoCH -> price must first move AWAY from the level by at least
      `min_excursion_pct` (the excursion) -> THEN a pullback bringing price
      back within `distance_tolerance_pct` of the level counts as the
      retest touch -> reaction over the next few candles decides whether it
      held. No excursion ever recorded => `occurred` stays False, matching
      spec's "fake near-BOS without excursion -> NOT retest" case.

    FIXED (review #2, item 3): the excursion check and the touch/return
    check used to both run against the SAME candle `i` within a single
    iteration — a single wide-range candle whose high provided enough
    excursion could, on that very same bar, also have a low that
    "touched" the level, satisfying both conditions at once with no real
    separate return ever happening. Fixed so the touch/return check only
    starts from the candle STRICTLY AFTER the one where excursion was
    first confirmed — excursion and retest can never be the same candle.
    See test_structure.py::test_retest_excursion_and_touch_cannot_be_same_candle.
    """
    n = len(df)
    start = event.break_index + 1
    end = min(event.break_index + 1 + max_wait_candles, n)
    level = event.level
    direction = event.direction
    if start >= n:
        return RetestResult(False, False, distance_tolerance_pct, False, False, level, direction)

    tol = level * distance_tolerance_pct / 100
    min_excursion = level * min_excursion_pct / 100
    highs = df["high"].values
    lows = df["low"].values
    close = df["close"].values

    excursion_achieved = False
    excursion_confirmed_at: Optional[int] = None
    max_excursion_pct = 0.0

    for i in range(start, end):
        if direction == "bullish":
            excursion_now = (highs[i] - level) / level * 100
        else:
            excursion_now = (level - lows[i]) / level * 100
        max_excursion_pct = max(max_excursion_pct, excursion_now)
        if not excursion_achieved and excursion_now * level / 100 >= min_excursion:
            excursion_achieved = True
            excursion_confirmed_at = i
            continue  # the candle that JUST achieved excursion cannot also be the touch — check touches from the next candle only

        if not excursion_achieved or (excursion_confirmed_at is not None and i <= excursion_confirmed_at):
            continue  # haven't moved away from the level yet, or this is still the excursion candle itself

        touched = (lows[i] <= level + tol) if direction == "bullish" else (highs[i] >= level - tol)
        if not touched:
            continue

        reaction_window_end = min(i + 3, n - 1)
        if direction == "bullish":
            invalidated = close[i] < level - tol
            reacted = any((close[j] - level) / level * 100 >= reaction_min_pct for j in range(i, reaction_window_end + 1))
        else:
            invalidated = close[i] > level + tol
            reacted = any((level - close[j]) / level * 100 >= reaction_min_pct for j in range(i, reaction_window_end + 1))

        return RetestResult(
            occurred=True, held=reacted and not invalidated, distance_tolerance_pct=distance_tolerance_pct,
            reaction_confirmed=reacted, invalidated=invalidated, retest_level=level, direction=direction,
            bars_since_bos=i - event.break_index, excursion_pct=round(max_excursion_pct, 3),
        )

    return RetestResult(False, False, distance_tolerance_pct, False, False, level, direction,
                         bars_since_bos=None, excursion_pct=round(max_excursion_pct, 3))
