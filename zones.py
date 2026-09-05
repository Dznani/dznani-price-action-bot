"""
zones.py — location / confluence engine (spec section 7).

Every detector returns zones with: zone high/low, type, timeframe,
freshness, distance from current price, and a confluence score. None of
these are claimed to be "institutional" with certainty — they are
technical heuristics, documented below.

Order Block heuristic (documented): the last opposite-colored candle
immediately before a displacement move that produced a BOS. E.g. for a
bullish OB: the last bearish (red) candle before the impulsive bullish
candle(s) that broke structure. Zone = that candle's open-close body.

FVG heuristic: classic 3-candle imbalance. Bullish FVG when candle[i-1]'s
low is above candle[i+1]'s high is WRONG — correct definition used here:
bullish FVG = candle[i+1].low > candle[i-1].high (gap between wick i-1 and
wick i+1, candle i is the impulse candle in the middle). Symmetric for
bearish.

IFVG: a previously-identified FVG whose full range gets closed through
(not just wicked) by a later candle — it then "flips" and is offered as a
zone in the opposite role.

Directional confluence (spec Part 12, audit fix): every Zone now carries an
explicit `polarity` ("bullish" | "bearish" | "neutral"). Fibonacci zones are
tagged by the direction of the impulse leg they were built from — a fib
zone is not inherently bullish or bearish, but a fib level of a *bullish*
leg used as long support is bullish confluence, so `fibonacci_zones()`
requires an explicit `leg_direction` argument now. `location_confluence_score()`
takes a `direction` and only sums zones whose polarity matches — opposing
zones are never silently added into the same total as supporting ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

FIB_LEVELS_DEFAULT = [0.382, 0.5, 0.618]

_BULLISH_TYPES = {"bullish_fvg", "ifvg_bullish", "bullish_ob", "resistance_to_support"}
_BEARISH_TYPES = {"bearish_fvg", "ifvg_bearish", "bearish_ob", "support_to_resistance"}


@dataclass
class Zone:
    zone_type: str
    timeframe: str
    zone_high: float
    zone_low: float
    freshness_index: int          # candles since formed (0 = just formed)
    distance_from_price_pct: float
    confluence_score: float
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["polarity"] = self.polarity
        return d

    @property
    def polarity(self) -> str:
        if self.zone_type in _BULLISH_TYPES or self.zone_type.startswith("fib_bullish"):
            return "bullish"
        if self.zone_type in _BEARISH_TYPES or self.zone_type.startswith("fib_bearish"):
            return "bearish"
        return "neutral"


def is_bullish(zone: "Zone") -> bool:
    """spec Part 11 — explicit helper, unit-tested directly rather than
    only exercised indirectly through scoring."""
    return zone.polarity == "bullish"


def is_bearish(zone: "Zone") -> bool:
    return zone.polarity == "bearish"


# --------------------------------------------------------------------------- #
# Fibonacci
# --------------------------------------------------------------------------- #
def fibonacci_zones(
    swing_low: float,
    swing_high: float,
    current_price: float,
    timeframe: str,
    leg_direction: str,
    levels: Optional[List[float]] = None,
    band_pct: float = 0.15,
) -> List[Zone]:
    """
    FIXED (audit, spec Part 14 + Part 12): `leg_direction` ("bullish" |
    "bearish") must now be passed explicitly and identifies which
    structural impulse leg produced swing_low/swing_high — a fib level is
    only bullish confluence (support for a long) when it's a retracement of
    a genuinely bullish leg, and bearish confluence for a bearish leg. The
    caller (strategy.py) is responsible for actually passing the relevant
    structural leg (spec: "swing_low -> swing_high" for bullish, "swing_high
    -> swing_low" for bearish), not an arbitrary max/min of recent candles.
    """
    if leg_direction not in ("bullish", "bearish"):
        raise ValueError("leg_direction must be 'bullish' or 'bearish'")
    levels = levels or FIB_LEVELS_DEFAULT
    rng = swing_high - swing_low
    zones = []
    if rng <= 0:
        return zones
    for lvl in levels:
        price = swing_high - rng * lvl
        band = price * band_pct / 100
        dist_pct = abs(current_price - price) / current_price * 100
        zones.append(Zone(
            zone_type=f"fib_{leg_direction}_{lvl}",
            timeframe=timeframe,
            zone_high=round(price + band, 8),
            zone_low=round(price - band, 8),
            freshness_index=0,
            distance_from_price_pct=round(dist_pct, 3),
            confluence_score=0.4 if lvl in (0.5, 0.618) else 0.3,
            meta={"level": lvl, "swing_low": swing_low, "swing_high": swing_high, "leg_direction": leg_direction},
        ))
    return zones


# --------------------------------------------------------------------------- #
# Fair Value Gaps (three-candle definition)
# --------------------------------------------------------------------------- #
def detect_fvgs(df: pd.DataFrame, timeframe: str, lookback: int = 80) -> List[Zone]:
    n = len(df)
    start = max(1, n - lookback)
    zones: List[Zone] = []
    highs, lows, close = df["high"].values, df["low"].values, df["close"].values
    current_price = float(close[-1])

    for i in range(start, n - 1):
        # bullish FVG: gap between candle i-1's high and candle i+1's low
        if lows[i + 1] > highs[i - 1]:
            zone_low, zone_high = highs[i - 1], lows[i + 1]
            zones.append(_make_fvg_zone("bullish_fvg", timeframe, zone_low, zone_high, i, n, current_price))
        # bearish FVG: gap between candle i-1's low and candle i+1's high
        if lows[i - 1] > highs[i + 1]:
            zone_low, zone_high = highs[i + 1], lows[i - 1]
            zones.append(_make_fvg_zone("bearish_fvg", timeframe, zone_low, zone_high, i, n, current_price))
    return zones


def _make_fvg_zone(kind, timeframe, zone_low, zone_high, formed_idx, n, current_price) -> Zone:
    dist_pct = min(abs(current_price - zone_low), abs(current_price - zone_high)) / current_price * 100
    freshness = n - 1 - (formed_idx + 1)
    score = 0.5 if freshness < 20 else 0.3
    return Zone(
        zone_type=kind, timeframe=timeframe, zone_high=round(zone_high, 8), zone_low=round(zone_low, 8),
        freshness_index=freshness, distance_from_price_pct=round(dist_pct, 3),
        confluence_score=score, meta={"formed_index": formed_idx},
    )


def detect_ifvgs(df: pd.DataFrame, fvgs: List[Zone], timeframe: str) -> List[Zone]:
    """An FVG is 'inverted' (IFVG) once a later CLOSE fully clears through
    its range (not just a wick) — it then flips role: a bullish FVG that
    gets closed-through becomes bearish resistance, and vice versa."""
    close = df["close"].values
    n = len(df)
    current_price = float(close[-1])
    out: List[Zone] = []
    for fvg in fvgs:
        formed_idx = fvg.meta["formed_index"]
        for j in range(formed_idx + 2, n):
            if fvg.zone_type == "bullish_fvg" and close[j] < fvg.zone_low:
                dist_pct = abs(current_price - fvg.zone_low) / current_price * 100
                out.append(Zone("ifvg_bearish", timeframe, fvg.zone_high, fvg.zone_low, n - 1 - j,
                                 round(dist_pct, 3), 0.45, {"origin": "bullish_fvg", "flipped_at": j}))
                break
            if fvg.zone_type == "bearish_fvg" and close[j] > fvg.zone_high:
                dist_pct = abs(current_price - fvg.zone_high) / current_price * 100
                out.append(Zone("ifvg_bullish", timeframe, fvg.zone_high, fvg.zone_low, n - 1 - j,
                                 round(dist_pct, 3), 0.45, {"origin": "bearish_fvg", "flipped_at": j}))
                break
    return out


# --------------------------------------------------------------------------- #
# Order Blocks (heuristic, documented above)
# --------------------------------------------------------------------------- #
def detect_order_blocks(df: pd.DataFrame, events, timeframe: str, lookback_events: int = 4) -> List[Zone]:
    """
    events: list of structure.StructureEvent (CHoCH/BOS) — for each BOS,
    walk back from the break candle to the last opposite-colored candle
    and use its body as the OB zone. This is a technical heuristic, not a
    claim of institutional origin (spec section 7 requirement).
    """
    n = len(df)
    open_, close_ = df["open"].values, df["close"].values
    current_price = float(close_[-1])
    zones: List[Zone] = []

    bos_events = [e for e in events if e.kind == "BOS"][-lookback_events:]
    for ev in bos_events:
        i = ev.break_index
        j = i
        if ev.direction == "bullish":
            while j >= 0 and close_[j] >= open_[j]:
                j -= 1
        else:
            while j >= 0 and close_[j] <= open_[j]:
                j -= 1
        if j < 0:
            continue
        body_low = min(open_[j], close_[j])
        body_high = max(open_[j], close_[j])
        dist_pct = min(abs(current_price - body_low), abs(current_price - body_high)) / current_price * 100
        kind = "bullish_ob" if ev.direction == "bullish" else "bearish_ob"
        # spec Part 13: every OB return must include an explicit `validity`
        # (not just distance/freshness) — invalidated if price has since
        # closed all the way back through the OB body opposite to its
        # direction (the common invalidation rule for this heuristic).
        if ev.direction == "bullish":
            invalidated = bool((close_[j + 1 :] < body_low).any()) if j + 1 < n else False
        else:
            invalidated = bool((close_[j + 1 :] > body_high).any()) if j + 1 < n else False
        zones.append(Zone(kind, timeframe, round(body_high, 8), round(body_low, 8),
                           freshness_index=n - 1 - j, distance_from_price_pct=round(dist_pct, 3),
                           confluence_score=0.5 if not invalidated else 0.1,
                           meta={"bos_index": i, "candle_index": j, "creation_index": j,
                                 "validity": "invalidated" if invalidated else "valid",
                                 "heuristic": "last opposite-colored candle before the BOS impulse — not a claim of institutional origin"}))
    return zones


# --------------------------------------------------------------------------- #
# Flip zones (support <-> resistance)
# --------------------------------------------------------------------------- #
def flip_zone_from_event(event, timeframe: str, current_price: float, band_pct: float = 0.1) -> Zone:
    band = event.level * band_pct / 100
    dist_pct = abs(current_price - event.level) / current_price * 100
    kind = "resistance_to_support" if event.direction == "bullish" else "support_to_resistance"
    return Zone(kind, timeframe, round(event.level + band, 8), round(event.level - band, 8),
                freshness_index=0, distance_from_price_pct=round(dist_pct, 3),
                confluence_score=0.45, meta={"source_event": event.kind, "break_index": event.break_index})


# --------------------------------------------------------------------------- #
# Premium / Discount
# --------------------------------------------------------------------------- #
def premium_discount(range_low: float, range_high: float, current_price: float) -> Dict[str, Any]:
    """
    FIXED (audit smoke-test finding): when price has moved well beyond the
    structural leg's range (a strong continuation past the original
    impulse), the raw pct_of_range can blow up to absurd values (e.g.
    4175%) that are technically correct as a ratio but useless to display.
    zone classification still uses the real, unclamped pct so "extreme
    premium/discount" is captured; pct_of_range is clamped to [0, 2] (0-200%)
    for display, with `beyond_range` flagging when price has moved outside
    the original leg entirely.
    """
    if range_high <= range_low:
        return {"zone": "unknown", "pct_of_range": 0.5, "beyond_range": False}
    pct = (current_price - range_low) / (range_high - range_low)
    if pct >= 0.6:
        zone = "premium"
    elif pct <= 0.4:
        zone = "discount"
    else:
        zone = "equilibrium"
    return {
        "zone": zone,
        "pct_of_range": round(max(0.0, min(pct, 1.0)), 3),
        "raw_pct": round(pct, 3),
        "beyond_range": bool(pct < 0 or pct > 1),
    }


# --------------------------------------------------------------------------- #
# Confluence scoring across all zones near price — DIRECTIONAL (spec Part 12)
# --------------------------------------------------------------------------- #
def location_confluence_score(zones: List[Zone], current_price: float, direction: str, proximity_pct: float = 1.5) -> Dict[str, Any]:
    """
    FIXED (audit, spec Part 12): this used to sum every zone's
    confluence_score regardless of polarity — a bearish OB sitting right at
    price would silently add to a bullish setup's confluence total. Now it
    only sums zones whose polarity supports `direction` ("BUY" -> bullish
    zones, "SELL" -> bearish zones). Opposing zones are still surfaced
    (as `opposing_score`) since a bearish zone near price for a long is a
    headwind — but they reduce confidence via the caller's own scoring,
    never get folded silently into "confluence".
    """
    wants_bull = direction == "BUY"
    supporting, opposing = 0.0, 0.0
    for z in zones:
        near = z.zone_low - current_price * proximity_pct / 100 <= current_price <= z.zone_high + current_price * proximity_pct / 100
        if not near:
            continue
        if (z.polarity == "bullish") == wants_bull and z.polarity != "neutral":
            supporting += z.confluence_score
        elif z.polarity != "neutral":
            opposing += z.confluence_score
    return {"supporting_score": round(min(supporting, 1.0), 2), "opposing_score": round(min(opposing, 1.0), 2)}
