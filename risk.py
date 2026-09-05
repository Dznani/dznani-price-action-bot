"""
risk.py — position sizing, structural stop loss, and scale-out logic
for Dznani Signals Bot (Price Action Edition).

PRIMARY STOP LOSS IS STRUCTURAL (spec section 12): for a long, below the
relevant protected HL / invalidation zone; ATR is only a volatility BUFFER
added beyond that structural point, never the primary reason for placement.
The legacy ATR-only calculate_stop_loss() is kept for backward
compatibility (e.g. quick /scan replies with no structure context) but
strategy.py's live/backtest path uses calculate_structural_stop_loss().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Constants (overridable via config/settings where noted)
# --------------------------------------------------------------------------- #
SL_ATR_MULTIPLIER = 1.5
SL_MIN_PCT = 0.015
SL_MAX_PCT = 0.06

TP1_PCT = 0.05
TP2_PCT = 0.10
TP3_PCT = 0.15
TP1_SELL_FRACTION = 0.40
TP2_SELL_FRACTION = 0.40
TP3_SELL_FRACTION = 0.20

MAX_HOLDING_HOURS = 336  # 14 days

POSITION_SIZE_BY_STRENGTH = {
    5: 15000.0,
    4: 7500.0,
    3: 2500.0,
}

DEFAULT_CAPITAL = 25000.0
DEFAULT_DAILY_LOSS_LIMIT_PCT = 0.05  # 5% of capital


# --------------------------------------------------------------------------- #
# Stop Loss
# --------------------------------------------------------------------------- #
def calculate_stop_loss(entry: float, atr: float, direction: str, atr_multiplier: float = SL_ATR_MULTIPLIER) -> float:
    """
    ATR-based stop loss: Entry ± (ATR * multiplier), clamped to
    [SL_MIN_PCT, SL_MAX_PCT] of entry price.
    """
    if entry <= 0:
        raise ValueError("entry must be positive")

    raw_distance = atr * atr_multiplier
    raw_pct = raw_distance / entry
    clamped_pct = min(max(raw_pct, SL_MIN_PCT), SL_MAX_PCT)
    distance = entry * clamped_pct

    if direction == "BUY":
        return round(entry - distance, 8)
    elif direction == "SELL":
        return round(entry + distance, 8)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def stop_loss_pct(entry: float, stop_loss: float) -> float:
    return abs(entry - stop_loss) / entry


# --------------------------------------------------------------------------- #
# Structural stop loss (spec section 12) — PRIMARY method for the new engine
# --------------------------------------------------------------------------- #
@dataclass
class StructuralSL:
    structural_invalidation: float
    volatility_buffer: float
    final_sl: float
    risk_pct: float
    dollar_risk: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def calculate_structural_stop_loss(
    entry: float,
    direction: str,
    protected_level: Optional[float],
    atr: float,
    position_size_usd: float,
    atr_buffer_mult: float = 0.5,
    fallback_atr_mult: float = SL_ATR_MULTIPLIER,
) -> StructuralSL:
    """
    Long: SL = protected HL minus a small ATR buffer (so a normal wick back
    to the level doesn't stop the trade out one tick early).
    Bearish/dip-buy context: SL = protected LH plus a small ATR buffer.
    If no protected level is known yet (e.g. brand-new structure), falls
    back to the plain ATR-multiple method so the bot never returns a signal
    with no stop.

    IMPORTANT (fixed 2024 audit, spec Part 15): this function reports the
    REAL structural risk_pct, unclamped. It used to silently clamp risk_pct
    into [SL_MIN_PCT, SL_MAX_PCT] and reprice the SL to fit — which quietly
    moved the stop away from the actual structural invalidation point
    whenever that point implied "too much" or "too little" risk. That is
    exactly the anti-pattern the spec calls out: "Do NOT move a structural
    stop closer merely because a global maximum-risk percentage was
    exceeded... Instead: NO TRADE." Enforcing a max-acceptable-risk ceiling
    is now strategy.py's job (it rejects the trade with a specific reason
    if risk_pct is too large) — this function's only job is to report the
    truth about where structure says the stop belongs.
    """
    if entry <= 0:
        raise ValueError("entry must be positive")

    buffer = atr * atr_buffer_mult

    if protected_level is None:
        sl = calculate_stop_loss(entry, atr, direction, fallback_atr_mult)
        invalidation = sl
    elif direction == "BUY":
        invalidation = protected_level
        sl = round(protected_level - buffer, 8)
        if sl >= entry:  # protected level is above/at entry — structure invalid for a long here
            sl = calculate_stop_loss(entry, atr, direction, fallback_atr_mult)
            invalidation = sl
    elif direction == "SELL":
        invalidation = protected_level
        sl = round(protected_level + buffer, 8)
        if sl <= entry:
            sl = calculate_stop_loss(entry, atr, direction, fallback_atr_mult)
            invalidation = sl
    else:
        raise ValueError(f"Unknown direction: {direction}")

    risk_pct = stop_loss_pct(entry, sl)  # real, unclamped structural risk
    dollar_risk = round(position_size_usd * risk_pct, 2)

    return StructuralSL(
        structural_invalidation=round(invalidation, 8) if invalidation is not None else sl,
        volatility_buffer=round(buffer, 8),
        final_sl=sl,
        risk_pct=round(risk_pct * 100, 3),
        dollar_risk=dollar_risk,
    )


# --------------------------------------------------------------------------- #
# 30/70 position model (spec section 11)
# --------------------------------------------------------------------------- #
AGGRESSIVE_ENTRY_PCT = 0.30
CONFIRMATION_ADD_PCT = 0.70


@dataclass
class PositionPlan:
    max_position_usd: float
    aggressive_usd: float
    confirmation_usd: float
    risk_per_trade_usd: float
    structural_risk_pct: float
    exceeds_available_capital: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def calculate_position_plan(
    risk_per_trade_usd: float,
    structural_risk_pct: float,
    aggressive_pct: float = AGGRESSIVE_ENTRY_PCT,
    confirmation_pct: float = CONFIRMATION_ADD_PCT,
    available_capital_usd: Optional[float] = None,
) -> PositionPlan:
    """
    Position sizing is derived from dollar risk / structural SL distance —
    NOT a fixed strength-based dollar amount. Example from spec: risk $250,
    structural risk 4% -> max position $6,250 -> 30% ($1,875) initial,
    70% ($4,375) reserved for confirmation.

    CRITICAL (spec 11/30): the 70% may only ever be added when the
    predefined confirmation conditions (CHoCH+BOS, or BOS+retest, per the
    chosen entry model) are met AND the resulting position still satisfies
    the R:R/risk constraints — never because the first 30% is losing, and
    never to average down. Enforcing "when" is strategy.py's job; this
    function only computes the sizes.

    FIXED (backtest diagnostic finding, review #3): dollar risk / risk_pct
    has no ceiling on its own — for a very tight structural stop (small
    but nonzero risk_pct, e.g. 0.01%), this formula happily returns a
    position size in the millions of dollars ($8.3M was observed in a real
    180-day backtest run) to keep dollar risk at exactly $250. That's not
    a sizing decision, it's the formula being asked a question with no
    sane answer — no spot account this bot is designed for can actually
    hold an $8.3M position. `available_capital_usd`, when provided, caps
    `max_position_usd` at that ceiling and sets `exceeds_available_capital`
    so the caller (strategy.py) can reject the trade outright rather than
    silently truncating and pretending the resulting R:R still applies at
    a smaller, arbitrary size.
    """
    if structural_risk_pct <= 0:
        raise ValueError("structural_risk_pct must be positive")

    max_position_usd = risk_per_trade_usd / (structural_risk_pct / 100)
    exceeds_capital = False
    if available_capital_usd is not None and max_position_usd > available_capital_usd:
        exceeds_capital = True
        max_position_usd = available_capital_usd

    return PositionPlan(
        max_position_usd=round(max_position_usd, 2),
        aggressive_usd=round(max_position_usd * aggressive_pct, 2),
        confirmation_usd=round(max_position_usd * confirmation_pct, 2),
        risk_per_trade_usd=risk_per_trade_usd,
        structural_risk_pct=structural_risk_pct,
        exceeds_available_capital=exceeds_capital,
    )


# --------------------------------------------------------------------------- #
# Take Profit / Scale-out plan
# --------------------------------------------------------------------------- #
@dataclass
class ScaleOutLevel:
    label: str
    pct: float
    price: float
    sell_fraction: float
    sl_after_hit: float  # where the stop loss should move to once this TP fills


def build_scale_out_plan(entry: float, direction: str) -> List[ScaleOutLevel]:
    """
    Returns the TP1/TP2/TP3 plan for a trade:
      TP1 (+5%)  -> sell 40%, move SL to breakeven (entry)
      TP2 (+10%) -> sell 40%, move SL to TP1
      TP3 (+15%) -> sell 20%, close
    Works symmetrically for SELL (dip-buy / short-style) trades.
    """
    sign = 1 if direction == "BUY" else -1

    tp1_price = entry * (1 + sign * TP1_PCT)
    tp2_price = entry * (1 + sign * TP2_PCT)
    tp3_price = entry * (1 + sign * TP3_PCT)

    return [
        ScaleOutLevel("TP1", TP1_PCT, round(tp1_price, 8), TP1_SELL_FRACTION, sl_after_hit=round(entry, 8)),
        ScaleOutLevel("TP2", TP2_PCT, round(tp2_price, 8), TP2_SELL_FRACTION, sl_after_hit=round(tp1_price, 8)),
        ScaleOutLevel("TP3", TP3_PCT, round(tp3_price, 8), TP3_SELL_FRACTION, sl_after_hit=round(tp2_price, 8)),
    ]


# --------------------------------------------------------------------------- #
# Position sizing
# --------------------------------------------------------------------------- #
def position_size_for_strength(strength: int) -> float:
    """5/5 -> $15,000 | 4/5 -> $7,500 | 3/5 -> $2,500 | else -> 0 (skip)."""
    return POSITION_SIZE_BY_STRENGTH.get(strength, 0.0)


# --------------------------------------------------------------------------- #
# Daily loss limit
# --------------------------------------------------------------------------- #
def daily_loss_limit_breached(
    daily_pnl: float,
    capital: float = DEFAULT_CAPITAL,
    limit_pct: float = DEFAULT_DAILY_LOSS_LIMIT_PCT,
) -> bool:
    """True if today's realized loss has hit/exceeded the daily loss limit."""
    limit_amount = capital * limit_pct
    return daily_pnl <= -limit_amount


def daily_loss_buffer_remaining(
    daily_pnl: float,
    capital: float = DEFAULT_CAPITAL,
    limit_pct: float = DEFAULT_DAILY_LOSS_LIMIT_PCT,
) -> float:
    """Dollar amount of further loss allowed before the daily limit is hit."""
    limit_amount = capital * limit_pct
    remaining = limit_amount + daily_pnl  # daily_pnl is negative when losing
    return max(remaining, 0.0)


# --------------------------------------------------------------------------- #
# Max holding period
# --------------------------------------------------------------------------- #
def is_past_max_holding(open_date_iso: str, max_hours: int = MAX_HOLDING_HOURS) -> bool:
    """True if a trade opened at `open_date_iso` (ISO 8601) has exceeded the max holding period."""
    opened = datetime.fromisoformat(open_date_iso)
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - opened) >= timedelta(hours=max_hours)


# --------------------------------------------------------------------------- #
# Structure-aware TP checking (spec section 17) and stop management (18)
# --------------------------------------------------------------------------- #
def check_targets_against_structure(
    targets: List[ScaleOutLevel],
    direction: str,
    obstacle_levels: List[float],
) -> Dict[str, Any]:
    """
    Flags any fixed TP that sits beyond the nearest opposing structural
    level — the bot must not blindly assume a fixed-percent target will be
    reached if structure stands in the way before it.
    """
    conflicts = {}
    for lvl in targets:
        if direction == "BUY":
            blockers = [o for o in obstacle_levels if lvl.price >= o >= min(t.price for t in targets) * 0 and o < lvl.price]
            blockers = [o for o in obstacle_levels if o < lvl.price]
        else:
            blockers = [o for o in obstacle_levels if o > lvl.price]
        if blockers:
            nearest = max(blockers) if direction == "BUY" else min(blockers)
            conflicts[lvl.label] = {"target_price": lvl.price, "conflicting_level": nearest}
    return {"has_conflict": bool(conflicts), "conflicts": conflicts}


def next_stop_after_tp(
    tp_label: str,
    entry: float,
    tp1_price: float,
    protected_structure_level: Optional[float],
    direction: str,
    current_sl: float,
) -> float:
    """
    FIXED (audit, spec Part 21): after TP1, the stop is NO LONGER moved to
    breakeven automatically. The spec explicitly reversed that default:
    "Do NOT automatically move to breakeven. Check whether a new protected
    HL has formed. If yes: trail below structure. If no: keep the existing
    valid stop." So this now requires `current_sl` (the stop that was
    actually in place going into this TP) and returns it unchanged when no
    fresh protected structure exists — it never fabricates a breakeven
    level out of thin air. After TP2, the runner trails behind the most
    protective of {TP1 level, fresh protected structure, current stop} and
    never loosens back toward entry.
    """
    if tp_label == "TP1":
        if protected_structure_level is not None:
            if direction == "BUY" and protected_structure_level > current_sl:
                return protected_structure_level
            if direction == "SELL" and protected_structure_level < current_sl:
                return protected_structure_level
        return current_sl  # no fresh protected structure -> keep the existing valid stop, do NOT force breakeven
    if tp_label == "TP2":
        candidates = [p for p in (tp1_price, protected_structure_level, current_sl) if p is not None]
        return max(candidates) if direction == "BUY" else min(candidates)
    raise ValueError(f"Unknown label: {tp_label}")


# --------------------------------------------------------------------------- #
# P&L helper
# --------------------------------------------------------------------------- #
def calculate_pnl(entry_price: float, exit_price: float, size_usd: float, direction: str = "BUY") -> float:
    """Simple spot P&L in USD given entry/exit price and position size in USD."""
    if entry_price <= 0:
        return 0.0
    qty = size_usd / entry_price
    if direction == "BUY":
        return round((exit_price - entry_price) * qty, 2)
    else:  # SELL / short-style bookkeeping, if ever used
        return round((entry_price - exit_price) * qty, 2)
