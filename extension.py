"""
extension.py — Entry Extension / Chase Filter (Layer 5 of the entry
engine).

CORE PRINCIPLE this module exists to enforce: DIRECTION != ENTRY LOCATION.
A bullish BOS confirms direction. It does not, by itself, mean "buy now."
By the time a BOS is confirmed, price has already moved away from the
level that broke — sometimes by a normal, tradeable amount, sometimes by
an amount that makes the trade a chase with poor R:R and structural risk
sitting far below current price.

This module answers one question: given the BOS level, the current price,
recent volatility (ATR), and where price sits in the premium/discount
range, is NOW a reasonable place to enter, or has the move already run
too far?

Classification (spec-mandated three-tier label):
    EARLY        — price is still close to the break; a normal entry.
    EXTENDED     — price has moved meaningfully away; wait for a pullback,
                   but the setup is still alive.
    OVEREXTENDED — the move is mature (large % move + high volatility
                   expansion + deep premium/discount); do not chase at all,
                   only a genuine retest back near the break level would
                   revalidate this.

None of these three inputs alone is trusted (spec Part 9: "do not use one
arbitrary percentage only") — displacement_pct, atr_multiple, and
premium/discount position are each scored 0-1 and combined into a single
0-100 chase_score, which is what actually drives the EARLY/EXTENDED/
OVEREXTENDED label. Thresholds are configurable (settings-driven), not
hardcoded magic numbers buried in logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

ExtensionLabel = Literal["EARLY", "EXTENDED", "OVEREXTENDED"]


@dataclass
class ExtensionResult:
    label: ExtensionLabel
    chase_score: float                  # 0-100, higher = more extended/chased
    displacement_pct: float             # % price has moved from the break level
    atr_multiple: float                 # displacement expressed in ATRs
    premium_discount_component: float   # 0-1 contribution from where price sits in range
    preferred_entry_low: Optional[float]
    preferred_entry_high: Optional[float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def evaluate_extension(
    current_price: float,
    break_level: float,
    direction: str,
    atr: float,
    premium_discount_pct_of_range: float,
    extended_chase_score: float = 40.0,
    overextended_chase_score: float = 70.0,
    displacement_full_scale_pct: float = 8.0,
    atr_full_scale_multiple: float = 3.0,
) -> ExtensionResult:
    """
    Scores how far price has run since `break_level` (the CHoCH/BOS level)
    and returns a chase_score (0-100) plus an EARLY/EXTENDED/OVEREXTENDED
    label. Also computes a preferred pullback entry zone anchored around
    the break level, for use when the label is not EARLY.

    Three independent 0-1 components, averaged:
      1. displacement_pct / displacement_full_scale_pct (default: an 8%
         move since the break is "fully extended" on this axis alone)
      2. atr_multiple / atr_full_scale_multiple (default: a move of 3x
         ATR since the break is "fully extended" on this axis)
      3. premium/discount position: how deep into premium (for a long) or
         discount (for a short) current price sits — 0 at equilibrium,
         1 at the extreme edge of the range in the unfavorable direction.

    Each component is independently bounded to [0, 1] and none can alone
    force OVEREXTENDED — a large ATR move with price still near
    equilibrium, or a large % move with tiny ATR (i.e. a genuinely
    low-volatility structural break), both get a moderated score rather
    than an automatic reject. This directly implements the "do not use one
    arbitrary percentage only... create a configurable score" requirement.
    """
    if break_level <= 0 or current_price <= 0:
        raise ValueError("break_level and current_price must be positive")

    wants_bull = direction == "BUY"
    raw_displacement_pct = (current_price - break_level) / break_level * 100
    # Only displacement IN THE TRADE'S FAVOR counts as "chase risk" — if
    # price is behind the break level (hasn't followed through yet), that's
    # not extension, that's just... not there yet, handled elsewhere.
    displacement_pct = max(0.0, raw_displacement_pct if wants_bull else -raw_displacement_pct)

    atr_multiple = (displacement_pct / 100 * break_level) / atr if atr > 0 else 0.0

    disp_component = _clip01(displacement_pct / displacement_full_scale_pct)
    atr_component = _clip01(atr_multiple / atr_full_scale_multiple)

    # premium_discount_pct_of_range is 0 (bottom of range) .. 1 (top of
    # range), already clamped by zones.premium_discount(). A long chasing
    # INTO premium (close to 1) is exactly the "deep in premium" case the
    # spec calls out; a short chasing into discount (close to 0) mirrors it.
    pd_component = _clip01(premium_discount_pct_of_range if wants_bull else (1 - premium_discount_pct_of_range))

    chase_score = round((disp_component + atr_component + pd_component) / 3 * 100, 1)

    if chase_score < extended_chase_score:
        label: ExtensionLabel = "EARLY"
        reason = f"Price is still close to the break (chase score {chase_score}/100) — normal entry conditions."
    elif chase_score < overextended_chase_score:
        label = "EXTENDED"
        reason = (f"Price has moved meaningfully from the break (chase score {chase_score}/100, "
                  f"{displacement_pct:.2f}% / {atr_multiple:.2f}x ATR since break) — wait for a pullback.")
    else:
        label = "OVEREXTENDED"
        reason = (f"Move is mature (chase score {chase_score}/100, {displacement_pct:.2f}% / {atr_multiple:.2f}x ATR "
                  f"since break, deep in {'premium' if wants_bull else 'discount'}) — do not chase.")

    # Preferred pullback zone: anchored around the break level itself (the
    # classic "broken resistance becomes support" / "broken support becomes
    # resistance" zone), with a small ATR-based band so it's a zone, not a
    # single price. Only meaningful once price isn't already there (EARLY).
    if label == "EARLY":
        preferred_low = preferred_high = None
    else:
        band = max(atr * 0.5, break_level * 0.002)
        if wants_bull:
            preferred_low, preferred_high = round(break_level - band, 8), round(break_level + band, 8)
        else:
            preferred_low, preferred_high = round(break_level - band, 8), round(break_level + band, 8)

    return ExtensionResult(
        label=label, chase_score=chase_score, displacement_pct=round(displacement_pct, 3),
        atr_multiple=round(atr_multiple, 3), premium_discount_component=round(pd_component, 3),
        preferred_entry_low=preferred_low, preferred_entry_high=preferred_high, reason=reason,
    )
