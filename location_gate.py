"""
location_gate.py — Location Quality Gate (Layer: entry confirmation).

THE RULE THIS MODULE ENCODES (explicit, not a blanket "never buy premium"):

    PREMIUM + NO RETEST + NO FRESH SUPPORT + HIGH OPPOSING LOCATION
        = WAIT (bad entry location)

    PREMIUM + NEW CONSOLIDATION + NEW HL + FLIP SUPPORT + VALID RETEST
        = still a valid new long, even though the broader range is
          technically premium

A blanket "premium = never buy" rule would be wrong and would turn this
into a rigid Fibonacci-only system — a bullish continuation can legitimately
occur in premium if a fresh local structure (a new HL, a flip zone, or a
held retest) has formed to defend it. What actually matters is whether
*something concrete* is defending the current price, not which half of the
broader swing range price happens to sit in.

This module is deliberately decoupled from structure.py/zones.py's object
types — it takes plain booleans/floats so it's directly unit-testable
without needing to construct full StructureState/Zone objects, matching
the pattern used by extension.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LocationGateResult:
    blocked: bool
    in_unfavorable_zone: bool
    fresh_support_present: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class LocationAssessment:
    """Card/backtest-ready location result, built on the existing hard gate."""
    location_grade: str
    location_score: float
    premium_discount: str
    pct_of_range: float
    nearest_valid_zone: Any
    zone_type: str
    distance_to_zone: float
    fresh_support: bool
    fresh_resistance: bool
    location_valid: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def assess_location(*, direction: str, premium_discount: Dict[str, Any], supporting_score: float,
                    opposing_score: float, has_fresh_hl_or_lh: bool, has_flip_zone_nearby: bool,
                    retest_held: bool, nearest_valid_zone: Any = None, zone_type: str = "",
                    distance_to_zone: float = 0.0, opposing_score_threshold: float = 0.3) -> LocationAssessment:
    """Return an explicit A-D location grade without making premium a ban."""
    gate = evaluate_location(direction, premium_discount.get("zone", "unknown"), supporting_score,
                             opposing_score, has_fresh_hl_or_lh, has_flip_zone_nearby,
                             retest_held, opposing_score_threshold)
    net = max(0.0, min(1.0, supporting_score - opposing_score + (0.25 if gate.fresh_support_present else 0.0)))
    score = round(net * 100, 1)
    if gate.blocked:
        grade = "D"
    elif score >= 70:
        grade = "A"
    elif score >= 40:
        grade = "B"
    elif score >= 15:
        grade = "C"
    else:
        grade = "D"
    # Discount/equilibrium can be usable with modest confluence; a premium
    # continuation must have fresh support and at least B-quality evidence.
    zone = premium_discount.get("zone", "unknown")
    unfavorable = gate.in_unfavorable_zone
    valid = not gate.blocked and grade in ("A", "B")
    if unfavorable and not gate.fresh_support_present:
        valid = False
    return LocationAssessment(grade, score, zone, float(premium_discount.get("pct_of_range", 0.5)),
                              nearest_valid_zone, zone_type, float(distance_to_zone),
                              gate.fresh_support_present if direction == "BUY" else False,
                              gate.fresh_support_present if direction == "SELL" else False,
                              valid, gate.reason)


def evaluate_location(
    direction: str,
    zone: str,
    supporting_score: float,
    opposing_score: float,
    has_fresh_hl_or_lh: bool,
    has_flip_zone_nearby: bool,
    retest_held: bool,
    opposing_score_threshold: float = 0.3,
) -> LocationGateResult:
    """
    direction: "BUY" or "SELL"
    zone: "premium" | "discount" | "equilibrium" (from zones.premium_discount)
    supporting_score / opposing_score: from zones.location_confluence_score
    has_fresh_hl_or_lh: a new higher-low (BUY) / lower-high (SELL) confirmed
        AFTER the break level formed — i.e. genuine new local structure,
        not just the pre-existing swing that originally supported the BOS
    has_flip_zone_nearby: a resistance->support (BUY) / support->resistance
        (SELL) flip zone sits at/near current price
    retest_held: the BOS level was retested and held (structure.RetestResult.held)

    "Fresh support" is deliberately an OR of three independent signals —
    any one of them (a new HL, a flip zone, or a held retest) is enough to
    say something concrete is defending price, which is what actually
    matters, not the coarse premium/discount label alone.
    """
    wants_bull = direction == "BUY"
    # "Unfavorable zone" = premium for a long (chasing into resistance
    # territory), discount for a short (chasing into support territory) —
    # this is a preference signal, never an automatic block by itself.
    in_unfavorable_zone = (zone == "premium" and wants_bull) or (zone == "discount" and not wants_bull)

    fresh_support_present = has_fresh_hl_or_lh or has_flip_zone_nearby or retest_held

    if not in_unfavorable_zone:
        return LocationGateResult(
            blocked=False, in_unfavorable_zone=False, fresh_support_present=fresh_support_present,
            reason=f"Location favorable ({zone}) — no location-based restriction.",
        )

    if fresh_support_present:
        # This is the explicit "still valid" case: premium, but a fresh
        # local structure is defending the entry. Do not block.
        supporting_reasons = []
        if has_fresh_hl_or_lh:
            supporting_reasons.append("new HL/LH")
        if has_flip_zone_nearby:
            supporting_reasons.append("flip zone")
        if retest_held:
            supporting_reasons.append("held retest")
        return LocationGateResult(
            blocked=False, in_unfavorable_zone=True, fresh_support_present=True,
            reason=f"In {zone} but defended by fresh local structure ({', '.join(supporting_reasons)}) — location accepted.",
        )

    # In the unfavorable zone, nothing concrete defending it. Only block if
    # the location score itself agrees (opposing >= supporting AND
    # meaningfully high) — a high-confluence unfavorable-zone setup with
    # weak opposing evidence still isn't automatically rejected; this
    # mirrors extension.py's "no single signal alone forces a reject".
    if opposing_score >= supporting_score and opposing_score > opposing_score_threshold:
        return LocationGateResult(
            blocked=True, in_unfavorable_zone=True, fresh_support_present=False,
            reason=(f"{zone.upper()} + no retest + no fresh support + opposing location score "
                    f"({opposing_score:.2f}) outweighs supporting ({supporting_score:.2f}) — bad entry location."),
        )

    return LocationGateResult(
        blocked=False, in_unfavorable_zone=True, fresh_support_present=False,
        reason=f"In {zone} with no fresh support, but opposing location evidence is weak — not blocked outright.",
    )
