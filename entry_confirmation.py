"""Entry-confirmation gate: setup confirmation is not entry confirmation.

This module is deliberately data-shape agnostic so it can be unit-tested and
reused by live scanning and walk-forward backtesting.  It does not decide
direction, location, risk, or R:R; it only answers whether price has actually
validated an otherwise-valid entry area.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal


ConfirmationStatus = Literal["CONFIRMED", "PENDING", "FAILED"]


@dataclass
class EntryConfirmationResult:
    status: ConfirmationStatus
    quality: str
    returned_to_zone: bool
    zone_held: bool
    rejection_present: bool
    displacement_present: bool
    micro_structure_confirmed: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def evaluate_entry_confirmation(
    *,
    location_valid: bool,
    returned_to_zone: bool,
    zone_held: bool,
    rejection_present: bool,
    displacement_present: bool,
    micro_structure_confirmed: bool,
    invalidated: bool = False,
    breakout_support_confirmed: bool = False,
) -> EntryConfirmationResult:
    """Evaluate a retest or fresh breakout-support entry.

    A touch alone is never confirmation.  A standard retest needs a return,
    hold, rejection/displacement, and micro confirmation.  A freshly formed
    flip-support + higher-low may confirm a continuation without revisiting the
    original BOS level, but it still needs both support and micro structure.
    """
    if invalidated:
        return EntryConfirmationResult("FAILED", "D", returned_to_zone, False,
                                       rejection_present, displacement_present,
                                       micro_structure_confirmed,
                                       "Entry area failed: price broke the retest/support structure.")
    if not location_valid:
        return EntryConfirmationResult("PENDING", "D", returned_to_zone, zone_held,
                                       rejection_present, displacement_present,
                                       micro_structure_confirmed,
                                       "No valid entry location yet.")

    retest_confirmed = (returned_to_zone and zone_held and rejection_present
                        and displacement_present and micro_structure_confirmed)
    breakout_confirmed = (breakout_support_confirmed and zone_held
                          and micro_structure_confirmed and displacement_present)
    if retest_confirmed:
        return EntryConfirmationResult("CONFIRMED", "A", True, True, True, True, True,
                                       "Retest returned to the zone, held, rejected, displaced, and confirmed micro structure.")
    if breakout_confirmed:
        return EntryConfirmationResult("CONFIRMED", "B", returned_to_zone, True,
                                       rejection_present, True, True,
                                       "Fresh breakout support held with displacement and micro structure confirmation.")
    if returned_to_zone and not zone_held:
        return EntryConfirmationResult("FAILED", "D", True, False, rejection_present,
                                       displacement_present, micro_structure_confirmed,
                                       "Price touched the entry area but did not hold it.")
    return EntryConfirmationResult("PENDING", "C" if returned_to_zone else "D",
                                   returned_to_zone, zone_held, rejection_present,
                                   displacement_present, micro_structure_confirmed,
                                   "Waiting for a held zone, rejection/displacement, and micro structure confirmation.")
