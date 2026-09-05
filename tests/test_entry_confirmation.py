"""Tests for the explicit setup-vs-entry confirmation boundary."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import entry_confirmation as ec  # noqa: E402
import location_gate as lg  # noqa: E402


def test_touch_without_reaction_is_pending_not_confirmed():
    result = ec.evaluate_entry_confirmation(
        location_valid=True, returned_to_zone=True, zone_held=True,
        rejection_present=False, displacement_present=False,
        micro_structure_confirmed=False,
    )
    assert result.status == "PENDING"
    assert result.quality == "C"


def test_held_retest_rejection_and_micro_structure_confirm_entry():
    result = ec.evaluate_entry_confirmation(
        location_valid=True, returned_to_zone=True, zone_held=True,
        rejection_present=True, displacement_present=True,
        micro_structure_confirmed=True,
    )
    assert result.status == "CONFIRMED"
    assert result.quality == "A"


def test_failed_retest_invalidates_entry():
    result = ec.evaluate_entry_confirmation(
        location_valid=True, returned_to_zone=True, zone_held=False,
        rejection_present=False, displacement_present=False,
        micro_structure_confirmed=False, invalidated=True,
    )
    assert result.status == "FAILED"


def test_premium_continuation_requires_fresh_support_for_valid_location():
    bad = lg.assess_location(
        direction="BUY", premium_discount={"zone": "premium", "pct_of_range": 0.9},
        supporting_score=0.5, opposing_score=0.1, has_fresh_hl_or_lh=False,
        has_flip_zone_nearby=False, retest_held=False,
    )
    good = lg.assess_location(
        direction="BUY", premium_discount={"zone": "premium", "pct_of_range": 0.9},
        supporting_score=0.7, opposing_score=0.1, has_fresh_hl_or_lh=True,
        has_flip_zone_nearby=False, retest_held=False,
    )
    assert bad.location_valid is False
    assert good.location_valid is True
    assert good.location_grade in ("A", "B")
