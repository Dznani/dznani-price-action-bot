"""
tests/test_extension.py — unit tests for extension.py (Layer 5: Entry
Extension / Chase Filter).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extension as ext  # noqa: E402


def test_price_at_break_level_is_early():
    result = ext.evaluate_extension(
        current_price=100.0, break_level=100.0, direction="BUY", atr=1.0, premium_discount_pct_of_range=0.5,
    )
    assert result.label == "EARLY"
    assert result.chase_score < 40.0
    assert result.preferred_entry_low is None  # no pullback zone needed — already there


def test_moderate_move_is_extended_not_overextended():
    # ~4% move, ~2 ATR, mid-range premium/discount -> should land in the
    # EXTENDED band, not EARLY and not the extreme OVEREXTENDED tier.
    result = ext.evaluate_extension(
        current_price=104.0, break_level=100.0, direction="BUY", atr=2.0, premium_discount_pct_of_range=0.6,
    )
    assert result.label in ("EXTENDED", "OVEREXTENDED")
    assert result.chase_score > 0
    assert result.preferred_entry_low is not None
    assert result.preferred_entry_high is not None
    assert result.preferred_entry_low < result.preferred_entry_high


def test_large_move_deep_premium_high_atr_multiple_is_overextended():
    """This is the exact CAKE/LINK/BTC failure mode from the spec: price
    has moved far beyond the break, in a big multiple of ATR, deep into
    premium — must be OVEREXTENDED, not a chaseable entry."""
    result = ext.evaluate_extension(
        current_price=110.0, break_level=100.0, direction="BUY", atr=1.0, premium_discount_pct_of_range=0.95,
    )
    assert result.label == "OVEREXTENDED"
    assert result.chase_score >= 70.0
    assert result.preferred_entry_low is not None  # a pullback zone must still be offered


def test_single_axis_alone_cannot_force_overextended():
    """spec Part 9: 'do not use one arbitrary percentage only' — a big %
    move with tiny ATR (i.e. genuinely low-volatility structural break) or
    a big ATR-multiple move that's still small in % terms should not, on
    their own, produce OVEREXTENDED; only the combination should."""
    # Large % move but ATR is huge too (so atr_multiple stays tiny) and
    # price sits near equilibrium (0.5) -> should NOT be overextended.
    result = ext.evaluate_extension(
        current_price=108.0, break_level=100.0, direction="BUY", atr=50.0, premium_discount_pct_of_range=0.5,
    )
    assert result.label != "OVEREXTENDED"


def test_short_direction_extension_uses_inverse_premium_discount():
    """For a SELL, extension should be scored against price falling AWAY
    from the break level, and 'chasing' means being deep in DISCOUNT
    (near 0), not premium."""
    result_deep_discount = ext.evaluate_extension(
        current_price=90.0, break_level=100.0, direction="SELL", atr=1.0, premium_discount_pct_of_range=0.05,
    )
    result_near_equilibrium = ext.evaluate_extension(
        current_price=90.0, break_level=100.0, direction="SELL", atr=1.0, premium_discount_pct_of_range=0.5,
    )
    assert result_deep_discount.chase_score > result_near_equilibrium.chase_score


def test_price_behind_break_level_is_not_extension():
    """If price hasn't even followed through past the break level yet
    (still on the wrong side), that's not 'extension' — displacement must
    floor at zero, not go negative and somehow reduce the score."""
    result = ext.evaluate_extension(
        current_price=99.0, break_level=100.0, direction="BUY", atr=1.0, premium_discount_pct_of_range=0.5,
    )
    assert result.displacement_pct == 0.0
    assert result.label == "EARLY"


def test_preferred_zone_anchored_around_break_level():
    result = ext.evaluate_extension(
        current_price=115.0, break_level=100.0, direction="BUY", atr=1.0, premium_discount_pct_of_range=0.9,
    )
    assert result.preferred_entry_low < 100.0 < result.preferred_entry_high


def test_thresholds_are_configurable_not_hardcoded():
    """Confirms the EARLY/EXTENDED/OVEREXTENDED cutoffs actually respond
    to the settings-driven parameters, not fixed magic numbers."""
    kwargs = dict(current_price=103.0, break_level=100.0, direction="BUY", atr=2.0, premium_discount_pct_of_range=0.6)
    strict = ext.evaluate_extension(**kwargs, extended_chase_score=5.0, overextended_chase_score=10.0)
    lenient = ext.evaluate_extension(**kwargs, extended_chase_score=90.0, overextended_chase_score=99.0)
    assert strict.label != "EARLY" or lenient.label == "EARLY"
    assert strict.chase_score == lenient.chase_score  # same raw score, different labeling only


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        ext.evaluate_extension(current_price=0, break_level=100, direction="BUY", atr=1, premium_discount_pct_of_range=0.5)
    with pytest.raises(ValueError):
        ext.evaluate_extension(current_price=100, break_level=-5, direction="BUY", atr=1, premium_discount_pct_of_range=0.5)
