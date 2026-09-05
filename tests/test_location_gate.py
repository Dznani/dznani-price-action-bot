"""
tests/test_location_gate.py — unit tests for location_gate.py.

These tests exist specifically to prevent the gate from ever regressing
into a blanket "never buy premium" rule, per the explicit design
requirement: premium alone is never sufficient to block; only
premium + no retest + no fresh support + high opposing evidence is.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import location_gate as lg  # noqa: E402


def test_premium_alone_does_not_block():
    """The core anti-regression test: premium with NO opposing evidence at
    all must never block, even with no fresh support — a rigid
    'never buy premium' rule is exactly what this must NOT become."""
    result = lg.evaluate_location(
        direction="BUY", zone="premium", supporting_score=0.5, opposing_score=0.0,
        has_fresh_hl_or_lh=False, has_flip_zone_nearby=False, retest_held=False,
    )
    assert result.blocked is False


def test_discount_never_blocks_a_long():
    result = lg.evaluate_location(
        direction="BUY", zone="discount", supporting_score=0.1, opposing_score=0.9,
        has_fresh_hl_or_lh=False, has_flip_zone_nearby=False, retest_held=False,
    )
    assert result.blocked is False
    assert result.in_unfavorable_zone is False


def test_premium_no_retest_no_support_high_opposing_blocks():
    """The exact bad case named in the spec: PREMIUM + NO RETEST + NO
    SUPPORT + HIGH OPPOSING LOCATION = WAIT."""
    result = lg.evaluate_location(
        direction="BUY", zone="premium", supporting_score=0.1, opposing_score=0.6,
        has_fresh_hl_or_lh=False, has_flip_zone_nearby=False, retest_held=False,
    )
    assert result.blocked is True
    assert "premium" in result.reason.lower() or "PREMIUM" in result.reason


def test_premium_with_fresh_hl_is_accepted():
    """The exact 'still valid' case named in the spec: PREMIUM + NEW HL =
    accepted, even with weak location scores otherwise."""
    result = lg.evaluate_location(
        direction="BUY", zone="premium", supporting_score=0.1, opposing_score=0.6,
        has_fresh_hl_or_lh=True, has_flip_zone_nearby=False, retest_held=False,
    )
    assert result.blocked is False
    assert result.fresh_support_present is True


def test_premium_with_flip_zone_is_accepted():
    result = lg.evaluate_location(
        direction="BUY", zone="premium", supporting_score=0.1, opposing_score=0.6,
        has_fresh_hl_or_lh=False, has_flip_zone_nearby=True, retest_held=False,
    )
    assert result.blocked is False


def test_premium_with_held_retest_is_accepted():
    result = lg.evaluate_location(
        direction="BUY", zone="premium", supporting_score=0.1, opposing_score=0.6,
        has_fresh_hl_or_lh=False, has_flip_zone_nearby=False, retest_held=True,
    )
    assert result.blocked is False


def test_premium_no_support_but_weak_opposing_not_blocked():
    """Even with no fresh support, weak opposing evidence alone shouldn't
    force a block — mirrors extension.py's no-single-signal-alone rule."""
    result = lg.evaluate_location(
        direction="BUY", zone="premium", supporting_score=0.5, opposing_score=0.15,
        has_fresh_hl_or_lh=False, has_flip_zone_nearby=False, retest_held=False,
    )
    assert result.blocked is False


def test_short_direction_mirrors_into_discount():
    """For a SELL, the unfavorable zone is DISCOUNT (chasing into support),
    not premium — confirms the mirror logic, not just a copy-paste bug."""
    blocked_case = lg.evaluate_location(
        direction="SELL", zone="discount", supporting_score=0.1, opposing_score=0.6,
        has_fresh_hl_or_lh=False, has_flip_zone_nearby=False, retest_held=False,
    )
    not_blocked_case = lg.evaluate_location(
        direction="SELL", zone="premium", supporting_score=0.1, opposing_score=0.6,
        has_fresh_hl_or_lh=False, has_flip_zone_nearby=False, retest_held=False,
    )
    assert blocked_case.blocked is True
    assert not_blocked_case.blocked is False  # premium is FAVORABLE for a short


def test_equilibrium_never_blocks():
    result = lg.evaluate_location(
        direction="BUY", zone="equilibrium", supporting_score=0.0, opposing_score=0.9,
        has_fresh_hl_or_lh=False, has_flip_zone_nearby=False, retest_held=False,
    )
    assert result.blocked is False


def test_any_single_fresh_support_signal_is_sufficient():
    """Confirms it's an OR, not an AND, across the three support signals —
    a fresh HL alone (without a flip zone or retest) must be enough."""
    for kwargs in [
        dict(has_fresh_hl_or_lh=True, has_flip_zone_nearby=False, retest_held=False),
        dict(has_fresh_hl_or_lh=False, has_flip_zone_nearby=True, retest_held=False),
        dict(has_fresh_hl_or_lh=False, has_flip_zone_nearby=False, retest_held=True),
    ]:
        result = lg.evaluate_location(
            direction="BUY", zone="premium", supporting_score=0.1, opposing_score=0.9, **kwargs,
        )
        assert result.blocked is False, f"expected not blocked with {kwargs}"
