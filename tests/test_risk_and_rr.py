"""
tests/test_risk_and_rr.py

Covers spec test cases #15 (structural SL), #16 (R:R rejection),
#17 (30% aggressive entry), #18 (70% confirmation addition),
#19 (no averaging down), #20 (TP1/TP2/TP3 fractions).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import risk  # noqa: E402
import rr_engine  # noqa: E402


def test_structural_stop_loss_uses_protected_level_not_just_atr():
    entry = 100.0
    protected_low = 96.0  # a real HL well within the sane 1.5-6% band
    result = risk.calculate_structural_stop_loss(entry, "BUY", protected_low, atr=1.0, position_size_usd=1000.0)
    assert result.final_sl < entry
    # SL should sit close to the structural level, not an arbitrary ATR multiple far from it.
    assert abs(result.final_sl - protected_low) < 2.5
    assert result.dollar_risk > 0


def test_rr_engine_rejects_setup_with_insufficient_room():
    # Entry 100, SL 96 (4% risk), resistance only 1.8% away -> R:R < 1, hard reject.
    result = rr_engine.evaluate_rr(
        entry=100.0, stop_loss=96.0, direction="BUY",
        resistance_levels=[101.8], minimum_rr=2.0,
    )
    assert result.rr < 1.0
    assert result.passes_minimum is False


def test_rr_engine_accepts_setup_with_ample_room():
    result = rr_engine.evaluate_rr(
        entry=100.0, stop_loss=96.0, direction="BUY",
        resistance_levels=[112.0], minimum_rr=2.0,
    )
    assert result.rr >= 2.0
    assert result.passes_minimum is True


def test_position_plan_30_70_split():
    plan = risk.calculate_position_plan(risk_per_trade_usd=250.0, structural_risk_pct=4.0)
    assert round(plan.max_position_usd, 2) == 6250.0
    assert round(plan.aggressive_usd, 2) == 1875.0
    assert round(plan.confirmation_usd, 2) == 4375.0
    assert round(plan.aggressive_usd + plan.confirmation_usd, 2) == round(plan.max_position_usd, 2)


def test_no_averaging_down_stop_never_loosens_toward_losing_price():
    # Simulates the rule at the risk.py level: next_stop_after_tp only ever
    # moves the stop toward/through breakeven or a fresh protected level in
    # the trade's favor — it has no path that moves the stop further away
    # from entry (which is what "adding because it's losing" would require).
    #
    # FIXED (audit, spec Part 21): TP1 with NO fresh protected structure
    # must now keep the EXISTING stop unchanged — NOT force a move to
    # breakeven. That's an explicit reversal from the previous default.
    entry = 100.0
    original_sl = 96.0
    tp1 = 105.0
    sl_after_tp1_no_structure = risk.next_stop_after_tp("TP1", entry, tp1, protected_structure_level=None, direction="BUY", current_sl=original_sl)
    assert sl_after_tp1_no_structure == original_sl  # kept unchanged, NOT moved to breakeven

    sl_after_tp1_with_structure = risk.next_stop_after_tp("TP1", entry, tp1, protected_structure_level=102.0, direction="BUY", current_sl=original_sl)
    assert sl_after_tp1_with_structure == 102.0  # trails to the fresh protected structure since it's tighter than the old stop

    # A fresh protected level that is WORSE (looser) than the current stop must not loosen it.
    sl_after_tp1_worse_structure = risk.next_stop_after_tp("TP1", entry, tp1, protected_structure_level=94.0, direction="BUY", current_sl=original_sl)
    assert sl_after_tp1_worse_structure == original_sl


def test_scale_out_fractions_sum_to_one():
    scale_out = risk.build_scale_out_plan(100.0, "BUY")
    total_fraction = sum(lvl.sell_fraction for lvl in scale_out)
    assert abs(total_fraction - 1.0) < 1e-9
    labels = [lvl.label for lvl in scale_out]
    assert labels == ["TP1", "TP2", "TP3"]
    assert scale_out[0].price < scale_out[1].price < scale_out[2].price
