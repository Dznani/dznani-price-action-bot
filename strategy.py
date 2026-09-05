"""
strategy.py — price-action decision engine for Dznani Signals Bot
(PROFESSIONAL PRICE ACTION EDITION).

Decision order (spec, non-negotiable):
    4H STRUCTURE -> 1H STRUCTURE -> LIQUIDITY -> SWEEP/BREAKOUT -> CHoCH
    -> LOCATION -> BOS -> RETEST -> R:R / ROOM -> ENTRY MODEL -> MANAGEMENT

Indicators (indicators.py) are secondary confirmation ONLY. They can boost
or reduce confidence on an already-valid price-action setup; they can never
create the primary direction and can never overrule the R:R hard filter.

evaluate_symbol() is the single entry point. It ALWAYS returns a dict
(never a bare None) once liquidity/length pass, because a NO TRADE / WATCHLIST
card is a first-class, required output (spec sections 19/21) — silence is
only correct when the liquidity filter or data-length guard rejects the
symbol outright (nothing to say yet).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

import indicators as ind
import extension as ext
import entry_confirmation as entry_confirm
import liquidity as liq
import location_gate as loc_gate
import risk
import rr_engine
import structure as struct_engine
import zones as zn

logger = logging.getLogger("dznani.strategy")

MIN_4H_CANDLES = 60
MIN_1H_CANDLES = 80

DEFAULT_WEIGHTS = {
    "htf_structure": 15,
    "structure_1h": 10,
    "liquidity": 10,
    "choch": 8,
    "location": 12,
    "bos": 15,
    "retest": 10,
    "rr": 10,
    "volume": 5,
    "momentum": 3,
    "indicator_confluence": 2,
}


def _grade(score_pct: float) -> str:
    if score_pct >= 85:
        return "A+"
    if score_pct >= 75:
        return "A"
    if score_pct >= 65:
        return "B"
    if score_pct >= 50:
        return "C"
    return "D"


def _score_components(direction, state_4h, state_1h, sweep, choch_event, bos_event, retest, location_score, rr_result, df_1h) -> Dict[str, float]:
    wants_bull = direction == "BUY"

    htf_ok = (state_4h.trend == "BULLISH") if wants_bull else (state_4h.trend == "BEARISH")
    htf_transition = state_4h.trend == "TRANSITION"
    htf_score = state_4h.confidence if htf_ok else (state_4h.confidence * 0.5 if htf_transition else 0.0)

    struct1h_ok = (state_1h.trend == "BULLISH") if wants_bull else (state_1h.trend == "BEARISH")
    struct1h_score = state_1h.confidence if struct1h_ok else 0.0

    liquidity_score = 0.0
    if sweep is not None:
        matches = (sweep.sweep_direction == "down" and wants_bull) or (sweep.sweep_direction == "up" and not wants_bull)
        if matches:
            liquidity_score = sweep.confidence

    choch_score = 0.0
    if choch_event is not None and (choch_event.direction == "bullish") == wants_bull:
        choch_score = 1.0

    bos_score = 0.0
    if bos_event is not None and (bos_event.direction == "bullish") == wants_bull:
        bos_score = 1.0
    elif choch_score:
        bos_score = 0.4  # CHoCH without BOS yet — partial credit

    retest_score = 0.0
    if retest is not None:
        retest_score = 1.0 if retest.held else (0.4 if retest.occurred else 0.0)

    rr_score = min(1.0, rr_result.rr / 2.0) if rr_result.rr > 0 else 0.0

    volume_ratio = _volume_ratio(df_1h)
    volume_score = min(1.0, max(0.0, (volume_ratio - 1.0) / 1.0))

    momentum_state = ind.calculate_momentum_state(df_1h, 10)
    momentum_score = 1.0 if (momentum_state == "Positive") == wants_bull and momentum_state is not None else 0.0

    indicator_score = _indicator_confluence_score(df_1h, direction)

    return {
        "htf_structure": htf_score, "structure_1h": struct1h_score, "liquidity": liquidity_score,
        "choch": choch_score, "location": location_score, "bos": bos_score, "retest": retest_score,
        "rr": rr_score, "volume": volume_score, "momentum": momentum_score, "indicator_confluence": indicator_score,
    }


def _volume_ratio(df: pd.DataFrame) -> float:
    avg_vol = df["volume"].rolling(20).mean().iloc[-1]
    last_vol = df["volume"].iloc[-1]
    if pd.isna(avg_vol) or avg_vol == 0:
        return 1.0
    return float(last_vol / avg_vol)


def _indicator_confluence_score(df: pd.DataFrame, direction: str) -> float:
    """Secondary-only: RSI divergence, MFI, ADX+EMA stack. Never the primary
    trigger — just how many of 3 secondary checks line up."""
    close = df["close"]
    rsi = ind.calculate_rsi(close, 14)
    mfi = ind.calculate_mfi(df, 14)
    divergence = ind.detect_divergence(df, rsi, lookback=6)
    ema_stack = ind.check_ema_stack(df)
    adx_value = float(ind.calculate_adx(df, 14).iloc[-1])

    hits = 0
    if direction == "BUY":
        if divergence == "Bullish":
            hits += 1
        if float(mfi.iloc[-1]) < 35:
            hits += 1
        if adx_value > 25 and ema_stack == "Bullish":
            hits += 1
    else:
        if divergence == "Bearish":
            hits += 1
        if float(mfi.iloc[-1]) > 65:
            hits += 1
        if adx_value > 25 and ema_stack == "Bearish":
            hits += 1
    return hits / 3


def _weighted_score(components: Dict[str, float], weights: Dict[str, float]) -> float:
    total_weight = sum(weights.values())
    total = sum(components[k] * weights.get(k, 0) for k in components)
    return round(total / total_weight * 100, 1) if total_weight else 0.0


def _select_entry_model(state_4h, state_1h, sweep, choch_event, bos_event, location_assessment, confirmation, direction: str) -> Optional[str]:
    """
    FIXED (audit, spec Part 19): tightened to match the spec's literal
    per-model conditions instead of a loose "whatever's available" ladder.
      Model A (aggressive, 30%): 4H context + 1H in TRANSITION (not just
        "any non-neutral trend") + an actual liquidity sweep event + CHoCH
        + STRONG location confluence (supporting_score >= 0.5).
      Model B (primary confirmation model): CHoCH + BOS + an A/B location
        AND explicit entry confirmation.  BOS plus a zone score is not an
        entry; this is the architectural distinction the old selection
        missed.
      Model C is deliberately not returned here: it is recorded separately
        as an experimental candidate until controlled testing approves it.
    """
    wants_bull = direction == "BUY"
    choch_ok = choch_event is not None and (choch_event.direction == "bullish") == wants_bull
    bos_ok = bos_event is not None and (bos_event.direction == "bullish") == wants_bull
    sweep_ok = sweep is not None and (
        (sweep.sweep_direction == "down" and wants_bull) or (sweep.sweep_direction == "up" and not wants_bull)
    )

    if choch_ok and bos_ok and location_assessment.location_valid and confirmation.status == "CONFIRMED":
        return "B"
    if choch_ok and state_1h.trend == "TRANSITION" and sweep_ok and location_assessment.location_grade == "A":
        return "A"
    return None


def evaluate_symbol(
    symbol: str,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    settings: Dict[str, Any],
    exchange_name: str = "Binance",
    quote_volume_24h: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    min_liquidity_usd = float(settings.get("min_liquidity_usd", 0) or 0)
    if quote_volume_24h is not None and min_liquidity_usd > 0 and quote_volume_24h < min_liquidity_usd:
        logger.debug("%s: skipped — below min_liquidity_usd", symbol)
        return None

    if len(df_4h) < MIN_4H_CANDLES or len(df_1h) < MIN_1H_CANDLES:
        logger.debug("%s: not enough candles (4H=%d, 1H=%d)", symbol, len(df_4h), len(df_1h))
        return None

    minimum_rr = float(settings.get("minimum_rr", 2.0))
    bos_disp_pct = float(settings.get("bos_min_displacement_pct", 0.15))
    require_min_displacement = bool(settings.get("require_min_displacement", False))
    min_swing_significance_pct = float(settings.get("min_swing_significance_pct", 0.3))
    equal_tolerance_pct = float(settings.get("liquidity_equal_tolerance_pct", 0.15))
    sweep_confirm_candles = int(settings.get("sweep_confirm_candles", 3))
    retest_tolerance_pct = float(settings.get("retest_tolerance_pct", 0.3))
    retest_min_excursion_pct = float(settings.get("retest_min_excursion_pct", 0.3))
    risk_per_trade_usd = float(settings.get("risk_per_trade_usd", 250.0))
    max_structural_risk_pct = float(settings.get("max_structural_risk_pct", 8.0))
    weights = {**DEFAULT_WEIGHTS, **(settings.get("indicator_weighting") or {})}

    current_price = float(df_1h["close"].iloc[-1])
    atr_1h = float(ind.calculate_atr(df_1h, 14).iloc[-1])

    state_4h = struct_engine.analyze_structure(df_4h, left=2, right=2)
    state_1h = struct_engine.analyze_structure_1h(
        df_1h, left=2, right=2, bos_min_displacement_pct=bos_disp_pct, require_min_displacement=require_min_displacement,
        min_swing_significance_pct=min_swing_significance_pct,
    )

    pools = liq.detect_liquidity_pools(df_1h, state_1h, equal_tolerance_pct=equal_tolerance_pct)
    sweep = liq.best_recent_sweep(df_1h, pools, sweep_confirm_candles=sweep_confirm_candles)

    choch_event = struct_engine.latest_event(state_1h, "CHoCH")
    bos_event = struct_engine.latest_event(state_1h, "BOS")

    # Conflicting HTF/LTF structure (spec Part 24): 4H flatly opposes what
    # 1H is doing (not just "not yet confirmed") -> explicit NO TRADE, not
    # a generic WATCHLIST card.
    conflicting_structure = (
        (state_4h.trend == "BEARISH" and bos_event is not None and bos_event.direction == "bullish")
        or (state_4h.trend == "BULLISH" and bos_event is not None and bos_event.direction == "bearish")
    )

    bullish_evidence = (state_4h.trend in ("BULLISH", "TRANSITION")) and (
        (choch_event and choch_event.direction == "bullish") or (bos_event and bos_event.direction == "bullish"))
    bearish_evidence = (state_4h.trend in ("BEARISH", "TRANSITION")) and (
        (choch_event and choch_event.direction == "bearish") or (bos_event and bos_event.direction == "bearish"))
    direction = "BUY" if bullish_evidence else ("SELL" if bearish_evidence else None)

    if conflicting_structure and direction is None:
        # 1H is clearly breaking one way while 4H context flatly opposes it —
        # surface this as an explicit rejection rather than silent WATCHLIST.
        bos_dir = "BUY" if bos_event.direction == "bullish" else "SELL"
        card = _developing_or_no_trade_card(symbol, exchange_name, current_price, state_4h, state_1h, sweep, [], {"zone": "unknown", "pct_of_range": 0.5}, df_1h, quote_volume_24h)
        card["signal_type"] = "NO TRADE"
        card["decision"] = "NO TRADE"
        card["reason"] = f"NO TRADE — conflicting HTF/LTF structure: 4H is {state_4h.trend} while 1H just gave a {bos_event.direction} BOS."
        return card

    # Relevant structural leg for Fibonacci (spec Part 14): use the
    # engine's own protected level (now correctly tied to the state
    # machine — see structure.py Part 4 fix) as one end of the leg, and the
    # most extreme swing in the move's direction as the other end. Falls
    # back to raw recent swings only when structure hasn't produced a
    # protected level yet.
    leg_direction = "bullish" if direction != "SELL" else "bearish"
    if leg_direction == "bullish":
        swing_low = state_1h.protected_low or (state_1h.last_HL.price if state_1h.last_HL else (state_1h.swing_lows[-1].price if state_1h.swing_lows else current_price * 0.97))
        swing_high = state_1h.last_HH.price if state_1h.last_HH else (state_1h.swing_highs[-1].price if state_1h.swing_highs else current_price * 1.03)
    else:
        swing_high = state_1h.protected_high or (state_1h.last_LH.price if state_1h.last_LH else (state_1h.swing_highs[-1].price if state_1h.swing_highs else current_price * 1.03))
        swing_low = state_1h.last_LL.price if state_1h.last_LL else (state_1h.swing_lows[-1].price if state_1h.swing_lows else current_price * 0.97)

    fib_zones = zn.fibonacci_zones(swing_low, swing_high, current_price, "1H", leg_direction)
    fvg_zones = zn.detect_fvgs(df_1h, "1H")
    ifvg_zones = zn.detect_ifvgs(df_1h, fvg_zones, "1H")
    ob_zones = zn.detect_order_blocks(df_1h, state_1h.events, "1H")
    flip_zones = [zn.flip_zone_from_event(e, "1H", current_price) for e in state_1h.events[-3:]]
    prem_disc = zn.premium_discount(swing_low, swing_high, current_price)
    all_zones = fib_zones + fvg_zones + ifvg_zones + ob_zones + flip_zones

    if direction is None:
        return _developing_or_no_trade_card(symbol, exchange_name, current_price, state_4h, state_1h, sweep, all_zones, prem_disc, df_1h, quote_volume_24h)

    location = zn.location_confluence_score(all_zones, current_price, direction)
    location_score = location["supporting_score"]

    retest = struct_engine.detect_retest(df_1h, bos_event, retest_tolerance_pct, min_excursion_pct=retest_min_excursion_pct) if bos_event else None
    invalid_retest = bool(retest and retest.occurred and not retest.held)

    # --- Layer 5: Entry Extension / Chase Filter ---
    # DIRECTION != ENTRY LOCATION (spec: a bullish BOS confirms direction,
    # never "buy now" by itself). Measures how far current_price has run
    # since the break level and returns EARLY/EXTENDED/OVEREXTENDED. See
    # extension.py for the full scoring rationale.
    break_event = bos_event or choch_event
    if break_event is not None:
        extension_result = ext.evaluate_extension(
            current_price=current_price, break_level=break_event.level, direction=direction, atr=atr_1h,
            premium_discount_pct_of_range=prem_disc.get("pct_of_range", 0.5),
            extended_chase_score=float(settings.get("extended_chase_score", 40.0)),
            overextended_chase_score=float(settings.get("overextended_chase_score", 70.0)),
            displacement_full_scale_pct=float(settings.get("displacement_full_scale_pct", 8.0)),
            atr_full_scale_multiple=float(settings.get("atr_full_scale_multiple", 3.0)),
        )
    else:
        extension_result = None

    if direction == "BUY":
        protected_level = state_1h.protected_low
        resistance_levels = [s.price for s in state_1h.swing_highs if s.price > current_price]
        resistance_levels += [s.price for s in state_4h.swing_highs if s.price > current_price]
        if state_4h.protected_high and state_4h.protected_high > current_price:
            resistance_levels.append(state_4h.protected_high)
        # spec Section 11: pull zone-based obstacles (order blocks, flip
        # zones) into the room calculation too — a bearish OB or a
        # support-flipped-to-resistance zone above price is just as real a
        # ceiling as a raw swing high, and the old version only looked at
        # swing points, which is why "Nearest resistance: —" could show up
        # even when a real structural level existed nearby.
        for z in ob_zones + flip_zones:
            if z.zone_type in ("bearish_ob", "support_to_resistance") and z.zone_low > current_price:
                resistance_levels.append(z.zone_low)
        support_levels: List[float] = []
    else:
        protected_level = state_1h.protected_high
        support_levels = [s.price for s in state_1h.swing_lows if s.price < current_price]
        support_levels += [s.price for s in state_4h.swing_lows if s.price < current_price]
        if state_4h.protected_low and state_4h.protected_low < current_price:
            support_levels.append(state_4h.protected_low)
        for z in ob_zones + flip_zones:
            if z.zone_type in ("bullish_ob", "resistance_to_support") and z.zone_high < current_price:
                support_levels.append(z.zone_high)
        resistance_levels = []

    # --- Location Quality Gate ---
    # The explicit rule (not a blanket "never buy premium"): premium + no
    # retest + no fresh support + high opposing location = bad entry, but
    # a genuinely fresh HL/flip-zone/held-retest still validates an entry
    # even within a broader premium range. See location_gate.py.
    if break_event is not None:
        if direction == "BUY":
            has_fresh_anchor = any(s.label == "HL" and s.index > break_event.break_index for s in state_1h.swing_lows)
            relevant_flip_type = "resistance_to_support"
        else:
            has_fresh_anchor = any(s.label == "LH" and s.index > break_event.break_index for s in state_1h.swing_highs)
            relevant_flip_type = "support_to_resistance"
        flip_proximity_pct = float(settings.get("flip_zone_proximity_pct", 1.5))
        has_flip_nearby = any(
            z.zone_type == relevant_flip_type
            and z.zone_low - current_price * flip_proximity_pct / 100 <= current_price <= z.zone_high + current_price * flip_proximity_pct / 100
            for z in flip_zones
        )
        location_gate_result = loc_gate.evaluate_location(
            direction=direction, zone=prem_disc.get("zone", "equilibrium"),
            supporting_score=location["supporting_score"], opposing_score=location["opposing_score"],
            has_fresh_hl_or_lh=has_fresh_anchor, has_flip_zone_nearby=has_flip_nearby, retest_held=bool(retest and retest.held),
            opposing_score_threshold=float(settings.get("location_opposing_threshold", 0.3)),
        )
    else:
        location_gate_result = None

    supporting_zones = [z for z in all_zones if (z.polarity == "bullish") == (direction == "BUY") and z.polarity != "neutral"]
    nearest_zone = min(supporting_zones, key=lambda z: z.distance_from_price_pct) if supporting_zones else None
    location_assessment = loc_gate.assess_location(
        direction=direction, premium_discount=prem_disc,
        supporting_score=location["supporting_score"], opposing_score=location["opposing_score"],
        has_fresh_hl_or_lh=has_fresh_anchor if break_event is not None else False,
        has_flip_zone_nearby=has_flip_nearby if break_event is not None else False,
        retest_held=bool(retest and retest.held),
        nearest_valid_zone=nearest_zone.to_dict() if nearest_zone else None,
        zone_type=nearest_zone.zone_type if nearest_zone else "",
        distance_to_zone=nearest_zone.distance_from_price_pct if nearest_zone else 0.0,
        opposing_score_threshold=float(settings.get("location_opposing_threshold", 0.3)),
    )
    entry_confirmation = entry_confirm.evaluate_entry_confirmation(
        location_valid=location_assessment.location_valid,
        returned_to_zone=bool(retest and retest.occurred),
        zone_held=bool(retest and retest.held) or (has_fresh_anchor and has_flip_nearby),
        rejection_present=bool(retest and retest.reaction_confirmed),
        displacement_present=bool(break_event and break_event.displacement_pct > 0),
        micro_structure_confirmed=bool(has_fresh_anchor),
        invalidated=invalid_retest,
        breakout_support_confirmed=bool(has_fresh_anchor and has_flip_nearby),
    )

    atr_buffer_mult = float(settings.get("sl_atr_buffer_mult", 0.5))
    provisional_sl = risk.calculate_structural_stop_loss(
        current_price, direction, protected_level, atr_1h, position_size_usd=1.0, atr_buffer_mult=atr_buffer_mult
    )

    # FIXED (production bug, VPS crash on INJ/USDT): a genuinely degenerate
    # structural stop — one that sits so close to entry that risk_pct
    # rounds to 0.000% — used to reach risk.calculate_position_plan(),
    # which correctly refuses to divide by a non-positive risk_pct and
    # raises. That crashed the entire batch backtest run instead of
    # treating this the way it should be treated: as an invalid setup.
    # A stop essentially AT entry provides no real protection and implies
    # an unbounded position size for a fixed dollar risk — this is a
    # NO TRADE case, not an exception.
    if provisional_sl.risk_pct <= 0:
        return {
            "symbol": symbol, "exchange": exchange_name, "quote_volume_24h": quote_volume_24h,
            "timeframe": "4H/1H", "current_price": current_price, "direction": direction,
            "signal_type": "NO TRADE", "decision": "NO TRADE",
            "reason": "NO TRADE — structural stop is essentially at entry price (risk_pct rounds to 0%) — "
                      "no real invalidation distance to size against.",
            "entry_model": None, "structure_4h": state_4h.to_dict(), "structure_1h": state_1h.to_dict(),
            "liquidity": {"pools": len(pools), "sweep": sweep.to_dict() if sweep else None},
            "choch": choch_event.__dict__ if choch_event else None, "bos": bos_event.__dict__ if bos_event else None,
            "retest": retest.__dict__ if retest else None,
            "location": {"supporting_score": location["supporting_score"], "opposing_score": location["opposing_score"],
                         "zones_near_price": location["supporting_score"], "premium_discount": prem_disc},
            "entry": current_price, "stop_loss": provisional_sl.final_sl, "targets": [],
            "setup_score": 0.0, "setup_grade": "D", "trade_quality_score": 0.0, "trade_quality_grade": "D",
            "indicators": _indicator_snapshot(df_1h, direction),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    rr_result = rr_engine.evaluate_rr(current_price, provisional_sl.final_sl, direction, resistance_levels, support_levels, minimum_rr=minimum_rr)

    components = _score_components(direction, state_4h, state_1h, sweep, choch_event, bos_event, retest, location_score, rr_result, df_1h)
    setup_score = _weighted_score(components, weights)
    setup_grade = _grade(setup_score)

    entry_model = _select_entry_model(state_4h, state_1h, sweep, choch_event, bos_event, location_assessment, entry_confirmation, direction)
    model_c_candidate = bool(
        choch_event is not None and bos_event is not None and bool(retest and retest.held)
        and entry_confirmation.status == "CONFIRMED" and location_assessment.location_valid
    )

    rr_component = min(1.0, rr_result.rr / minimum_rr) if rr_result.rr > 0 else 0.0
    trade_quality_score = round(min(100.0, rr_component * 65 + (setup_score / 100) * 35), 1)
    if not rr_result.passes_minimum:
        trade_quality_score = min(trade_quality_score, 45.0)  # R:R failure hard-caps trade quality regardless of setup grade
    if location["opposing_score"] > location["supporting_score"]:
        trade_quality_score = min(trade_quality_score, 55.0)  # opposing confluence outweighs supporting -> cap, don't ignore it
    trade_quality_grade = _grade(trade_quality_score)

    # --- Entry Quality Score (spec Section 14) ---
    # Deliberately SEPARATE from setup_score/trade_quality_score and built
    # from ONLY structure/location/retest/extension/room/risk — zero
    # indicator weight, so RSI/MFI/ADX/EMA/volume/momentum can never move
    # this number no matter how strongly they agree. This is the score
    # meant to answer "is NOW a good place to enter", as distinct from
    # "is the underlying setup good" (setup_score, which still exists).
    structure_component = (components["htf_structure"] + components["structure_1h"] + components["choch"] + components["bos"]) / 4
    location_component = max(0.0, location["supporting_score"] - location["opposing_score"])
    retest_component = components["retest"]
    extension_component = 1 - (extension_result.chase_score / 100) if extension_result else 0.5
    room_component = rr_component  # same room/R:R-derived signal used elsewhere, not recomputed differently
    risk_component = 1.0 if provisional_sl.risk_pct <= max_structural_risk_pct else 0.0
    entry_quality_weights = {"structure": 0.25, "location": 0.20, "retest": 0.15, "extension": 0.20, "room": 0.10, "risk": 0.10}
    entry_quality_score = round((
        structure_component * entry_quality_weights["structure"]
        + location_component * entry_quality_weights["location"]
        + retest_component * entry_quality_weights["retest"]
        + extension_component * entry_quality_weights["extension"]
        + room_component * entry_quality_weights["room"]
        + risk_component * entry_quality_weights["risk"]
    ) * 100, 1)
    entry_quality_grade = _grade(entry_quality_score)

    capital = float(settings.get("capital", risk.DEFAULT_CAPITAL))
    position_plan = risk.calculate_position_plan(risk_per_trade_usd, provisional_sl.risk_pct, available_capital_usd=capital)

    # FIXED (backtest diagnostic finding, review #3): a structural stop so
    # tight that hitting the dollar-risk target would require a position
    # bigger than the account's actual capital is not a real, executable
    # trade — it's the sizing formula being asked an impossible question.
    # A real 180-day backtest run produced an $8.3M implied position size
    # against a $25k account on exactly this pattern. Reject it outright
    # rather than silently truncating the size and reporting an R:R that
    # no longer applies at the truncated size.
    if position_plan.exceeds_available_capital:
        return {
            "symbol": symbol, "exchange": exchange_name, "quote_volume_24h": quote_volume_24h,
            "timeframe": "4H/1H", "current_price": current_price, "direction": direction,
            "signal_type": "NO TRADE", "decision": "NO TRADE",
            "reason": f"NO TRADE — structural risk ({provisional_sl.risk_pct}%) is so tight that hitting the "
                      f"${risk_per_trade_usd:,.0f} dollar-risk target would require a "
                      f"${risk_per_trade_usd / (provisional_sl.risk_pct / 100):,.0f} position, far beyond the "
                      f"${capital:,.0f} account — not an executable trade.",
            "entry_model": None, "structure_4h": state_4h.to_dict(), "structure_1h": state_1h.to_dict(),
            "liquidity": {"pools": len(pools), "sweep": sweep.to_dict() if sweep else None},
            "choch": choch_event.__dict__ if choch_event else None, "bos": bos_event.__dict__ if bos_event else None,
            "retest": retest.__dict__ if retest else None,
            "location": {"supporting_score": location["supporting_score"], "opposing_score": location["opposing_score"],
                         "zones_near_price": location["supporting_score"], "premium_discount": prem_disc},
            "entry": current_price, "stop_loss": provisional_sl.final_sl, "targets": [],
            "setup_score": setup_score, "setup_grade": setup_grade, "trade_quality_score": 0.0, "trade_quality_grade": "D",
            "indicators": _indicator_snapshot(df_1h, direction),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    structural_sl = risk.calculate_structural_stop_loss(
        current_price, direction, protected_level, atr_1h, position_size_usd=position_plan.aggressive_usd, atr_buffer_mult=atr_buffer_mult
    )

    scale_out = risk.build_scale_out_plan(current_price, "BUY")
    obstacle_levels = resistance_levels if direction == "BUY" else support_levels
    tp_conflict = risk.check_targets_against_structure(scale_out, "BUY", obstacle_levels)

    # R:R to each fixed target and to the nearest structural obstacle (spec
    # Part 17/20) — kept separate from the account-risk-% position sizing,
    # never mixed together.
    risk_distance = rr_result.risk_distance
    rr_to_targets = {
        lvl["label"]: round(abs(lvl["price"] - current_price) / risk_distance, 3) if risk_distance > 0 else 0.0
        for lvl in [{"label": l.label, "price": l.price} for l in scale_out]
    }
    rr_to_targets["RESISTANCE" if direction == "BUY" else "SUPPORT"] = rr_result.rr

    excessive_structural_risk = structural_sl.risk_pct > max_structural_risk_pct

    if direction == "SELL":
        signal_type = "DIP-BUY PLAN" if entry_model else "BEARISH CONTEXT"
        decision = "WAIT"
        reason = "Bearish structure — spot-only bot does not open shorts. Levels below are for dip-buy planning only."
    elif excessive_structural_risk:
        signal_type, decision = "NO TRADE", "NO TRADE"
        reason = (f"NO TRADE — structural risk too large: structural invalidation implies "
                  f"{structural_sl.risk_pct}% risk, above the {max_structural_risk_pct}% ceiling. "
                  f"Per spec, the stop is not moved closer to force a smaller size — wait for a tighter structural setup.")
    elif invalid_retest:
        signal_type, decision = "INVALIDATED", "NO TRADE"
        reason = "INVALIDATED — price returned to the BOS level but failed to hold (reclaimed beyond tolerance)."
    elif extension_result is not None and extension_result.label == "OVEREXTENDED":
        # spec Section 2/8/9/23: DIRECTION != ENTRY LOCATION. A confirmed
        # CHoCH/BOS with strong indicators is NOT a chase license — this
        # gate fires regardless of how good setup_score/R:R look, exactly
        # matching the spec's "BOS + volume 3x + ADX high... still NO CHASE"
        # example. Checked BEFORE the R:R gate so the person sees the real
        # reason (chase), not a misleadingly generic "R:R too low" — even
        # though a chased entry's R:R is very often ALSO poor as a
        # consequence of the same extension.
        signal_type, decision = "NO CHASE", "WAIT"
        reason = f"NO CHASE — {extension_result.reason}"
    elif extension_result is not None and extension_result.label == "EXTENDED":
        signal_type, decision = "WAIT FOR RETEST", "WAIT"
        reason = f"WAIT FOR RETEST — {extension_result.reason}"
    elif location_gate_result is not None and location_gate_result.blocked:
        # spec: PREMIUM + NO RETEST + NO SUPPORT + HIGH OPPOSING LOCATION
        # = WAIT — checked distinctly from extension so the person sees
        # the real reason (bad location, not distance/ATR).
        signal_type, decision = "WAIT FOR RETEST", "WAIT"
        reason = f"WAIT FOR RETEST — {location_gate_result.reason}"
    elif entry_confirmation.status == "FAILED":
        signal_type, decision = "INVALIDATED", "NO TRADE"
        reason = f"INVALIDATED — {entry_confirmation.reason}"
    elif not location_assessment.location_valid:
        signal_type, decision = "WATCH", "WAIT"
        reason = f"WATCH — location grade {location_assessment.location_grade}: {location_assessment.reason}"
    elif entry_model is None:
        signal_type = "WATCHLIST / DEVELOPING SETUP"
        decision = "WAIT"
        if choch_event and not bos_event:
            reason = "CHoCH present but no BOS yet — conservative models (B/C) need BOS confirmation first."
        elif choch_event and bos_event and entry_confirmation.status == "PENDING":
            reason = f"WAIT FOR ENTRY CONFIRMATION — {entry_confirmation.reason}"
            signal_type = "WAIT FOR ENTRY CONFIRMATION"
        else:
            reason = "Bullish bias forming but CHoCH/BOS confirmation not yet in place."
    elif not rr_result.passes_minimum:
        signal_type, decision = "NO TRADE", "NO TRADE"
        near_obstacle = rr_result.available_room_pct < risk_distance / current_price * 100 * 1.5 if current_price else False
        obstacle_note = "major resistance too close" if (direction == "BUY" and near_obstacle) else "major support too close" if near_obstacle else "insufficient room to the nearest obstacle"
        reason = (f"NO TRADE — insufficient R:R ({obstacle_note}): setup quality {setup_grade} but R:R {rr_result.rr} "
                  f"is below minimum {minimum_rr} — nearest {'resistance' if direction == 'BUY' else 'support'} only "
                  f"{rr_result.available_room_pct}% away. Trade quality {trade_quality_grade}.")
    else:
        decision = "VALID"
        reason = f"{entry_model}-model long: setup {setup_grade}, trade quality {trade_quality_grade}, R:R {rr_result.rr}."
        # spec Section 22 vocabulary: Model A (30% initial, least
        # confirmation) is an EARLY LONG; Model B/C (BOS+location, or
        # BOS+retest+reaction) are both CONFIRMATION LONG — the spec's
        # two-tier A/B naming (Section 11) collapses the old three-model
        # (A/B/C) display into two display tiers, while entry_model still
        # tracks the underlying A/B/C mechanics for position sizing.
        signal_type = "EARLY LONG" if entry_model == "A" else "CONFIRMATION LONG"

    return {
        "symbol": symbol, "exchange": exchange_name, "quote_volume_24h": quote_volume_24h,
        "timeframe": "4H/1H", "current_price": current_price, "direction": direction,
        "signal_type": signal_type, "decision": decision, "reason": reason, "entry_model": entry_model,
        "structure_4h": state_4h.to_dict(), "structure_1h": state_1h.to_dict(),
        "liquidity": {"pools": len(pools), "sweep": sweep.to_dict() if sweep else None},
        "choch": choch_event.__dict__ if choch_event else None,
        "bos": bos_event.__dict__ if bos_event else None,
        "retest": retest.__dict__ if retest else None,
        "location": {
            "supporting_score": location["supporting_score"], "opposing_score": location["opposing_score"],
            "zones_near_price": location["supporting_score"], "premium_discount": prem_disc,
            "assessment": location_assessment.to_dict(),
            "fib": [z.to_dict() for z in fib_zones], "order_blocks": [z.to_dict() for z in ob_zones[-3:]],
            "fvg": [z.to_dict() for z in fvg_zones[-3:]], "ifvg": [z.to_dict() for z in ifvg_zones[-3:]],
            "flip_zones": [z.to_dict() for z in flip_zones],
        },
        "entry": current_price, "stop_loss": structural_sl.final_sl,
        "structural_invalidation": structural_sl.structural_invalidation, "stop_loss_pct": structural_sl.risk_pct,
        "excessive_structural_risk": excessive_structural_risk, "max_structural_risk_pct": max_structural_risk_pct,
        "extension": extension_result.to_dict() if extension_result else None,
        "location_gate": location_gate_result.to_dict() if location_gate_result else None,
        "entry_confirmation": entry_confirmation.to_dict(), "model_c_candidate": model_c_candidate,
        "position_plan": position_plan.to_dict(),
        "nearest_resistance": rr_result.nearest_obstacle if direction == "BUY" else None,
        "nearest_support": rr_result.nearest_obstacle if direction == "SELL" else None,
        "available_room_pct": rr_result.available_room_pct, "rr": rr_result.to_dict(),
        "rr_to_targets": rr_to_targets,
        "targets": [{"label": lvl.label, "price": lvl.price, "pct": lvl.pct, "sell_fraction": lvl.sell_fraction} for lvl in scale_out],
        "targets_structure_conflict": tp_conflict,
        "setup_score": setup_score, "setup_grade": setup_grade,
        "trade_quality_score": trade_quality_score, "trade_quality_grade": trade_quality_grade,
        "entry_quality_score": entry_quality_score, "entry_quality_grade": entry_quality_grade,
        "score_components": {k: round(v, 2) for k, v in components.items()},
        "indicators": _indicator_snapshot(df_1h, direction),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _developing_or_no_trade_card(symbol, exchange_name, current_price, state_4h, state_1h, sweep, zones_list, prem_disc, df_1h, quote_volume_24h) -> Dict[str, Any]:
    return {
        "symbol": symbol, "exchange": exchange_name, "quote_volume_24h": quote_volume_24h,
        "timeframe": "4H/1H", "current_price": current_price, "direction": None,
        "signal_type": "WATCHLIST / DEVELOPING SETUP" if state_4h.trend != "NEUTRAL" else "NO TRADE",
        "decision": "WAIT" if state_4h.trend != "NEUTRAL" else "NO TRADE",
        "reason": "No CHoCH or BOS aligned with 4H context yet — nothing actionable.",
        "entry_model": None, "structure_4h": state_4h.to_dict(), "structure_1h": state_1h.to_dict(),
        "liquidity": {"sweep": sweep.to_dict() if sweep else None},
        "choch": None, "bos": None, "retest": None,
        "location": {"premium_discount": prem_disc, "supporting_score": 0.0, "opposing_score": 0.0, "zones_near_price": 0.0},
        "entry": None, "stop_loss": None, "targets": [],
        "setup_score": 0.0, "setup_grade": "D", "trade_quality_score": 0.0, "trade_quality_grade": "D",
        "indicators": _indicator_snapshot(df_1h, "BUY"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _indicator_snapshot(df: pd.DataFrame, direction: str) -> Dict[str, Any]:
    close = df["close"]
    rsi = ind.calculate_rsi(close, 14)
    mfi = ind.calculate_mfi(df, 14)
    adx = ind.calculate_adx(df, 14)
    divergence = ind.detect_divergence(df, rsi, lookback=6)
    pattern = ind.detect_price_action(df)
    momentum_state = ind.calculate_momentum_state(df, 10)
    ema_stack = ind.check_ema_stack(df)
    squeeze_on_series, momentum_osc_series = ind.calculate_sqz_mom(df)
    return {
        "rsi": round(float(rsi.iloc[-1]), 1), "rsi_divergence": divergence, "price_action_pattern": pattern,
        "volume_spike_ratio": round(_volume_ratio(df), 2), "mfi": round(float(mfi.iloc[-1]), 1),
        "momentum_state": momentum_state, "adx": round(float(adx.iloc[-1]), 1), "ema_stack": ema_stack,
        "squeeze_on": bool(squeeze_on_series.iloc[-1]), "squeeze_momentum": round(float(momentum_osc_series.iloc[-1]), 2),
    }


def should_send_signal(symbol: str, last_signal_time, duplicate_hours: int = 6) -> bool:
    if last_signal_time is None:
        return True
    now = datetime.now(timezone.utc)
    elapsed_hours = (now - last_signal_time).total_seconds() / 3600
    return elapsed_hours > duplicate_hours
