"""
backtest.py — Stateful Walk-Forward Backtester for Dznani Price Action Bot.

Runs the SAME core strategy logic as the live scanner with the PERSISTENT WATCH
ENGINE to track setups across future candles as price returns to preferred zones.

Key Features:
  - Real Binance historical data (1H + 4H) via CCXT
  - Walk-forward simulation with zero lookahead (closed 4H bars sliced by 1H close time)
  - Stateful Persistent Watch Manager (Model B pullback / zone confirmation)
  - Model A early aggressive entries (30% initial, 70% confirmation add)
  - Structural Stop Loss with configurable ATR buffer
  - Structure-aware Take Profit management (TP1 +5%, TP2 +10%, TP3 +15%)
  - Full Rejection Funnel reporting
  - Watch & Trade statistics with MFE/MAE analysis
  - In-Sample vs Out-of-Sample validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

import risk
import strategy
from database import DEFAULT_SETTINGS, Database
from exchange import BinanceExchange
from utils import setup_logging
from watch_engine import WatchManager, WatchSetup

logger = logging.getLogger("dznani.backtest")

WARMUP_1H = strategy.MIN_1H_CANDLES + 10
MAX_ADD_WAIT_CANDLES = 24  # how long Model A waits for BOS confirmation before 70% add expires


@dataclass
class SimulatedTrade:
    symbol: str
    entry_model: str
    setup_score: float
    setup_grade: str
    trade_quality_grade: str
    rr: float
    market_regime_4h: str
    entry_time: str
    entry_price: float
    exit_time: Optional[str]
    exit_reason: str
    return_pct: float
    position_size: float
    pnl_usd: float
    hit_tp1: bool
    hit_tp2: bool
    hit_tp3: bool
    added_70pct: bool
    is_out_of_sample: bool
    structure_1h_trend: str = ""
    stop_loss: float = 0.0
    structural_invalidation: float = 0.0
    stop_loss_pct: float = 0.0
    choch_present: bool = False
    choch_direction: str = ""
    bos_present: bool = False
    bos_direction: str = ""
    retest_occurred: bool = False
    retest_held: bool = False
    sweep_present: bool = False
    sweep_direction: str = ""
    premium_discount_zone: str = ""
    nearest_resistance: Optional[float] = None
    available_room_pct: float = 0.0
    location_supporting_score: float = 0.0
    location_opposing_score: float = 0.0
    entry_signal_type: str = ""
    chase_score: float = 0.0
    extension_label: str = ""
    displacement_from_bos_pct: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    holding_hours: float = 0.0
    r_multiple: float = 0.0
    diagnostic_flag: str = ""


def _slice_4h(df_4h: pd.DataFrame, now_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Excludes 4H candles that have not fully closed by now_ts (the close time
    of the most recent closed 1H candle).
    """
    closed_mask = df_4h["timestamp"] + pd.Timedelta(hours=4) <= now_ts
    return df_4h[closed_mask]


def simulate_trade(
    df_1h: pd.DataFrame,
    entry_idx: int,
    signal: Dict[str, Any],
    settings: Dict[str, Any],
    max_holding_hours: int = risk.MAX_HOLDING_HOURS,
) -> Dict[str, Any]:
    """
    Simulates forward from entry_idx applying:
      - Structural SL with structure-aware trailing after TP1/TP2
      - Scale-out targets (TP1 40% at +5%, TP2 40% at +10%, TP3 20% at +15%)
      - Model A 70% confirmation add on fresh BOS within MAX_ADD_WAIT_CANDLES
      - Same-candle conservative rule: low <= SL is checked before high >= TP.
    """
    entry = float(signal["entry"])
    entry_time = df_1h["timestamp"].iloc[entry_idx]
    pos_plan = signal.get("position_plan") or {}
    entry_model = signal.get("entry_model", "B")

    current_sl = float(signal["stop_loss"])
    initial_sl = current_sl
    targets = {t["label"]: t for t in signal.get("targets", [])}
    if not targets:
        # Fallback scale-out if targets list is empty
        scale_out = risk.build_scale_out_plan(entry, "BUY")
        targets = {t.label: {"label": t.label, "price": t.price, "sell_fraction": t.sell_fraction} for t in scale_out}

    tp_order = ["TP1", "TP2", "TP3"]
    hit = {"TP1": False, "TP2": False, "TP3": False}

    full_size = float(pos_plan.get("max_position_usd", 2500.0))
    active_size = float(pos_plan.get("aggressive_usd", full_size * 0.3)) if entry_model == "A" else full_size
    added_70pct = entry_model != "A"

    remaining_fraction = 1.0
    weighted_return_pct = 0.0
    exit_reason = "data_end"
    exit_time = df_1h["timestamp"].iloc[-1]
    exit_idx = len(df_1h) - 1
    protected_level = signal.get("structural_invalidation")
    max_favorable_price = entry
    max_adverse_price = entry

    n = len(df_1h)
    for j in range(entry_idx + 1, n):
        row = df_1h.iloc[j]
        high_val = float(row["high"])
        low_val = float(row["low"])
        close_val = float(row["close"])
        max_favorable_price = max(max_favorable_price, high_val)
        max_adverse_price = min(max_adverse_price, low_val)
        elapsed_hours = (row["timestamp"] - entry_time).total_seconds() / 3600

        # Model A: 70% confirmation add upon fresh BOS
        if entry_model == "A" and not added_70pct and (j - entry_idx) <= MAX_ADD_WAIT_CANDLES:
            window = df_1h.iloc[: j + 1]
            try:
                mini_state = strategy.struct_engine.analyze_structure_1h(
                    window, bos_min_displacement_pct=settings.get("bos_min_displacement_pct", 0.15)
                )
                latest_bos = strategy.struct_engine.latest_event(mini_state, "BOS")
                if latest_bos and latest_bos.direction == "bullish" and latest_bos.break_index == j:
                    active_size = full_size
                    added_70pct = True
            except Exception:
                pass

        # Max holding period exit
        if elapsed_hours > max_holding_hours:
            exit_price = close_val
            weighted_return_pct += remaining_fraction * (exit_price - entry) / entry * 100
            exit_reason, exit_time, exit_idx, remaining_fraction = "max_holding", row["timestamp"], j, 0.0
            break

        # Stop-loss check (conservative same-candle priority)
        if low_val <= current_sl:
            weighted_return_pct += remaining_fraction * (current_sl - entry) / entry * 100
            if current_sl == signal["stop_loss"]:
                exit_reason = "structural_stop_loss"
            elif current_sl == entry:
                exit_reason = "breakeven_stop"
            else:
                exit_reason = "trailing_structure_stop"
            exit_time, exit_idx, remaining_fraction = row["timestamp"], j, 0.0
            break

        # Take profit check
        for label in tp_order:
            if hit[label] or label not in targets:
                continue
            tp_price = targets[label]["price"]
            if high_val < tp_price:
                continue
            frac = targets[label]["sell_fraction"]
            weighted_return_pct += frac * (tp_price - entry) / entry * 100
            remaining_fraction -= frac
            hit[label] = True
            if label == "TP1":
                current_sl = risk.next_stop_after_tp("TP1", entry, targets["TP1"]["price"], protected_level, "BUY", current_sl)
            elif label == "TP2":
                current_sl = risk.next_stop_after_tp("TP2", entry, targets["TP1"]["price"], protected_level, "BUY", current_sl)
            elif label == "TP3":
                exit_reason, exit_time, exit_idx = "tp3_full", row["timestamp"], j

        if hit.get("TP3"):
            remaining_fraction = 0.0
            break

    if remaining_fraction > 0 and exit_reason == "data_end":
        exit_price = float(df_1h["close"].iloc[-1])
        weighted_return_pct += remaining_fraction * (exit_price - entry) / entry * 100

    pnl_usd = round(weighted_return_pct / 100 * active_size, 2)
    mfe_pct = round((max_favorable_price - entry) / entry * 100, 3)
    mae_pct = round((max_adverse_price - entry) / entry * 100, 3)
    holding_hours = round((exit_time - entry_time).total_seconds() / 3600, 1)

    # Calculate R-multiple achieved
    sl_distance = abs(entry - initial_sl)
    r_multiple = round(((entry * (weighted_return_pct / 100)) / sl_distance), 2) if sl_distance > 0 else 0.0

    # Diagnostic flag
    tp1_price = targets.get("TP1", {}).get("price", entry * 1.05)
    future_high = float(df_1h["high"].iloc[exit_idx + 1 :].max()) if exit_idx + 1 < n else entry
    if exit_reason == "structural_stop_loss" and future_high >= tp1_price:
        diagnostic_flag = "POSSIBLE STOP ISSUE"
    elif mfe_pct < 0.25 and mae_pct > 0.25:
        diagnostic_flag = "ENTRY FAILURE"
    elif mfe_pct >= 5.0 and weighted_return_pct <= 0:
        diagnostic_flag = "EXIT MANAGEMENT ISSUE"
    else:
        diagnostic_flag = ""

    return {
        "return_pct": round(weighted_return_pct, 3),
        "exit_reason": exit_reason,
        "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time),
        "exit_idx": exit_idx,
        "pnl_usd": pnl_usd,
        "position_size": active_size,
        "added_70pct": added_70pct,
        "hit_tp1": hit["TP1"],
        "hit_tp2": hit["TP2"],
        "hit_tp3": hit["TP3"],
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "holding_hours": holding_hours,
        "r_multiple": r_multiple,
        "diagnostic_flag": diagnostic_flag,
    }


def run_backtest_symbol(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    symbol: str,
    settings: Dict[str, Any],
    oos_cutoff_ts=None,
    warmup: int = WARMUP_1H,
) -> Dict[str, Any]:
    """
    Stateful walk-forward backtest over 1H historical candles.
    Uses WatchManager to track pending setups across candles.
    """
    trades: List[SimulatedTrade] = []
    watch_mgr = WatchManager(default_expiry_candles=int(settings.get("max_watch_candles", 36)))

    funnel = {
        "candles_evaluated": 0,
        "4h_bullish_or_transition": 0,
        "choch_events": 0,
        "bos_events": 0,
        "location_valid_a_or_b": 0,
        "extension_early": 0,
        "extension_extended": 0,
        "extension_overextended": 0,
        "immediate_valid_entries": 0,
        "watches_created": 0,
        "watches_zone_touched": 0,
        "watches_reaction_confirmed": 0,
        "watches_passed_risk": 0,
        "watches_passed_rr": 0,
        "trades_executed": 0,
    }

    rejection_reasons: Dict[str, int] = {
        "conflicting_htf_structure": 0,
        "no_choch_or_bos": 0,
        "poor_location": 0,
        "overextended_no_chase": 0,
        "extended_wait_retest": 0,
        "watch_expired_before_touch": 0,
        "watch_invalidated_by_stop_breach": 0,
        "watch_failed_reaction": 0,
        "watch_rejected_excessive_risk": 0,
        "watch_rejected_insufficient_rr": 0,
        "immediate_rejected_rr": 0,
        "immediate_rejected_risk": 0,
    }

    if len(df_1h) < warmup + 2 or len(df_4h) < 30:
        logger.warning("%s: not enough candles (1H=%d, 4H=%d) — skipping", symbol, len(df_1h), len(df_4h))
        return {
            "trades": trades,
            "funnel": funnel,
            "rejections": rejection_reasons,
            "watch_stats": watch_mgr.get_statistics(),
        }

    duplicate_hours = int(settings.get("duplicate_signal_hours", 6))
    last_signal_time = None

    i = warmup
    while i < len(df_1h) - 1:
        window_1h = df_1h.iloc[: i + 1]
        now_ts = window_1h["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        window_4h = _slice_4h(df_4h, now_ts)
        if len(window_4h) < 30:
            i += 1
            continue

        funnel["candles_evaluated"] += 1

        # ------------------------------------------------------------------ #
        # Step 1: Update persistent watches against current candle
        # ------------------------------------------------------------------ #
        try:
            confirmed_watch_signals = watch_mgr.evaluate_active_watches(window_1h, window_4h, settings)
        except Exception as e:
            logger.error("%s: evaluate_active_watches failed at candle %s: %s", symbol, window_1h["timestamp"].iloc[-1], e)
            confirmed_watch_signals = []

        for signal in confirmed_watch_signals:
            funnel["watches_passed_rr"] += 1
            funnel["trades_executed"] += 1
            candle_time = window_1h["timestamp"].iloc[-1]
            last_signal_time = candle_time.to_pydatetime() if hasattr(candle_time, "to_pydatetime") else candle_time

            result = simulate_trade(df_1h, i, signal, settings)
            trades.append(_create_simulated_trade(symbol, signal, result, window_1h, oos_cutoff_ts, now_ts))
            i = max(result["exit_idx"], i + 1)
            break  # Move to next candle after trade exit

        # ------------------------------------------------------------------ #
        # Step 2: Evaluate current candle for new structural setups
        # ------------------------------------------------------------------ #
        try:
            signal = strategy.evaluate_symbol(symbol, window_4h, window_1h, settings)
        except Exception as e:
            logger.error("%s: evaluate_symbol failed at candle %s: %s", symbol, window_1h["timestamp"].iloc[-1], e)
            i += 1
            continue

        if signal:
            struct_4h = signal.get("structure_4h") or {}
            if struct_4h.get("trend") in ("BULLISH", "TRANSITION"):
                funnel["4h_bullish_or_transition"] += 1

            if signal.get("choch"):
                funnel["choch_events"] += 1
            if signal.get("bos"):
                funnel["bos_events"] += 1

            loc_assessment = (signal.get("location") or {}).get("assessment") or {}
            if loc_assessment.get("location_valid"):
                funnel["location_valid_a_or_b"] += 1

            ext_info = signal.get("extension")
            if ext_info:
                label = ext_info.get("label")
                if label == "EARLY":
                    funnel["extension_early"] += 1
                elif label == "EXTENDED":
                    funnel["extension_extended"] += 1
                    rejection_reasons["extended_wait_retest"] += 1
                elif label == "OVEREXTENDED":
                    funnel["extension_overextended"] += 1
                    rejection_reasons["overextended_no_chase"] += 1

            decision = signal.get("decision")
            signal_type = signal.get("signal_type", "")

            # If setup is WAIT (e.g. EXTENDED, OVEREXTENDED, WAIT FOR RETEST, WATCH), register persistent watch!
            if decision == "WAIT" and signal.get("direction") == "BUY":
                created_watch = watch_mgr.create_watch_from_signal(
                    signal=signal,
                    candle_idx=i,
                    timestamp=window_1h["timestamp"].iloc[-1],
                    expiry_candles=int(settings.get("max_watch_candles", 36)),
                )
                if created_watch:
                    funnel["watches_created"] += 1

            elif decision == "VALID" and signal.get("direction") == "BUY":
                funnel["immediate_valid_entries"] += 1
                candle_time = window_1h["timestamp"].iloc[-1]
                if last_signal_time is not None and not strategy.should_send_signal(symbol, last_signal_time, duplicate_hours):
                    i += 1
                    continue
                last_signal_time = candle_time.to_pydatetime() if hasattr(candle_time, "to_pydatetime") else candle_time

                funnel["trades_executed"] += 1
                result = simulate_trade(df_1h, i, signal, settings)
                trades.append(_create_simulated_trade(symbol, signal, result, window_1h, oos_cutoff_ts, now_ts))
                i = max(result["exit_idx"], i + 1)
                continue

            elif decision == "NO TRADE":
                reason = signal.get("reason", "")
                if "insufficient R:R" in reason:
                    rejection_reasons["immediate_rejected_rr"] += 1
                elif "structural risk" in reason:
                    rejection_reasons["immediate_rejected_risk"] += 1
                elif "conflicting HTF/LTF" in reason:
                    rejection_reasons["conflicting_htf_structure"] += 1
                elif "No CHoCH or BOS" in reason:
                    rejection_reasons["no_choch_or_bos"] += 1

        i += 1

    watch_stats = watch_mgr.get_statistics()
    funnel["watches_zone_touched"] = watch_stats["zone_touched_count"]
    funnel["watches_reaction_confirmed"] = watch_stats["zone_touched_count"] - watch_stats["expired_count"]
    funnel["watches_passed_risk"] = watch_stats["zone_touched_count"] - watch_stats["rejected_at_entry_count"]

    rejection_reasons["watch_expired_before_touch"] = watch_stats["expired_count"]
    rejection_reasons["watch_invalidated_by_stop_breach"] = watch_stats["invalidated_count"]
    rejection_reasons["watch_rejected_insufficient_rr"] = watch_stats["rejected_at_entry_count"]

    return {
        "trades": trades,
        "funnel": funnel,
        "rejections": rejection_reasons,
        "watch_stats": watch_stats,
    }


def _create_simulated_trade(
    symbol: str,
    signal: Dict[str, Any],
    result: Dict[str, Any],
    window_1h: pd.DataFrame,
    oos_cutoff_ts,
    now_ts: pd.Timestamp,
) -> SimulatedTrade:
    loc = signal.get("location") or {}
    return SimulatedTrade(
        symbol=symbol,
        entry_model=signal.get("entry_model", "B"),
        setup_score=float(signal.get("setup_score", 0.0)),
        setup_grade=signal.get("setup_grade", "B"),
        trade_quality_grade=signal.get("trade_quality_grade", "A"),
        rr=float((signal.get("rr") or {}).get("rr", 0.0)),
        market_regime_4h=(signal.get("structure_4h") or {}).get("trend", "BULLISH"),
        entry_time=window_1h["timestamp"].iloc[-1].isoformat(),
        entry_price=float(signal.get("entry", 0.0)),
        exit_time=result["exit_time"],
        exit_reason=result["exit_reason"],
        return_pct=result["return_pct"],
        position_size=result["position_size"],
        pnl_usd=result["pnl_usd"],
        hit_tp1=result["hit_tp1"],
        hit_tp2=result["hit_tp2"],
        hit_tp3=result["hit_tp3"],
        added_70pct=result["added_70pct"],
        is_out_of_sample=bool(oos_cutoff_ts is not None and now_ts >= oos_cutoff_ts),
        structure_1h_trend=(signal.get("structure_1h") or {}).get("trend", ""),
        stop_loss=float(signal.get("stop_loss", 0.0)),
        structural_invalidation=float(signal.get("structural_invalidation", 0.0)),
        stop_loss_pct=float(signal.get("stop_loss_pct", 0.0)),
        choch_present=signal.get("choch") is not None,
        choch_direction=(signal.get("choch") or {}).get("direction", ""),
        bos_present=signal.get("bos") is not None,
        bos_direction=(signal.get("bos") or {}).get("direction", ""),
        retest_occurred=bool((signal.get("retest") or {}).get("occurred")),
        retest_held=bool((signal.get("retest") or {}).get("held")),
        sweep_present=bool((signal.get("liquidity") or {}).get("sweep")),
        sweep_direction=((signal.get("liquidity") or {}).get("sweep") or {}).get("sweep_direction", ""),
        premium_discount_zone=(loc.get("premium_discount") or {}).get("zone", ""),
        nearest_resistance=signal.get("nearest_resistance"),
        available_room_pct=float(signal.get("available_room_pct", 0.0)),
        location_supporting_score=float(loc.get("supporting_score", 0.0)),
        location_opposing_score=float(loc.get("opposing_score", 0.0)),
        entry_signal_type=signal.get("signal_type", ""),
        chase_score=float((signal.get("extension") or {}).get("chase_score", 0.0)),
        extension_label=(signal.get("extension") or {}).get("label", ""),
        displacement_from_bos_pct=float((signal.get("extension") or {}).get("displacement_pct", 0.0)),
        mfe_pct=result["mfe_pct"],
        mae_pct=result["mae_pct"],
        holding_hours=result["holding_hours"],
        r_multiple=result["r_multiple"],
        diagnostic_flag=result["diagnostic_flag"],
    )


# --------------------------------------------------------------------------- #
# Statistics & Summaries
# --------------------------------------------------------------------------- #
def _stats(subset: List[SimulatedTrade]) -> Dict[str, Any]:
    if not subset:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
            "total_pnl_usd": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "max_losing_streak": 0,
            "tp1_hit_rate": 0.0,
            "tp2_hit_rate": 0.0,
            "tp3_hit_rate": 0.0,
            "stop_loss_rate": 0.0,
            "expectancy_pct": 0.0,
            "avg_r_multiple": 0.0,
            "median_r_multiple": 0.0,
            "avg_mfe_pct": 0.0,
            "avg_mae_pct": 0.0,
            "avg_holding_hours": 0.0,
            "diagnostics": {},
        }

    wins = [t for t in subset if t.return_pct > 0]
    losses = [t for t in subset if t.return_pct <= 0]
    gross_profit = sum(t.return_pct for t in wins)
    gross_loss = abs(sum(t.return_pct for t in losses))
    n = len(subset)
    sl_exits = [t for t in subset if "stop" in t.exit_reason]
    win_rate = len(wins) / n
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0

    # Drawdown and Streak calculation
    pnl_series = [t.pnl_usd for t in subset]
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    streak = 0
    max_streak = 0
    for pnl in pnl_series:
        cum_pnl += pnl
        if cum_pnl > peak:
            peak = cum_pnl
        dd = peak - cum_pnl
        if dd > max_dd:
            max_dd = dd

        if pnl <= 0:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0

    r_multiples = sorted(t.r_multiple for t in subset)
    median_r = r_multiples[n // 2] if n > 0 else 0.0

    return {
        "trades": n,
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_return_pct": round(sum(t.return_pct for t in subset) / n, 3),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(-avg_loss, 3),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else float("inf"),
        "total_pnl_usd": round(sum(t.pnl_usd for t in subset), 2),
        "max_drawdown_usd": round(max_dd, 2),
        "max_losing_streak": max_streak,
        "tp1_hit_rate": round(sum(1 for t in subset if t.hit_tp1) / n * 100, 1),
        "tp2_hit_rate": round(sum(1 for t in subset if t.hit_tp2) / n * 100, 1),
        "tp3_hit_rate": round(sum(1 for t in subset if t.hit_tp3) / n * 100, 1),
        "stop_loss_rate": round(len(sl_exits) / n * 100, 1),
        "expectancy_pct": round(win_rate * avg_win - (1 - win_rate) * avg_loss, 3),
        "avg_r_multiple": round(sum(t.r_multiple for t in subset) / n, 2),
        "median_r_multiple": round(median_r, 2),
        "avg_mfe_pct": round(sum(t.mfe_pct for t in subset) / n, 3),
        "avg_mae_pct": round(sum(t.mae_pct for t in subset) / n, 3),
        "avg_holding_hours": round(sum(t.holding_hours for t in subset) / n, 1),
        "diagnostics": {
            flag: sum(1 for t in subset if t.diagnostic_flag == flag)
            for flag in ("POSSIBLE STOP ISSUE", "ENTRY FAILURE", "EXIT MANAGEMENT ISSUE")
        },
    }


def summarize_results(trades: List[SimulatedTrade]) -> Dict[str, Any]:
    in_sample = [t for t in trades if not t.is_out_of_sample]
    out_sample = [t for t in trades if t.is_out_of_sample]
    return {
        "overall": _stats(trades),
        "in_sample": _stats(in_sample),
        "out_of_sample": _stats(out_sample),
        "by_symbol": {s: _stats([t for t in trades if t.symbol == s]) for s in sorted({t.symbol for t in trades})},
        "by_entry_model": {m: _stats([t for t in trades if t.entry_model == m]) for m in ("A", "B", "C")},
        "by_market_regime": {
            r: _stats([t for t in trades if t.market_regime_4h == r])
            for r in ("BULLISH", "TRANSITION", "NEUTRAL", "BEARISH")
        },
        "by_location": {
            loc: _stats([t for t in trades if t.premium_discount_zone == loc])
            for loc in ("discount", "equilibrium", "premium")
        },
        "by_extension": {
            ext: _stats([t for t in trades if t.extension_label == ext])
            for ext in ("EARLY", "EXTENDED", "OVEREXTENDED")
        },
        "by_retest": {
            "Held": _stats([t for t in trades if t.retest_held]),
            "Touched_Only": _stats([t for t in trades if t.retest_occurred and not t.retest_held]),
            "None": _stats([t for t in trades if not t.retest_occurred]),
        },
    }


def print_rejection_funnel(funnel: Dict[str, int], rejections: Dict[str, int]) -> None:
    print("\n" + "=" * 68)
    print("  STRATEGY DECISION & REJECTION FUNNEL (Comprehensive Diagnostic)")
    print("=" * 68)
    print(f"  1. 1H Candles Evaluated:            {funnel.get('candles_evaluated', 0):>6}")
    print(f"  2. 4H Context Aligned (Bull/Trans): {funnel.get('4h_bullish_or_transition', 0):>6}")
    print(f"  3. 1H Structural Breaks (CHoCH/BOS):{funnel.get('bos_events', 0) + funnel.get('choch_events', 0):>6}")
    print(f"  4. Immediate Valid (Model A):       {funnel.get('immediate_valid_entries', 0):>6}")
    print(f"  5. Persistent Watches Created:      {funnel.get('watches_created', 0):>6}")
    print(f"  6. Watches Reached Preferred Zone:  {funnel.get('watches_zone_touched', 0):>6}")
    print(f"  7. Pullback Reaction Confirmed:     {funnel.get('watches_reaction_confirmed', 0):>6}")
    print(f"  8. Risk & Capital Checks Passed:    {funnel.get('watches_passed_risk', 0):>6}")
    print(f"  9. R:R & Room Checks Passed:        {funnel.get('watches_passed_rr', 0):>6}")
    print(f" 10. TOTAL EXECUTED TRADES:           {funnel.get('trades_executed', 0):>6}")
    print("-" * 68)
    print("  REJECTION REASON BREAKDOWN:")
    for reason, count in rejections.items():
        if count > 0:
            formatted_reason = reason.replace("_", " ").title()
            print(f"    • {formatted_reason:<42}: {count:>5}")
    print("-" * 68)


def print_watch_statistics(watch_stats: Dict[str, Any]) -> None:
    print("\n" + "=" * 68)
    print("  PERSISTENT WATCH STATISTICS (Lifecycle Conversions)")
    print("=" * 68)
    print(f"  Total Watches Registered:       {watch_stats.get('total_watches', 0)}")
    print(f"  Zone Touched (Pullback Arrived):{watch_stats.get('zone_touched_count', 0)} ({watch_stats.get('watch_to_zone_conversion_pct', 0)}%)")
    print(f"  Trades Entered:                 {watch_stats.get('trades_entered_count', 0)} ({watch_stats.get('watch_to_trade_conversion_pct', 0)}%)")
    print(f"  Zone -> Trade Conversion Rate:  {watch_stats.get('zone_to_trade_conversion_pct', 0)}%")
    print(f"  Invalidated by Stop Breach:     {watch_stats.get('invalidated_count', 0)}")
    print(f"  Expired (Max Candles Exceeded): {watch_stats.get('expired_count', 0)}")
    print(f"  Rejected at Entry (Risk/R:R):   {watch_stats.get('rejected_at_entry_count', 0)}")
    print(f"  Still Active / Watching:        {watch_stats.get('still_watching_count', 0)}")
    print(f"  Average Candles to Zone:        {watch_stats.get('avg_candles_to_zone', 0)} bars (~{watch_stats.get('avg_candles_to_zone', 0)}h)")
    print(f"  Average Watch Duration:         {watch_stats.get('avg_watch_duration_candles', 0)} bars")
    print("-" * 68)


def print_trade_summary(summary: Dict[str, Any], symbols: List[str], days: int) -> None:
    o = summary["overall"]
    print("\n" + "=" * 68)
    print(f"  TRADE PERFORMANCE SUMMARY — {', '.join(symbols)} ({days} Days)")
    print("=" * 68)
    print(f"  Total Trades:         {o['trades']}")
    print(f"  Win Rate:             {o['win_rate_pct']}%")
    print(f"  Profit Factor:        {o['profit_factor']}")
    print(f"  Expectancy:           {o['expectancy_pct']}% / trade")
    print(f"  Avg Return / Trade:   {o['avg_return_pct']}%")
    print(f"  Avg Win / Avg Loss:   +{o['avg_win_pct']}% / {o['avg_loss_pct']}%")
    print(f"  Avg R-Multiple:       {o['avg_r_multiple']} R (Median: {o['median_r_multiple']} R)")
    print(f"  Total P&L:            ${o['total_pnl_usd']:,.2f}")
    print(f"  Max Drawdown:         ${o['max_drawdown_usd']:,.2f}")
    print(f"  Max Losing Streak:    {o['max_losing_streak']} trades")
    print(f"  TP1 / TP2 / TP3 Hit:  {o['tp1_hit_rate']}% / {o['tp2_hit_rate']}% / {o['tp3_hit_rate']}%")
    print(f"  Stop-Loss Hit Rate:   {o['stop_loss_rate']}%")
    print(f"  Avg Holding Time:     {o['avg_holding_hours']} hours")
    print(f"  Avg MFE / Avg MAE:    +{o['avg_mfe_pct']}% / -{o['avg_mae_pct']}%")
    print("-" * 68)
    print("  IN-SAMPLE vs OUT-OF-SAMPLE")
    for name in ("in_sample", "out_of_sample"):
        s = summary[name]
        if s["trades"] > 0:
            print(f"    {name:<14}: {s['trades']:>3} trades | Win {s['win_rate_pct']:>5}% | PF {s['profit_factor']:>4} | P&L ${s['total_pnl_usd']:>9,.2f}")
    print("-" * 68)
    print("  BREAKDOWN BY ASSET")
    for s, stats in summary.get("by_symbol", {}).items():
        if stats["trades"] > 0:
            print(f"    {s:<14}: {stats['trades']:>3} trades | Win {stats['win_rate_pct']:>5}% | PF {stats['profit_factor']:>4} | P&L ${stats['total_pnl_usd']:>9,.2f} | Exp {stats['expectancy_pct']:>+5.2f}%")
    print("-" * 68)
    print("  BREAKDOWN BY ENTRY MODEL")
    for m, s in summary["by_entry_model"].items():
        if s["trades"] > 0:
            print(f"    Model {m:<8}: {s['trades']:>3} trades | Win {s['win_rate_pct']:>5}% | PF {s['profit_factor']:>4} | P&L ${s['total_pnl_usd']:>9,.2f}")
    print("-" * 68)
    print("  BREAKDOWN BY 4H MARKET REGIME")
    for r, s in summary["by_market_regime"].items():
        if s["trades"] > 0:
            print(f"    {r:<14}: {s['trades']:>3} trades | Win {s['win_rate_pct']:>5}% | PF {s['profit_factor']:>4} | P&L ${s['total_pnl_usd']:>9,.2f}")
    print("-" * 68)
    print("  BREAKDOWN BY LOCATION (PREMIUM/DISCOUNT)")
    for loc, s in summary["by_location"].items():
        if s["trades"] > 0:
            print(f"    {loc.title():<14}: {s['trades']:>3} trades | Win {s['win_rate_pct']:>5}% | PF {s['profit_factor']:>4} | P&L ${s['total_pnl_usd']:>9,.2f}")
    print("-" * 68)
    print("  DIAGNOSTICS & EXCURSIONS:")
    for flag, cnt in o.get("diagnostics", {}).items():
        if cnt > 0:
            print(f"    • {flag}: {cnt} trades")
    print("=" * 68 + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest the Dznani Price Action Engine against Binance history.")
    p.add_argument("--symbols", nargs="+", help="Symbols to test, e.g. BTC/USDT ETH/USDT")
    p.add_argument("--watchlist", action="store_true", help="Use symbols from watchlist.json instead of --symbols")
    p.add_argument("--days", type=int, default=180, help="Days of 1H history to test (default: 180)")
    p.add_argument("--oos-days", type=int, default=30, help="Reserve the most recent N days as out-of-sample (default: 30)")
    p.add_argument("--min-rr", type=float, default=None, help="Override minimum_rr for this run")
    p.add_argument("--atr-buffer", type=float, default=None, help="Override sl_atr_buffer_mult (default: 0.5)")
    p.add_argument("--max-watch-candles", type=int, default=None, help="Override max_watch_candles (default: 36)")
    p.add_argument("--output", default=None, help="CSV path for the full trade log")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml (for Binance API keys, if any)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging("logs", "INFO")

    db = Database()
    settings = dict(DEFAULT_SETTINGS)
    settings.update(db.get_settings())
    if args.min_rr is not None:
        settings["minimum_rr"] = args.min_rr
    if args.atr_buffer is not None:
        settings["sl_atr_buffer_mult"] = args.atr_buffer
    if args.max_watch_candles is not None:
        settings["max_watch_candles"] = args.max_watch_candles

    if args.watchlist:
        symbols = db.get_watchlist()
        if not symbols:
            print("watchlist.json is empty — pass --symbols instead.")
            sys.exit(1)
    elif args.symbols:
        symbols = [s.upper() if "/" in s else f"{s.upper()}/USDT" for s in args.symbols]
    else:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        print(f"No --symbols or --watchlist given — defaulting to {symbols}")

    binance_key, binance_secret = None, None
    try:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        binance_key = cfg.get("binance", {}).get("api_key") or None
        binance_secret = cfg.get("binance", {}).get("api_secret") or None
    except FileNotFoundError:
        pass

    exchange = BinanceExchange(api_key=binance_key, api_secret=binance_secret)
    until_ms = exchange.exchange.milliseconds()
    since_ms_1h = until_ms - args.days * 86400 * 1000
    since_ms_4h = until_ms - (args.days + 30) * 86400 * 1000
    oos_cutoff = pd.Timestamp(until_ms - args.oos_days * 86400 * 1000, unit="ms", tz="UTC")

    all_trades: List[SimulatedTrade] = []
    combined_funnel: Dict[str, int] = {}
    combined_rejections: Dict[str, int] = {}
    combined_watch_stats: Dict[str, Any] = {
        "total_watches": 0,
        "zone_touched_count": 0,
        "trades_entered_count": 0,
        "invalidated_count": 0,
        "expired_count": 0,
        "rejected_at_entry_count": 0,
        "still_watching_count": 0,
    }

    for symbol in symbols:
        print(f"Fetching {args.days}d of 1H + 4H history for {symbol} from Binance CCXT…")
        try:
            df_1h = exchange.fetch_ohlcv_range(symbol, timeframe="1h", since_ms=since_ms_1h, until_ms=until_ms)
            df_4h = exchange.fetch_ohlcv_range(symbol, timeframe="4h", since_ms=since_ms_4h, until_ms=until_ms)
        except Exception as e:
            print(f"  Failed to fetch {symbol}: {e}")
            continue

        print(f"  Loaded {len(df_1h)} 1H / {len(df_4h)} 4H candles. Running stateful walk-forward simulation…")
        result = run_backtest_symbol(df_4h, df_1h, symbol, settings, oos_cutoff_ts=oos_cutoff)
        trades = result["trades"]
        all_trades.extend(trades)

        for k, v in result["funnel"].items():
            combined_funnel[k] = combined_funnel.get(k, 0) + v
        for k, v in result["rejections"].items():
            combined_rejections[k] = combined_rejections.get(k, 0) + v
        for k in ("total_watches", "zone_touched_count", "trades_entered_count", "invalidated_count", "expired_count", "rejected_at_entry_count", "still_watching_count"):
            combined_watch_stats[k] += result["watch_stats"].get(k, 0)

        print(f"  {len(trades)} simulated trade(s) executed for {symbol}.")

    # Compute aggregate watch percentages
    tot_w = combined_watch_stats["total_watches"]
    zt = combined_watch_stats["zone_touched_count"]
    te = combined_watch_stats["trades_entered_count"]
    combined_watch_stats["watch_to_zone_conversion_pct"] = round(zt / tot_w * 100, 1) if tot_w else 0.0
    combined_watch_stats["watch_to_trade_conversion_pct"] = round(te / tot_w * 100, 1) if tot_w else 0.0
    combined_watch_stats["zone_to_trade_conversion_pct"] = round(te / zt * 100, 1) if zt else 0.0

    print_rejection_funnel(combined_funnel, combined_rejections)
    print_watch_statistics(combined_watch_stats)

    summary = summarize_results(all_trades)
    summary["funnel"] = combined_funnel
    summary["rejections"] = combined_rejections
    summary["watch_stats"] = combined_watch_stats

    print_trade_summary(summary, symbols, args.days)

    output_path = args.output or f"backtest_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame([asdict(t) for t in all_trades]).to_csv(output_path, index=False)
    print(f"Full trade log written to {output_path}")

    summary_path = output_path.replace(".csv", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary statistics written to {summary_path}")


if __name__ == "__main__":
    main()
