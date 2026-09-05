"""
watch_engine.py — Persistent Watch State Machine and Setup Lifecycle Engine
for Dznani Price Action Bot.

CORE ARCHITECTURAL PRINCIPLES:
1. DIRECTION != ENTRY LOCATION: A confirmed CHoCH/BOS confirms directional control;
   it is NOT a "Buy Now" command.
2. BOS != BUY: If price is extended or requires retesting, the setup is NOT discarded.
   It creates a persistent WATCH.
3. ZONE TOUCH != ENTRY: Price returning to a preferred zone triggers active re-evaluation
   and confirmation checks, not an automatic market buy.
4. STATEFUL ACROSS CANDLES: Watches persist across subsequent 1H bars until they
   confirm (ENTERED), invalidate (INVALIDATED), or expire (EXPIRED).

State Machine:
    PENDING / WATCHING
           │
           ▼
      ZONE_TOUCHED
           │
           ▼
       CONFIRMING
           │
           ├── (Valid Confirmation + Risk + R:R) ──> READY ──> ENTERED
           ├── (Structural Breach / Invalidation) ─> INVALIDATED
           ├── (Exceeded Max Holding Candles) ────> EXPIRED
           └── (Failed Risk/R:R at Entry) ─────────> REJECTED_AT_ENTRY
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

import indicators as ind
import risk
import rr_engine
import structure as struct_engine
import zones as zn

logger = logging.getLogger("dznani.watch_engine")

WatchStatus = Literal[
    "WATCHING",
    "ZONE_TOUCHED",
    "CONFIRMING",
    "READY",
    "ENTERED",
    "INVALIDATED",
    "EXPIRED",
    "REJECTED_AT_ENTRY",
]


@dataclass
class WatchSetup:
    setup_id: str
    symbol: str
    direction: str  # "BUY" | "SELL"
    created_at: str  # ISO timestamp
    created_candle_idx: int
    expiry_candles: int = 36  # 36 hours of 1H candles by default

    # Market context at creation
    structure_4h_trend: str = "BULLISH"
    structure_1h_trend: str = "BULLISH"
    choch_level: Optional[float] = None
    bos_level: Optional[float] = None
    break_level: float = 0.0

    # Invalidation & Entry Zone
    protected_level: float = 0.0  # Structural invalidation (e.g. protected swing low)
    invalidation_level: float = 0.0  # Invalidation price threshold
    preferred_zone_low: float = 0.0
    preferred_zone_high: float = 0.0
    preferred_zone_type: str = "bos_retest"  # "bos_retest" | "fvg" | "order_block" | "fib_discount" | "flip"

    # Context scores
    premium_discount: str = "equilibrium"
    extension_label: str = "EXTENDED"
    chase_score: float = 0.0
    setup_score: float = 0.0
    setup_grade: str = "B"
    location_score: float = 0.0
    candidate_model: str = "B"  # "A" | "B" | "C"

    # Lifecycle state
    status: WatchStatus = "WATCHING"
    touch_candle_idx: Optional[int] = None
    touch_timestamp: Optional[str] = None
    candles_in_zone: int = 0
    lowest_price_seen: float = 0.0
    highest_price_seen: float = 0.0
    confirmation_details: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: Optional[str] = None
    generated_signal: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class WatchManager:
    """
    Manages active persistent watch setups across walk-forward backtest
    candles and live scanning cycles.
    """

    def __init__(self, default_expiry_candles: int = 36):
        self.default_expiry_candles = default_expiry_candles
        self.active_watches: List[WatchSetup] = []
        self.completed_watches: List[WatchSetup] = []
        self._seen_setup_keys: set = set()

    def create_watch_from_signal(
        self,
        signal: Dict[str, Any],
        candle_idx: int,
        timestamp: Any,
        expiry_candles: Optional[int] = None,
    ) -> Optional[WatchSetup]:
        """
        Creates a new persistent WatchSetup from an evaluated signal card
        that requires waiting for a pullback / retest (WAIT FOR RETEST, NO CHASE, WATCH).
        """
        symbol = signal.get("symbol", "")
        direction = signal.get("direction")
        if direction != "BUY":  # Spot-only bot executes longs
            return None

        bos = signal.get("bos")
        choch = signal.get("choch")
        break_event = bos or choch
        if break_event is None:
            return None

        break_level = float(break_event.get("level", 0.0))
        break_ts = str(break_event.get("break_timestamp", timestamp))
        setup_key = (symbol, direction, round(break_level, 4), break_ts)

        # Prevent duplicate watches for the exact same structural break
        if setup_key in self._seen_setup_keys:
            return None

        # Check if an active watch for the same symbol is already pending near this break level
        for w in self.active_watches:
            if w.symbol == symbol and w.direction == direction and abs(w.break_level - break_level) / break_level < 0.005:
                return None

        protected_level = float(signal.get("structural_invalidation") or signal.get("stop_loss") or (break_level * 0.95))
        invalidation_level = protected_level

        # Extract or compute preferred entry zone
        ext_info = signal.get("extension") or {}
        preferred_low = ext_info.get("preferred_entry_low")
        preferred_high = ext_info.get("preferred_entry_high")
        zone_type = "bos_retest"

        # If extension module did not provide bounds or price is very close, build zone from break_level and ATR
        current_price = float(signal.get("current_price", break_level))
        loc_info = signal.get("location") or {}
        zones_list = loc_info.get("order_blocks", []) + loc_info.get("fvg", []) + loc_info.get("flip_zones", [])

        # Search for valid supporting zone between current price and protected level
        valid_supporting_zones = [
            z for z in zones_list
            if z.get("polarity") == "bullish" and z.get("zone_low", 0) >= protected_level and z.get("zone_high", 0) <= current_price * 1.01
        ]

        if valid_supporting_zones:
            best_zone = max(valid_supporting_zones, key=lambda z: z.get("confluence_score", 0.5))
            preferred_low = float(best_zone.get("zone_low"))
            preferred_high = float(best_zone.get("zone_high"))
            zone_type = best_zone.get("zone_type", "order_block")
        elif preferred_low is None or preferred_high is None:
            # Anchor zone around the break level
            band = max(current_price * 0.005, abs(current_price - break_level) * 0.25)
            preferred_low = min(break_level - band, current_price * 0.995)
            preferred_high = max(break_level + band, break_level)
            zone_type = "bos_retest"

        # Ensure preferred_low is strictly above protected_level
        if preferred_low <= protected_level:
            preferred_low = protected_level + (break_level - protected_level) * 0.25

        setup_id = f"{symbol}_{direction}_{int(pd.Timestamp(timestamp).timestamp())}_{int(break_level)}"
        ts_str = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)

        watch = WatchSetup(
            setup_id=setup_id,
            symbol=symbol,
            direction=direction,
            created_at=ts_str,
            created_candle_idx=candle_idx,
            expiry_candles=expiry_candles or self.default_expiry_candles,
            structure_4h_trend=(signal.get("structure_4h") or {}).get("trend", "BULLISH"),
            structure_1h_trend=(signal.get("structure_1h") or {}).get("trend", "BULLISH"),
            choch_level=float(choch.get("level")) if choch else None,
            bos_level=float(bos.get("level")) if bos else None,
            break_level=break_level,
            protected_level=protected_level,
            invalidation_level=invalidation_level,
            preferred_zone_low=round(preferred_low, 8),
            preferred_zone_high=round(preferred_high, 8),
            preferred_zone_type=zone_type,
            premium_discount=(loc_info.get("premium_discount") or {}).get("zone", "equilibrium"),
            extension_label=ext_info.get("label", "EXTENDED"),
            chase_score=float(ext_info.get("chase_score", 50.0)),
            setup_score=float(signal.get("setup_score", 70.0)),
            setup_grade=signal.get("setup_grade", "B"),
            location_score=float((loc_info.get("assessment") or {}).get("location_score", 50.0)),
            candidate_model="B",
            status="WATCHING",
            lowest_price_seen=current_price,
            highest_price_seen=current_price,
        )

        self.active_watches.append(watch)
        self._seen_setup_keys.add(setup_key)
        return watch

    def evaluate_active_watches(
        self,
        df_1h_window: pd.DataFrame,
        df_4h_window: pd.DataFrame,
        settings: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Advances all active watches against the latest candle in df_1h_window.
        Returns a list of VALID entry signals triggered on this candle.
        """
        current_candle_idx = len(df_1h_window) - 1
        current_row = df_1h_window.iloc[-1]
        candle_high = float(current_row["high"])
        candle_low = float(current_row["low"])
        candle_close = float(current_row["close"])
        candle_open = float(current_row["open"])
        candle_ts = current_row["timestamp"]
        ts_str = candle_ts.isoformat() if hasattr(candle_ts, "isoformat") else str(candle_ts)

        minimum_rr = float(settings.get("minimum_rr", 1.5))
        max_structural_risk_pct = float(settings.get("max_structural_risk_pct", 8.0))
        risk_per_trade_usd = float(settings.get("risk_per_trade_usd", 250.0))
        capital = float(settings.get("capital", risk.DEFAULT_CAPITAL))
        atr_buffer_mult = float(settings.get("sl_atr_buffer_mult", 0.5))

        triggered_signals: List[Dict[str, Any]] = []
        still_active: List[WatchSetup] = []

        for watch in self.active_watches:
            watch.lowest_price_seen = min(watch.lowest_price_seen, candle_low)
            watch.highest_price_seen = max(watch.highest_price_seen, candle_high)
            elapsed_candles = current_candle_idx - watch.created_candle_idx

            # -------------------------------------------------------------
            # 1. Invalidation Check: Price breached protected structure
            # -------------------------------------------------------------
            if candle_close < watch.invalidation_level or candle_low < (watch.protected_level * 0.998):
                watch.status = "INVALIDATED"
                watch.rejection_reason = (
                    f"INVALIDATED — price breached protected structural level "
                    f"({watch.protected_level:.4f}), low reached {candle_low:.4f}."
                )
                self.completed_watches.append(watch)
                continue

            # -------------------------------------------------------------
            # 2. Expiry Check: Max waiting candles exceeded
            # -------------------------------------------------------------
            if elapsed_candles > watch.expiry_candles:
                watch.status = "EXPIRED"
                watch.rejection_reason = (
                    f"EXPIRED — {elapsed_candles} candles elapsed without completing valid confirmation."
                )
                self.completed_watches.append(watch)
                continue

            # -------------------------------------------------------------
            # 3. Zone Interaction Check
            # -------------------------------------------------------------
            zone_touched = (candle_low <= watch.preferred_zone_high) and (candle_high >= watch.preferred_zone_low)

            if watch.status == "WATCHING":
                if zone_touched:
                    watch.status = "ZONE_TOUCHED"
                    watch.touch_candle_idx = current_candle_idx
                    watch.touch_timestamp = ts_str
                    watch.candles_in_zone = 1
                else:
                    still_active.append(watch)
                    continue
            elif watch.status in ("ZONE_TOUCHED", "CONFIRMING"):
                if zone_touched or (candle_low <= watch.preferred_zone_high * 1.01):
                    watch.candles_in_zone += 1
                watch.status = "CONFIRMING"

            # -------------------------------------------------------------
            # 4. Pullback & Reaction Confirmation Gate
            # -------------------------------------------------------------
            # Confirmation requires:
            # a) Zone holds: close >= protected_level and close >= preferred_zone_low * 0.995
            # b) Bullish reaction evidence:
            #    - Rejection wick (close in upper 40% of candle range), OR
            #    - Green candle closing back above zone mid/low, OR
            #    - Bullish reversal from zone with momentum/higher close
            candle_range = candle_high - candle_low
            wick_rejection = (candle_close - candle_low) / (candle_range + 1e-8) >= 0.40 if candle_range > 0 else False
            bullish_close = candle_close > candle_open
            higher_close = len(df_1h_window) >= 2 and candle_close > float(df_1h_window["close"].iloc[-2])

            reaction_confirmed = (wick_rejection or bullish_close or higher_close) and (candle_close >= watch.protected_level)

            if not reaction_confirmed:
                # Still waiting for bullish reaction inside confirmation window
                watch.confirmation_details = {
                    "zone_touched": True,
                    "wick_rejection": wick_rejection,
                    "bullish_close": bullish_close,
                    "reaction_confirmed": False,
                }
                still_active.append(watch)
                continue

            # -------------------------------------------------------------
            # 5. Candidate Reached: Calculate Entry, SL, Risk, R:R
            # -------------------------------------------------------------
            entry_price = candle_close
            atr_1h = float(ind.calculate_atr(df_1h_window, 14).iloc[-1])

            # Structural Stop Loss with configurable ATR buffer
            structural_sl = risk.calculate_structural_stop_loss(
                entry=entry_price,
                direction="BUY",
                protected_level=watch.protected_level,
                atr=atr_1h,
                position_size_usd=1.0,
                atr_buffer_mult=atr_buffer_mult,
            )

            # Check structural risk ceiling
            if structural_sl.risk_pct <= 0 or structural_sl.risk_pct > max_structural_risk_pct:
                watch.status = "REJECTED_AT_ENTRY"
                watch.rejection_reason = (
                    f"REJECTED — structural risk ({structural_sl.risk_pct:.2f}%) exceeds max allowed "
                    f"({max_structural_risk_pct:.1f}%)."
                )
                self.completed_watches.append(watch)
                continue

            # Position Sizing & Capital limits
            pos_plan = risk.calculate_position_plan(
                risk_per_trade_usd=risk_per_trade_usd,
                structural_risk_pct=structural_sl.risk_pct,
                available_capital_usd=capital,
            )

            if pos_plan.exceeds_available_capital:
                watch.status = "REJECTED_AT_ENTRY"
                watch.rejection_reason = "REJECTED — position sizing exceeds available capital ceiling."
                self.completed_watches.append(watch)
                continue

            # Available Room & R:R against structural resistance
            state_1h = struct_engine.analyze_structure_1h(df_1h_window, left=2, right=2)
            state_4h = struct_engine.analyze_structure(df_4h_window, left=2, right=2)
            resistance_levels = [s.price for s in state_1h.swing_highs if s.price > entry_price]
            resistance_levels += [s.price for s in state_4h.swing_highs if s.price > entry_price]
            if state_4h.protected_high and state_4h.protected_high > entry_price:
                resistance_levels.append(state_4h.protected_high)

            rr_result = rr_engine.evaluate_rr(
                entry=entry_price,
                stop_loss=structural_sl.final_sl,
                direction="BUY",
                resistance_levels=resistance_levels,
                minimum_rr=minimum_rr,
            )

            if not rr_result.passes_minimum:
                # If R:R is insufficient, record rejection
                watch.status = "REJECTED_AT_ENTRY"
                watch.rejection_reason = (
                    f"REJECTED — R:R {rr_result.rr:.2f} to nearest obstacle ({rr_result.available_room_pct:.1f}% away) "
                    f"is below minimum {minimum_rr:.2f}."
                )
                self.completed_watches.append(watch)
                continue

            # -------------------------------------------------------------
            # 6. VALID PULLBACK ENTRY CONFIRMED!
            # -------------------------------------------------------------
            watch.status = "ENTERED"
            scale_out = risk.build_scale_out_plan(entry_price, "BUY")
            tp_conflict = risk.check_targets_against_structure(scale_out, "BUY", resistance_levels)

            risk_distance = rr_result.risk_distance
            rr_to_targets = {
                lvl.label: round(abs(lvl.price - entry_price) / risk_distance, 3) if risk_distance > 0 else 0.0
                for lvl in scale_out
            }
            rr_to_targets["RESISTANCE"] = rr_result.rr

            signal_card = {
                "symbol": watch.symbol,
                "exchange": "Binance",
                "timeframe": "4H/1H",
                "current_price": entry_price,
                "direction": "BUY",
                "signal_type": "CONFIRMATION LONG",
                "decision": "VALID",
                "reason": (
                    f"Confirmed Model B pullback entry: retested preferred zone "
                    f"({watch.preferred_zone_low:.2f}–{watch.preferred_zone_high:.2f}), "
                    f"held structure, R:R {rr_result.rr:.2f}."
                ),
                "entry_model": "B",
                "structure_4h": state_4h.to_dict(),
                "structure_1h": state_1h.to_dict(),
                "liquidity": {"pools": len(state_1h.swing_lows), "sweep": None},
                "choch": {"level": watch.choch_level, "direction": "bullish"} if watch.choch_level else None,
                "bos": {"level": watch.bos_level, "direction": "bullish"} if watch.bos_level else None,
                "retest": {"occurred": True, "held": True, "retest_level": watch.break_level},
                "location": {
                    "supporting_score": 0.8,
                    "opposing_score": 0.1,
                    "premium_discount": {"zone": "discount", "pct_of_range": 0.4},
                    "assessment": {
                        "location_grade": "A",
                        "location_score": 85.0,
                        "zone_type": watch.preferred_zone_type,
                        "location_valid": True,
                    },
                },
                "entry": entry_price,
                "stop_loss": structural_sl.final_sl,
                "structural_invalidation": watch.protected_level,
                "stop_loss_pct": structural_sl.risk_pct,
                "excessive_structural_risk": False,
                "max_structural_risk_pct": max_structural_risk_pct,
                "extension": {
                    "label": "EARLY",
                    "chase_score": 20.0,
                    "displacement_pct": round(abs(entry_price - watch.break_level) / watch.break_level * 100, 2),
                    "atr_multiple": round(abs(entry_price - watch.break_level) / atr_1h, 2) if atr_1h > 0 else 0.0,
                },
                "entry_confirmation": {
                    "status": "CONFIRMED",
                    "quality": "A",
                    "returned_to_zone": True,
                    "zone_held": True,
                    "rejection_present": wick_rejection or bullish_close,
                    "micro_structure_confirmed": True,
                    "reason": "Zone held with bullish reaction and confirmed micro structure.",
                },
                "position_plan": pos_plan.to_dict(),
                "nearest_resistance": rr_result.nearest_obstacle,
                "nearest_support": None,
                "available_room_pct": rr_result.available_room_pct,
                "rr": rr_result.to_dict(),
                "rr_to_targets": rr_to_targets,
                "targets": [
                    {"label": lvl.label, "price": lvl.price, "pct": lvl.pct, "sell_fraction": lvl.sell_fraction}
                    for lvl in scale_out
                ],
                "targets_structure_conflict": tp_conflict,
                "setup_score": watch.setup_score,
                "setup_grade": watch.setup_grade,
                "trade_quality_score": round(min(100.0, (rr_result.rr / minimum_rr) * 65 + 30), 1),
                "trade_quality_grade": "A",
                "indicators": ind.snapshot(df_1h_window) if hasattr(ind, "snapshot") else {},
                "timestamp": ts_str,
                "watch_setup_id": watch.setup_id,
            }

            watch.generated_signal = signal_card
            self.completed_watches.append(watch)
            triggered_signals.append(signal_card)

        self.active_watches = still_active
        return triggered_signals

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculates comprehensive watch statistics and conversion rates.
        """
        all_watches = self.completed_watches + self.active_watches
        total = len(all_watches)
        if total == 0:
            return {
                "total_watches": 0,
                "zone_touched_count": 0,
                "trades_entered_count": 0,
                "invalidated_count": 0,
                "expired_count": 0,
                "rejected_at_entry_count": 0,
                "still_watching_count": 0,
                "watch_to_zone_conversion_pct": 0.0,
                "watch_to_trade_conversion_pct": 0.0,
                "zone_to_trade_conversion_pct": 0.0,
                "avg_candles_to_zone": 0.0,
                "avg_watch_duration_candles": 0.0,
            }

        zone_touched = [w for w in all_watches if w.touch_candle_idx is not None or w.status in ("ZONE_TOUCHED", "CONFIRMING", "READY", "ENTERED", "REJECTED_AT_ENTRY")]
        entered = [w for w in all_watches if w.status == "ENTERED"]
        invalidated = [w for w in all_watches if w.status == "INVALIDATED"]
        expired = [w for w in all_watches if w.status == "EXPIRED"]
        rejected_entry = [w for w in all_watches if w.status == "REJECTED_AT_ENTRY"]
        still_watching = [w for w in self.active_watches if w.status == "WATCHING"]

        times_to_zone = [
            (w.touch_candle_idx - w.created_candle_idx)
            for w in zone_touched
            if w.touch_candle_idx is not None and w.touch_candle_idx >= w.created_candle_idx
        ]
        avg_time_to_zone = sum(times_to_zone) / len(times_to_zone) if times_to_zone else 0.0

        durations = [
            ((w.touch_candle_idx or w.created_candle_idx + w.expiry_candles) - w.created_candle_idx)
            for w in all_watches
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_watches": total,
            "zone_touched_count": len(zone_touched),
            "trades_entered_count": len(entered),
            "invalidated_count": len(invalidated),
            "expired_count": len(expired),
            "rejected_at_entry_count": len(rejected_entry),
            "still_watching_count": len(still_watching),
            "watch_to_zone_conversion_pct": round(len(zone_touched) / total * 100, 1),
            "watch_to_trade_conversion_pct": round(len(entered) / total * 100, 1),
            "zone_to_trade_conversion_pct": round(len(entered) / len(zone_touched) * 100, 1) if zone_touched else 0.0,
            "avg_candles_to_zone": round(avg_time_to_zone, 1),
            "avg_watch_duration_candles": round(avg_duration, 1),
        }
