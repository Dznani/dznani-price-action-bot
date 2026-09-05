--- impulse_detector.py (原始)


+++ impulse_detector.py (修改后)
"""
impulse_detector.py — Impulse Discovery Layer for Dznani Price Action Bot.

This module implements LAYER 0 of the strategy architecture:

    LAYER 0 — IMPULSE DISCOVERY (this module)
    ↓
    LAYER 1 — MARKET CONTEXT
    ↓
    LAYER 2 — MARKET STRUCTURE
    ↓
    LAYER 3 — LIQUIDITY
    ↓
    LAYER 4 — LOCATION
    ↓
    LAYER 5 — EXTENSION / CHASE FILTER
    ↓
    LAYER 6 — PERSISTENT WATCH
    ↓
    LAYER 7 — ENTRY CONFIRMATION
    ↓
    LAYER 8 — RISK + R:R
    ↓
    FINAL DECISION

PHILOSOPHY:
    INDICATORS = DISCOVERY / RADAR
    PRICE ACTION = DIAGNOSIS + EXECUTION

Indicators must NEVER directly create a BUY signal. They can only create
SCOUTED IMPULSE CANDIDATES that then flow through the existing Price Action
execution engine.

KEY FEATURES:
    1. Pivot-based RSI divergence detection (Regular Bullish & Hidden Bullish)
    2. MACD momentum discovery (histogram contraction/expansion, recovery)
    3. Configurable Discovery Score (0-100)
    4. No lookahead bias - pivots confirmed only when valid
    5. Integration with Watch Engine for persistent tracking
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

import indicators as ind

logger = logging.getLogger("dznani.impulse_detector")


# --------------------------------------------------------------------------- #
# Data Models
# --------------------------------------------------------------------------- #

@dataclass
class RSIPivot:
    """Represents a confirmed RSI pivot point."""
    index: int
    timestamp: str
    price: float
    rsi_value: float
    pivot_type: str  # "low" | "high"
    confirmed_at: int  # candle index when this pivot became confirmed


@dataclass
class RSIDivergenceSignal:
    """Represents a detected RSI divergence."""
    divergence_type: Optional[str] = None  # "regular_bullish" | "hidden_bullish" | "regular_bearish" | "hidden_bearish" | None
    price_pivot_1: Optional[RSIPivot] = None
    price_pivot_2: Optional[RSIPivot] = None
    rsi_pivot_1: Optional[RSIPivot] = None
    rsi_pivot_2: Optional[RSIPivot] = None
    confidence: float = 0.0  # 0-1 score based on pivot clarity and separation
    detected_at_index: int = 0
    detected_at_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "divergence_type": self.divergence_type,
            "price_pivot_1": asdict(self.price_pivot_1) if self.price_pivot_1 else None,
            "price_pivot_2": asdict(self.price_pivot_2) if self.price_pivot_2 else None,
            "rsi_pivot_1": asdict(self.rsi_pivot_1) if self.rsi_pivot_1 else None,
            "rsi_pivot_2": asdict(self.rsi_pivot_2) if self.rsi_pivot_2 else None,
            "confidence": round(self.confidence, 3),
            "detected_at_index": self.detected_at_index,
            "detected_at_timestamp": self.detected_at_timestamp,
        }


@dataclass
class MACDMomentumState:
    """Represents the current MACD momentum condition."""
    macd_line: float = 0.0
    signal_line: float = 0.0
    histogram: float = 0.0
    histogram_trend: str = "neutral"  # "contracting_negative" | "expanding_negative" | "contracting_positive" | "expanding_positive" | "neutral"
    momentum_state: str = "neutral"  # "recovery" | "weakening" | "acceleration" | "deceleration" | "neutral"
    crossover_recent: bool = False
    crossover_direction: Optional[str] = None  # "bullish" | "bearish" | None
    confidence: float = 0.0  # 0-1 score based on momentum clarity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "macd_line": round(self.macd_line, 4),
            "signal_line": round(self.signal_line, 4),
            "histogram": round(self.histogram, 4),
            "histogram_trend": self.histogram_trend,
            "momentum_state": self.momentum_state,
            "crossover_recent": self.crossover_recent,
            "crossover_direction": self.crossover_direction,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ImpulseCandidate:
    """
    Represents a scouted impulse candidate from indicator discovery.
    This is NOT a trade signal - it's an input to the Price Action engine.
    """
    candidate_id: str
    symbol: str
    direction: str  # "BUY" | "SELL" (but only BUY executed in spot bot)
    created_at: str
    created_candle_idx: int
    expiry_candles: int = 50  # Max candles to wait for structure to form

    # Discovery components
    discovery_score: float = 0.0  # 0-100
    rsi_divergence: Optional[RSIDivergenceSignal] = None
    macd_state: Optional[MACDMomentumState] = None

    # Lifecycle state (extends watch engine states)
    status: str = "SCOUTED"  # SCOUTED -> STRUCTURE_FORMING -> STRUCTURE_CONFIRMED -> WATCHING_PULLBACK -> ZONE_TOUCHED -> CONFIRMING -> READY -> ENTERED
                             # Terminal: INVALIDATED | EXPIRED | REJECTED_AT_ENTRY

    # Structural anchor (populated by price action analysis)
    structural_anchor_level: Optional[float] = None
    invalidation_level: Optional[float] = None
    expiry_time: Optional[str] = None

    # Tracking
    highest_price_seen: float = 0.0
    lowest_price_seen: float = 0.0
    structure_confirmed_at: Optional[str] = None
    zone_touched_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "created_at": self.created_at,
            "created_candle_idx": self.created_candle_idx,
            "expiry_candles": self.expiry_candles,
            "discovery_score": round(self.discovery_score, 2),
            "rsi_divergence": self.rsi_divergence.to_dict() if self.rsi_divergence else None,
            "macd_state": self.macd_state.to_dict() if self.macd_state else None,
            "status": self.status,
            "structural_anchor_level": self.structural_anchor_level,
            "invalidation_level": self.invalidation_level,
            "expiry_time": self.expiry_time,
            "highest_price_seen": self.highest_price_seen,
            "lowest_price_seen": self.lowest_price_seen,
            "structure_confirmed_at": self.structure_confirmed_at,
            "zone_touched_at": self.zone_touched_at,
        }


# --------------------------------------------------------------------------- #
# Configuration Defaults
# --------------------------------------------------------------------------- #

DEFAULT_IMPULSE_CONFIG = {
    "enabled": True,

    "rsi": {
        "enabled": True,
        "period": 14,
        "regular_bullish_enabled": True,
        "hidden_bullish_enabled": True,
        "regular_bearish_enabled": False,  # Spot bot focuses on longs
        "hidden_bearish_enabled": False,
        "pivot_lookback": 5,  # Candles on each side for pivot confirmation
        "min_rsi_threshold": 25,  # For bullish divergence, RSI should be in oversold territory
        "max_rsi_threshold": 75,  # For bearish divergence, RSI should be in overbought territory
    },

    "macd": {
        "enabled": True,
        "fast": 12,
        "slow": 26,
        "signal": 9,
        "histogram_recovery_enabled": True,
        "histogram_expansion_enabled": True,
        "crossover_enabled": False,  # Less reliable, disabled by default
    },

    "scoring": {
        "rsi_regular_bullish_weight": 35,
        "rsi_hidden_bullish_weight": 20,
        "rsi_bearish_weight": 0,  # Disabled for spot-long focus
        "macd_recovery_weight": 25,
        "macd_expansion_weight": 20,
        "macd_crossover_weight": 0,
        "htf_context_bonus": 10,  # Bonus if 4H trend aligns
        "liquidity_sweep_bonus": 10,  # Bonus if liquidity sweep detected

        "thresholds": {
            "candidate_score": 30,  # Minimum score to create a candidate
            "strong_candidate_score": 60,  # Score indicating high-priority candidate
        },
    },

    "candidate_management": {
        "default_expiry_candles": 50,
        "prevent_duplicates": True,
        "duplicate_window_candles": 10,
    },
}


# --------------------------------------------------------------------------- #
# RSI Pivot Detection
# --------------------------------------------------------------------------- #

def detect_rsi_pivots(
    df: pd.DataFrame,
    rsi_series: pd.Series,
    lookback: int = 5,
    min_threshold: float = 25,
    max_threshold: float = 75,
) -> Tuple[List[RSIPivot], List[RSIPivot]]:
    """
    Detect confirmed RSI pivot lows and highs.

    A pivot is only confirmed after `lookback` candles have passed since the
    potential pivot point, preventing lookahead bias.

    Returns:
        (pivot_lows, pivot_highs): List of confirmed RSI pivots
    """
    if len(df) < lookback * 2 + 1:
        return [], []

    pivot_lows: List[RSIPivot] = []
    pivot_highs: List[RSIPivot] = []

    # We can only confirm pivots up to (len - lookback - 1) to avoid lookahead
    confirmable_end = len(df) - lookback - 1

    for i in range(lookback, confirmable_end + 1):
        # Check for pivot low
        rsi_window = rsi_series.iloc[i - lookback:i + lookback + 1]
        price_window = df["close"].iloc[i - lookback:i + lookback + 1]

        if len(rsi_window) < lookback * 2 + 1:
            continue

        rsi_center = rsi_window.iloc[lookback]
        price_center = price_window.iloc[lookback]

        # RSI pivot low: center is the minimum in the window
        if rsi_center == rsi_window.min():
            # Ensure it's a distinct low (not a flat plateau)
            left_min = rsi_window.iloc[:lookback].min()
            right_min = rsi_window.iloc[lookback + 1:].min()
            if rsi_center < left_min or rsi_center < right_min:
                pivot = RSIPivot(
                    index=i,
                    timestamp=str(df["timestamp"].iloc[i]),
                    price=float(price_center),
                    rsi_value=float(rsi_center),
                    pivot_type="low",
                    confirmed_at=i + lookback,  # Confirmed after lookback candles
                )
                if rsi_center <= max_threshold:  # Only consider if not extremely overbought
                    pivot_lows.append(pivot)

        # RSI pivot high: center is the maximum in the window
        if rsi_center == rsi_window.max():
            left_max = rsi_window.iloc[:lookback].max()
            right_max = rsi_window.iloc[lookback + 1:].max()
            if rsi_center > left_max or rsi_center > right_max:
                pivot = RSIPivot(
                    index=i,
                    timestamp=str(df["timestamp"].iloc[i]),
                    price=float(price_center),
                    rsi_value=float(rsi_center),
                    pivot_type="high",
                    confirmed_at=i + lookback,
                )
                if rsi_center >= min_threshold:  # Only consider if not extremely oversold
                    pivot_highs.append(pivot)

    return pivot_lows, pivot_highs


def detect_price_pivots(
    df: pd.DataFrame,
    lookback: int = 5,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Detect confirmed price swing lows and highs.

    Returns:
        (swing_lows, swing_highs): List of (index, price) tuples
    """
    if len(df) < lookback * 2 + 1:
        return [], []

    swing_lows: List[Tuple[int, float]] = []
    swing_highs: List[Tuple[int, float]] = []

    confirmable_end = len(df) - lookback - 1

    for i in range(lookback, confirmable_end + 1):
        price_window = df["low"].iloc[i - lookback:i + lookback + 1]
        high_window = df["high"].iloc[i - lookback:i + lookback + 1]

        if len(price_window) < lookback * 2 + 1:
            continue

        low_center = price_window.iloc[lookback]
        high_center = high_window.iloc[lookback]

        # Price swing low
        if low_center == price_window.min():
            left_min = price_window.iloc[:lookback].min()
            right_min = price_window.iloc[lookback + 1:].min()
            if low_center < left_min or low_center < right_min:
                swing_lows.append((i, float(low_center)))

        # Price swing high
        if high_center == high_window.max():
            left_max = high_window.iloc[:lookback].max()
            right_max = high_window.iloc[lookback + 1:].max()
            if high_center > left_max or high_center > right_max:
                swing_highs.append((i, float(high_center)))

    return swing_lows, swing_highs


# --------------------------------------------------------------------------- #
# RSI Divergence Detection
# --------------------------------------------------------------------------- #

def detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_series: pd.Series,
    config: Dict[str, Any],
) -> RSIDivergenceSignal:
    """
    Detect RSI divergence using confirmed pivots.

    Supports:
    - Regular Bullish: Price Lower Low, RSI Higher Low
    - Hidden Bullish: Price Higher Low, RSI Lower Low
    - Regular Bearish: Price Higher High, RSI Lower High
    - Hidden Bearish: Price Lower High, RSI Higher High

    CRITICAL: No lookahead bias - only uses confirmed pivots.
    """
    rsi_config = config.get("rsi", DEFAULT_IMPULSE_CONFIG["rsi"])

    if not rsi_config.get("enabled", True):
        return RSIDivergenceSignal(divergence_type=None)

    pivot_lookback = rsi_config.get("pivot_lookback", 5)
    min_thresh = rsi_config.get("min_rsi_threshold", 25)
    max_thresh = rsi_config.get("max_rsi_threshold", 75)

    # Get confirmed pivots
    rsi_lows, rsi_highs = detect_rsi_pivots(
        df, rsi_series, pivot_lookback, min_thresh, max_thresh
    )
    price_lows, price_highs = detect_price_pivots(df, pivot_lookback)

    if len(rsi_lows) < 2 or len(price_lows) < 2:
        return RSIDivergenceSignal(divergence_type=None)

    # Get the two most recent confirmed pivot lows
    last_rsi_low = rsi_lows[-1]
    prev_rsi_low = rsi_lows[-2]
    last_price_low = min(
        [(i, p) for i, p in price_lows if i <= last_rsi_low.index],
        key=lambda x: x[1],
        default=(last_rsi_low.index, last_rsi_low.price),
    )
    prev_price_candidates = [
        (i, p) for i, p in price_lows
        if i < last_price_low[0] and i >= prev_rsi_low.index
    ]
    prev_price_low = min(prev_price_candidates, key=lambda x: x[1], default=(prev_rsi_low.index, prev_rsi_low.price))

    result = RSIDivergenceSignal(detected_at_index=len(df) - 1)

    # Check for Regular Bullish Divergence
    if rsi_config.get("regular_bullish_enabled", True):
        price_lower_low = last_price_low[1] < prev_price_low[1]
        rsi_higher_low = last_rsi_low.rsi_value > prev_rsi_low.rsi_value

        if price_lower_low and rsi_higher_low:
            result.divergence_type = "regular_bullish"
            result.price_pivot_1 = RSIPivot(
                index=prev_price_low[0],
                timestamp=str(df["timestamp"].iloc[prev_price_low[0]]),
                price=prev_price_low[1],
                rsi_value=float(rsi_series.iloc[prev_price_low[0]]),
                pivot_type="low",
                confirmed_at=prev_price_low[0] + pivot_lookback,
            )
            result.price_pivot_2 = RSIPivot(
                index=last_price_low[0],
                timestamp=str(df["timestamp"].iloc[last_price_low[0]]),
                price=last_price_low[1],
                rsi_value=float(rsi_series.iloc[last_price_low[0]]),
                pivot_type="low",
                confirmed_at=last_price_low[0] + pivot_lookback,
            )
            result.rsi_pivot_1 = prev_rsi_low
            result.rsi_pivot_2 = last_rsi_low

            # Confidence based on RSI separation and oversold condition
            rsi_separation = (last_rsi_low.rsi_value - prev_rsi_low.rsi_value) / 100.0
            oversold_bonus = 0.2 if last_rsi_low.rsi_value < 40 else 0.0
            result.confidence = min(1.0, 0.5 + rsi_separation + oversold_bonus)
            result.detected_at_timestamp = str(df["timestamp"].iloc[-1])
            return result

    # Check for Hidden Bullish Divergence
    if rsi_config.get("hidden_bullish_enabled", True):
        price_higher_low = last_price_low[1] > prev_price_low[1]
        rsi_lower_low = last_rsi_low.rsi_value < prev_rsi_low.rsi_value

        if price_higher_low and rsi_lower_low:
            result.divergence_type = "hidden_bullish"
            result.price_pivot_1 = RSIPivot(
                index=prev_price_low[0],
                timestamp=str(df["timestamp"].iloc[prev_price_low[0]]),
                price=prev_price_low[1],
                rsi_value=float(rsi_series.iloc[prev_price_low[0]]),
                pivot_type="low",
                confirmed_at=prev_price_low[0] + pivot_lookback,
            )
            result.price_pivot_2 = RSIPivot(
                index=last_price_low[0],
                timestamp=str(df["timestamp"].iloc[last_price_low[0]]),
                price=last_price_low[1],
                rsi_value=float(rsi_series.iloc[last_price_low[0]]),
                pivot_type="low",
                confirmed_at=last_price_low[0] + pivot_lookback,
            )
            result.rsi_pivot_1 = prev_rsi_low
            result.rsi_pivot_2 = last_rsi_low

            # Confidence based on RSI separation
            rsi_separation = (prev_rsi_low.rsi_value - last_rsi_low.rsi_value) / 100.0
            result.confidence = min(1.0, 0.4 + rsi_separation)
            result.detected_at_timestamp = str(df["timestamp"].iloc[-1])
            return result

    # Bearish divergences (less relevant for spot-long bot but supported)
    if len(rsi_highs) >= 2 and len(price_highs) >= 2:
        last_rsi_high = rsi_highs[-1]
        prev_rsi_high = rsi_highs[-2]

        last_price_high = max(
            [(i, p) for i, p in price_highs if i <= last_rsi_high.index],
            key=lambda x: x[1],
            default=(last_rsi_high.index, last_rsi_high.price),
        )
        prev_price_candidates = [
            (i, p) for i, p in price_highs
            if i < last_price_high[0] and i >= prev_rsi_high.index
        ]
        prev_price_high = max(prev_price_candidates, key=lambda x: x[1], default=(prev_rsi_high.index, prev_rsi_high.price))

        # Regular Bearish
        if rsi_config.get("regular_bearish_enabled", False):
            price_higher_high = last_price_high[1] > prev_price_high[1]
            rsi_lower_high = last_rsi_high.rsi_value < prev_rsi_high.rsi_value

            if price_higher_high and rsi_lower_high:
                result.divergence_type = "regular_bearish"
                result.confidence = 0.6
                result.detected_at_timestamp = str(df["timestamp"].iloc[-1])
                return result

        # Hidden Bearish
        if rsi_config.get("hidden_bearish_enabled", False):
            price_lower_high = last_price_high[1] < prev_price_high[1]
            rsi_higher_high = last_rsi_high.rsi_value > prev_rsi_high.rsi_value

            if price_lower_high and rsi_higher_high:
                result.divergence_type = "hidden_bearish"
                result.confidence = 0.5
                result.detected_at_timestamp = str(df["timestamp"].iloc[-1])
                return result

    return result


# --------------------------------------------------------------------------- #
# MACD Momentum Detection
# --------------------------------------------------------------------------- #

def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD line, signal line, and histogram."""
    close = df["close"]

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def analyze_macd_momentum(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> MACDMomentumState:
    """
    Analyze MACD for momentum conditions.

    Does NOT use crossovers as primary signals (disabled by default).
    Instead focuses on:
    - Histogram contraction (bearish momentum weakening)
    - Histogram expansion (momentum acceleration)
    - Momentum recovery (histogram becoming less negative)
    """
    macd_config = config.get("macd", DEFAULT_IMPULSE_CONFIG["macd"])

    if not macd_config.get("enabled", True):
        return MACDMomentumState()

    fast = macd_config.get("fast", 12)
    slow = macd_config.get("slow", 26)
    signal_period = macd_config.get("signal", 9)

    macd_line, signal_line, histogram = calculate_macd(df, fast, slow, signal_period)

    if len(histogram) < 10:
        return MACDMomentumState()

    current_hist = float(histogram.iloc[-1])
    prev_hist = float(histogram.iloc[-2])
    prev_hist_2 = float(histogram.iloc[-3]) if len(histogram) >= 3 else prev_hist

    current_macd = float(macd_line.iloc[-1])
    current_signal = float(signal_line.iloc[-1])

    state = MACDMomentumState(
        macd_line=current_macd,
        signal_line=current_signal,
        histogram=current_hist,
    )

    # Determine histogram trend
    if current_hist < 0:
        if current_hist > prev_hist:  # Becoming less negative
            state.histogram_trend = "contracting_negative"
        elif current_hist < prev_hist:  # Becoming more negative
            state.histogram_trend = "expanding_negative"
    elif current_hist > 0:
        if current_hist > prev_hist:
            state.histogram_trend = "expanding_positive"
        elif current_hist < prev_hist:
            state.histogram_trend = "contracting_positive"

    # Determine momentum state
    momentum_state = "neutral"
    confidence = 0.0

    # Momentum Recovery: histogram was negative and is now rising
    if macd_config.get("histogram_recovery_enabled", True):
        hist_was_negative = prev_hist < 0 or prev_hist_2 < 0
        hist_rising = current_hist > prev_hist
        if hist_was_negative and hist_rising:
            momentum_state = "recovery"
            # More confidence if histogram is crossing toward zero
            if prev_hist < 0 and current_hist > prev_hist:
                confidence = 0.5 + min(0.3, abs(current_hist - prev_hist) * 10)
                if current_hist > 0:
                    confidence += 0.1

    # Histogram Expansion: positive and growing
    if macd_config.get("histogram_expansion_enabled", True):
        if current_hist > 0 and current_hist > prev_hist and prev_hist > 0:
            if momentum_state != "recovery":
                momentum_state = "acceleration"
            confidence = max(confidence, 0.4 + min(0.3, (current_hist - prev_hist) * 10))

    # Check for recent crossover (optional, less reliable)
    if macd_config.get("crossover_enabled", False):
        prev_macd = float(macd_line.iloc[-2])
        prev_sig = float(signal_line.iloc[-2])

        # Bullish crossover: MACD crosses above signal
        if prev_macd <= prev_sig and current_macd > current_signal:
            state.crossover_recent = True
            state.crossover_direction = "bullish"
            confidence = max(confidence, 0.3)

        # Bearish crossover
        elif prev_macd >= prev_sig and current_macd < current_signal:
            state.crossover_recent = True
            state.crossover_direction = "bearish"
            confidence = max(confidence, 0.3)

    state.momentum_state = momentum_state
    state.confidence = min(1.0, confidence)

    return state


# --------------------------------------------------------------------------- #
# Discovery Score Calculation
# --------------------------------------------------------------------------- #

def calculate_discovery_score(
    rsi_divergence: RSIDivergenceSignal,
    macd_state: MACDMomentumState,
    htf_trend: str = "NEUTRAL",
    liquidity_sweep_detected: bool = False,
    config: Dict[str, Any] = None,
) -> float:
    """
    Calculate an interpretable Discovery Score (0-100).

    Components:
    - RSI Regular Bullish Divergence (configurable weight)
    - RSI Hidden Bullish Divergence (configurable weight)
    - MACD Momentum Recovery (configurable weight)
    - MACD Histogram Expansion (configurable weight)
    - HTF Context Bonus (if 4H trend aligns)
    - Liquidity Sweep Bonus

    Returns:
        Score from 0-100
    """
    if config is None:
        config = DEFAULT_IMPULSE_CONFIG

    scoring = config.get("scoring", DEFAULT_IMPULSE_CONFIG["scoring"])
    weights = scoring

    score = 0.0

    # RSI Divergence Component
    if rsi_divergence and rsi_divergence.divergence_type:
        div_type = rsi_divergence.divergence_type
        div_confidence = rsi_divergence.confidence

        if div_type == "regular_bullish":
            weight = weights.get("rsi_regular_bullish_weight", 35)
            score += weight * div_confidence
        elif div_type == "hidden_bullish":
            weight = weights.get("rsi_hidden_bullish_weight", 20)
            score += weight * div_confidence
        elif div_type in ("regular_bearish", "hidden_bearish"):
            weight = weights.get("rsi_bearish_weight", 0)
            score += weight * div_confidence

    # MACD Momentum Component
    if macd_state:
        macd_confidence = macd_state.confidence

        if macd_state.momentum_state == "recovery":
            weight = weights.get("macd_recovery_weight", 25)
            score += weight * macd_confidence
        elif macd_state.momentum_state == "acceleration":
            weight = weights.get("macd_expansion_weight", 20)
            score += weight * macd_confidence
        elif macd_state.crossover_recent and macd_state.crossover_direction == "bullish":
            weight = weights.get("macd_crossover_weight", 0)
            score += weight * macd_confidence

    # HTF Context Bonus
    if htf_trend == "BULLISH":
        score += weights.get("htf_context_bonus", 10)

    # Liquidity Sweep Bonus
    if liquidity_sweep_detected:
        score += weights.get("liquidity_sweep_bonus", 10)

    return min(100.0, score)


# --------------------------------------------------------------------------- #
# Main Impulse Detector Class
# --------------------------------------------------------------------------- #

class ImpulseDetector:
    """
    Main class for impulse discovery and candidate management.

    Usage:
        detector = ImpulseDetector(config)
        candidate = detector.evaluate_symbol(symbol, df_1h, df_4h, ...)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config is not None else dict(DEFAULT_IMPULSE_CONFIG)
        self.active_candidates: List[ImpulseCandidate] = []
        self.completed_candidates: List[ImpulseCandidate] = []
        self._seen_candidate_keys: set = set()

    def evaluate_symbol(
        self,
        symbol: str,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        state_4h: Any = None,
        sweep_detected: bool = False,
        candle_idx: int = 0,
    ) -> Optional[ImpulseCandidate]:
        """
        Evaluate a symbol for impulse discovery.

        This creates an IMPULSE CANDIDATE if discovery conditions are met.
        The candidate must still pass through the Price Action engine.

        Returns:
            ImpulseCandidate if discovery threshold met, None otherwise
        """
        if not self.config.get("enabled", True):
            return None

        # Calculate RSI
        rsi_period = self.config.get("rsi", {}).get("period", 14)
        rsi_1h = ind.calculate_rsi(df_1h["close"], rsi_period)

        # Detect RSI divergence
        rsi_divergence = detect_rsi_divergence(df_1h, rsi_1h, self.config)

        # Analyze MACD momentum
        macd_state = analyze_macd_momentum(df_1h, self.config)

        # Get HTF context
        htf_trend = "NEUTRAL"
        if state_4h is not None:
            htf_trend = getattr(state_4h, "trend", "NEUTRAL")

        # Calculate discovery score
        discovery_score = calculate_discovery_score(
            rsi_divergence,
            macd_state,
            htf_trend,
            sweep_detected,
            self.config,
        )

        # Check threshold
        thresholds = self.config.get("scoring", {}).get("thresholds", {})
        candidate_threshold = thresholds.get("candidate_score", 30)

        if discovery_score < candidate_threshold:
            return None

        # Determine direction (focus on bullish for spot bot)
        direction = "BUY"
        if rsi_divergence.divergence_type in ("regular_bearish", "hidden_bearish"):
            direction = "SELL"

        # Create candidate
        ts_str = df_1h["timestamp"].iloc[-1]
        if hasattr(ts_str, "isoformat"):
            ts_str = ts_str.isoformat()

        candidate_id = f"{symbol}_{direction}_{int(pd.Timestamp(ts_str).timestamp())}_{int(discovery_score)}"

        # Prevent duplicates
        dup_window = self.config.get("candidate_management", {}).get("duplicate_window_candles", 10)
        for existing in self.active_candidates:
            if (existing.symbol == symbol and
                existing.direction == direction and
                abs(candle_idx - existing.created_candle_idx) < dup_window):
                return None

        candidate = ImpulseCandidate(
            candidate_id=candidate_id,
            symbol=symbol,
            direction=direction,
            created_at=ts_str,
            created_candle_idx=candle_idx,
            expiry_candles=self.config.get("candidate_management", {}).get("default_expiry_candles", 50),
            discovery_score=discovery_score,
            rsi_divergence=rsi_divergence if rsi_divergence.divergence_type else None,
            macd_state=macd_state if macd_state.confidence > 0 else None,
            status="SCOUTED",
            highest_price_seen=float(df_1h["close"].iloc[-1]),
            lowest_price_seen=float(df_1h["close"].iloc[-1]),
        )

        self.active_candidates.append(candidate)
        self._seen_candidate_keys.add(candidate_id)

        logger.info(
            "IMPULSE CANDIDATE: %s | Direction: %s | Score: %.1f | RSI: %s | MACD: %s",
            symbol, direction, discovery_score,
            rsi_divergence.divergence_type or "None",
            macd_state.momentum_state or "Neutral",
        )

        return candidate

    def update_candidates(
        self,
        df_1h: pd.DataFrame,
        candle_idx: int,
        structure_confirmed: bool = False,
        zone_touched: bool = False,
    ) -> None:
        """
        Update active candidates with latest price data and state changes.
        """
        current_price = float(df_1h["close"].iloc[-1])
        current_ts = df_1h["timestamp"].iloc[-1]
        if hasattr(current_ts, "isoformat"):
            current_ts = current_ts.isoformat()

        still_active: List[ImpulseCandidate] = []

        for candidate in self.active_candidates:
            candidate.highest_price_seen = max(candidate.highest_price_seen, float(df_1h["high"].iloc[-1]))
            candidate.lowest_price_seen = min(candidate.lowest_price_seen, float(df_1h["low"].iloc[-1]))

            elapsed = candle_idx - candidate.created_candle_idx

            # Check expiry
            if elapsed > candidate.expiry_candles:
                candidate.status = "EXPIRED"
                self.completed_candidates.append(candidate)
                continue

            # Update status based on progression
            if candidate.status == "SCOUTED" and structure_confirmed:
                candidate.status = "STRUCTURE_CONFIRMED"
                candidate.structure_confirmed_at = current_ts
            elif candidate.status == "STRUCTURE_FORMING" and structure_confirmed:
                candidate.status = "STRUCTURE_CONFIRMED"
                candidate.structure_confirmed_at = current_ts
            elif candidate.status == "STRUCTURE_CONFIRMED" and zone_touched:
                candidate.status = "ZONE_TOUCHED"
                candidate.zone_touched_at = current_ts

            still_active.append(candidate)

        self.active_candidates = still_active

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate candidate statistics."""
        all_candidates = self.completed_candidates + self.active_candidates
        total = len(all_candidates)

        if total == 0:
            return {
                "total_candidates": 0,
                "structure_confirmed_count": 0,
                "zone_touched_count": 0,
                "entered_count": 0,
                "invalidated_count": 0,
                "expired_count": 0,
                "avg_discovery_score": 0.0,
            }

        structure_confirmed = [c for c in all_candidates if c.status == "STRUCTURE_CONFIRMED"]
        zone_touched = [c for c in all_candidates if c.status == "ZONE_TOUCHED"]
        entered = [c for c in all_candidates if c.status == "ENTERED"]
        invalidated = [c for c in all_candidates if c.status == "INVALIDATED"]
        expired = [c for c in all_candidates if c.status == "EXPIRED"]

        scores = [c.discovery_score for c in all_candidates]

        return {
            "total_candidates": total,
            "structure_confirmed_count": len(structure_confirmed),
            "zone_touched_count": len(zone_touched),
            "entered_count": len(entered),
            "invalidated_count": len(invalidated),
            "expired_count": len(expired),
            "avg_discovery_score": round(sum(scores) / len(scores), 2),
        }


# --------------------------------------------------------------------------- #
# Integration Helper Functions
# --------------------------------------------------------------------------- #

def get_impulse_discovery_label(candidate: Optional[ImpulseCandidate]) -> str:
    """Get a human-readable label for impulse discovery state."""
    if candidate is None:
        return "NONE"

    status = candidate.status
    rsi_info = ""
    if candidate.rsi_divergence:
        rsi_info = f"RSI {candidate.rsi_divergence.divergence_type}"

    macd_info = ""
    if candidate.macd_state:
        macd_info = f"MACD {candidate.macd_state.momentum_state}"

    parts = [p for p in [rsi_info, macd_info] if p]
    indicators = ", ".join(parts) if parts else "Neutral"

    return f"{status}: {indicators} (Score: {candidate.discovery_score:.0f})"


def enrich_signal_with_discovery(
    signal: Dict[str, Any],
    candidate: Optional[ImpulseCandidate],
) -> Dict[str, Any]:
    """
    Enrich a strategy signal card with impulse discovery information.

    This adds discovery context WITHOUT bypassing price action logic.
    """
    if signal is None:
        return {}

    if candidate is None:
        signal["impulse_discovery"] = {
            "enabled": True,
            "status": "NONE",
            "discovery_score": 0,
            "rsi_divergence": None,
            "macd_state": None,
            "note": "No impulse candidate - setup identified through pure price action",
        }
        return signal

    signal["impulse_discovery"] = {
        "enabled": True,
        "status": candidate.status,
        "discovery_score": round(candidate.discovery_score, 1),
        "rsi_divergence": candidate.rsi_divergence.to_dict() if candidate.rsi_divergence else None,
        "macd_state": candidate.macd_state.to_dict() if candidate.macd_state else None,
        "candidate_id": candidate.candidate_id,
        "created_at": candidate.created_at,
    }

    return signal
