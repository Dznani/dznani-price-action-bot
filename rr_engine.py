"""
rr_engine.py — R:R and available-room engine (spec section 13).

This is a HARD trade-quality filter: a technically great setup with
insufficient room to the nearest opposing structure is NO TRADE, full stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RRResult:
    risk_distance: float
    reward_distance: float
    rr: float
    nearest_obstacle: Optional[float]
    available_room_pct: float
    passes_minimum: bool

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def nearest_obstacle_for_long(current_price: float, resistance_levels: List[float]) -> Optional[float]:
    above = [lvl for lvl in resistance_levels if lvl > current_price]
    return min(above) if above else None


def nearest_obstacle_for_context(current_price: float, support_levels: List[float]) -> Optional[float]:
    below = [lvl for lvl in support_levels if lvl < current_price]
    return max(below) if below else None


def evaluate_rr(
    entry: float,
    stop_loss: float,
    direction: str,
    resistance_levels: Optional[List[float]] = None,
    support_levels: Optional[List[float]] = None,
    minimum_rr: float = 2.0,
    fallback_reward_pct: float = 0.10,
) -> RRResult:
    """
    reward_distance is measured to the nearest opposing structural level
    (resistance for a long, support for bearish/dip-buy context) rather
    than a fixed target — an A+ setup with a wall 1.8% away and a 4%
    structural stop is a bad trade regardless of how clean the entry looks.
    Falls back to fallback_reward_pct of price if no obstacle is known yet
    (e.g. all-time-high breakout with nothing overhead).
    """
    risk_distance = abs(entry - stop_loss)
    resistance_levels = resistance_levels or []
    support_levels = support_levels or []

    if direction == "BUY":
        obstacle = nearest_obstacle_for_long(entry, resistance_levels)
    else:
        obstacle = nearest_obstacle_for_context(entry, support_levels)

    if obstacle is None:
        reward_distance = entry * fallback_reward_pct
    else:
        reward_distance = abs(obstacle - entry)

    rr = round(reward_distance / risk_distance, 3) if risk_distance > 0 else 0.0
    available_room_pct = round(reward_distance / entry * 100, 3)

    return RRResult(
        risk_distance=round(risk_distance, 8),
        reward_distance=round(reward_distance, 8),
        rr=rr,
        nearest_obstacle=obstacle,
        available_room_pct=available_room_pct,
        passes_minimum=rr >= minimum_rr,
    )
