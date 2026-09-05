"""
liquidity.py — liquidity pool and sweep/breakout engine (spec section 5).

CORE RULE: a sweep is NOT an automatic reversal signal. This module only
detects and measures what happened; strategy.py decides what it means.

Definitions
-----------
Liquidity pool  : a prior swing high/low, or a cluster of highs/lows sitting
                   within `equal_tolerance_pct` of each other (equal highs /
                   equal lows — classic resting-liquidity zones).
Liquidity sweep : price trades beyond a pool level (the wick clears it) and
                   then closes back on the origin side within
                   `sweep_confirm_candles` candles. Depth and rejection
                   strength are both measured so downstream scoring can
                   tell a decisive sweep from a marginal one.
Breakout + acceptance : price closes beyond the pool level and stays beyond
                   it for `sweep_confirm_candles` candles (no reclaim) —
                   classified separately from a sweep, never conflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

import structure as struct_engine

LiquidityType = Literal["swing_high", "swing_low", "equal_highs", "equal_lows"]


@dataclass
class LiquidityPool:
    liquidity_type: LiquidityType
    level: float
    touches: int
    freshness_index: int          # how many candles ago this pool last formed/was touched
    confidence: float


@dataclass
class SweepResult:
    liquidity_type: str
    liquidity_level: float
    sweep_direction: Optional[str]     # "up" (swept highs) | "down" (swept lows) | None
    sweep_depth_pct: float
    rejection_strength: float          # 0-1, how much of the wick was reclaimed by close
    accepted_beyond_level: bool        # True => breakout+acceptance, not a sweep
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def detect_liquidity_pools(
    df: pd.DataFrame,
    state: "struct_engine.StructureState",
    equal_tolerance_pct: float = 0.15,
    lookback_swings: int = 6,
) -> List[LiquidityPool]:
    """Builds pools from recent swing highs/lows plus equal-high/equal-low clusters."""
    n = len(df)
    pools: List[LiquidityPool] = []

    for s in state.swing_highs[-lookback_swings:]:
        pools.append(LiquidityPool("swing_high", s.price, touches=1, freshness_index=n - 1 - s.index, confidence=0.55))
    for s in state.swing_lows[-lookback_swings:]:
        pools.append(LiquidityPool("swing_low", s.price, touches=1, freshness_index=n - 1 - s.index, confidence=0.55))

    def _cluster(points, kind: LiquidityType):
        prices = sorted((s.price, s.index) for s in points)
        used = set()
        for i, (p1, idx1) in enumerate(prices):
            if idx1 in used:
                continue
            cluster = [(p1, idx1)]
            for p2, idx2 in prices[i + 1 :]:
                if abs(p2 - p1) / p1 * 100 <= equal_tolerance_pct:
                    cluster.append((p2, idx2))
            if len(cluster) >= 2:
                for _, idx in cluster:
                    used.add(idx)
                avg_price = sum(p for p, _ in cluster) / len(cluster)
                latest_idx = max(idx for _, idx in cluster)
                pools.append(LiquidityPool(kind, avg_price, touches=len(cluster),
                                            freshness_index=n - 1 - latest_idx, confidence=min(0.95, 0.6 + 0.1 * len(cluster))))

    _cluster(state.swing_highs[-lookback_swings:], "equal_highs")
    _cluster(state.swing_lows[-lookback_swings:], "equal_lows")

    pools.sort(key=lambda p: p.freshness_index)
    return pools


def detect_sweep(
    df: pd.DataFrame,
    pool: LiquidityPool,
    sweep_confirm_candles: int = 3,
    min_wick_clear_pct: float = 0.05,
) -> Optional[SweepResult]:
    """
    Looks at the most recent `sweep_confirm_candles` (+1 trigger candle)
    to see whether price traded through `pool.level` and how it reacted.
    Returns None if the level was never actually touched recently.
    """
    n = len(df)
    window = df.iloc[-(sweep_confirm_candles + 3) :]
    is_high_pool = pool.liquidity_type in ("swing_high", "equal_highs")
    level = pool.level

    trigger_i = None
    for i in range(len(window)):
        row = window.iloc[i]
        if is_high_pool and row["high"] > level:
            trigger_i = i
        elif not is_high_pool and row["low"] < level:
            trigger_i = i

    if trigger_i is None:
        return None

    trigger_row = window.iloc[trigger_i]
    after = window.iloc[trigger_i:]
    last_close = float(after["close"].iloc[-1])

    if is_high_pool:
        wick_extreme = float(after["high"].max())
        depth_pct = (wick_extreme - level) / level * 100
        reclaimed = last_close < level
        accepted = last_close > level and not reclaimed
        rejection_strength = max(0.0, min(1.0, (wick_extreme - last_close) / max(wick_extreme - level, 1e-9))) if reclaimed else 0.0
        direction = "up"
    else:
        wick_extreme = float(after["low"].min())
        depth_pct = (level - wick_extreme) / level * 100
        reclaimed = last_close > level
        accepted = last_close < level and not reclaimed
        rejection_strength = max(0.0, min(1.0, (last_close - wick_extreme) / max(level - wick_extreme, 1e-9))) if reclaimed else 0.0
        direction = "down"

    if depth_pct < min_wick_clear_pct and not accepted:
        return None

    confidence = pool.confidence * (0.5 + 0.5 * rejection_strength) if reclaimed else pool.confidence * 0.7

    return SweepResult(
        liquidity_type=pool.liquidity_type,
        liquidity_level=level,
        sweep_direction=direction if reclaimed else None,
        sweep_depth_pct=round(depth_pct, 3),
        rejection_strength=round(rejection_strength, 2),
        accepted_beyond_level=accepted,
        confidence=round(min(confidence, 0.95), 2),
    )


def best_recent_sweep(
    df: pd.DataFrame,
    pools: List[LiquidityPool],
    sweep_confirm_candles: int = 3,
) -> Optional[SweepResult]:
    """Scans all pools and returns the highest-confidence genuine sweep (reclaimed, not accepted)."""
    candidates = []
    for pool in pools:
        result = detect_sweep(df, pool, sweep_confirm_candles)
        if result and result.sweep_direction is not None:
            candidates.append(result)
    if not candidates:
        return None
    candidates.sort(key=lambda r: r.confidence, reverse=True)
    return candidates[0]
