"""
test_watch_engine.py — Comprehensive unit tests for the Persistent Watch Engine
and setup lifecycle state machine.
"""

import os
import sys
from datetime import datetime, timezone
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watch_engine import WatchManager, WatchSetup


def _make_dummy_1h_df(closes: list, base_ts: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    timestamps = pd.date_range(start=base_ts, periods=len(closes), freq="1h", tz="UTC")
    rows = []
    for i, c in enumerate(closes):
        h = c * 1.005
        l = c * 0.995
        o = closes[i - 1] if i > 0 else c
        rows.append({
            "timestamp": timestamps[i],
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 1000.0,
        })
    return pd.DataFrame(rows)


def _make_dummy_4h_df(closes: list, base_ts: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    timestamps = pd.date_range(start=base_ts, periods=len(closes), freq="4h", tz="UTC")
    rows = []
    for i, c in enumerate(closes):
        h = c * 1.01
        l = c * 0.99
        o = closes[i - 1] if i > 0 else c
        rows.append({
            "timestamp": timestamps[i],
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 4000.0,
        })
    return pd.DataFrame(rows)


def test_watch_creation_from_no_chase_and_wait_signals():
    wm = WatchManager(default_expiry_candles=24)

    mock_signal = {
        "symbol": "BTC/USDT",
        "direction": "BUY",
        "decision": "WAIT",
        "signal_type": "NO CHASE",
        "current_price": 50000.0,
        "bos": {"level": 48000.0, "break_timestamp": "2026-01-01T10:00:00Z", "direction": "bullish"},
        "structural_invalidation": 46000.0,
        "stop_loss": 45500.0,
        "extension": {
            "label": "OVEREXTENDED",
            "chase_score": 75.0,
            "preferred_entry_low": 47800.0,
            "preferred_entry_high": 48200.0,
        },
        "location": {
            "premium_discount": {"zone": "premium"},
            "assessment": {"location_score": 60.0},
        },
        "setup_score": 80.0,
        "setup_grade": "A",
    }

    watch = wm.create_watch_from_signal(mock_signal, candle_idx=10, timestamp="2026-01-01T10:00:00Z")
    assert watch is not None
    assert watch.symbol == "BTC/USDT"
    assert watch.direction == "BUY"
    assert watch.status == "WATCHING"
    assert watch.preferred_zone_low == 47800.0
    assert watch.preferred_zone_high == 48200.0
    assert watch.protected_level == 46000.0
    assert len(wm.active_watches) == 1


def test_duplicate_watch_prevention():
    wm = WatchManager()
    mock_signal = {
        "symbol": "ETH/USDT",
        "direction": "BUY",
        "decision": "WAIT",
        "signal_type": "WAIT FOR RETEST",
        "current_price": 3000.0,
        "bos": {"level": 2900.0, "break_timestamp": "2026-01-01T12:00:00Z"},
        "structural_invalidation": 2800.0,
        "extension": {"preferred_entry_low": 2880.0, "preferred_entry_high": 2920.0},
    }

    w1 = wm.create_watch_from_signal(mock_signal, candle_idx=5, timestamp="2026-01-01T12:00:00Z")
    w2 = wm.create_watch_from_signal(mock_signal, candle_idx=6, timestamp="2026-01-01T13:00:00Z")

    assert w1 is not None
    assert w2 is None
    assert len(wm.active_watches) == 1


def test_watch_invalidation_when_protected_level_breached():
    wm = WatchManager(default_expiry_candles=24)
    watch = WatchSetup(
        setup_id="test_invalidation",
        symbol="BTC/USDT",
        direction="BUY",
        created_at="2026-01-01T00:00:00Z",
        created_candle_idx=0,
        protected_level=46000.0,
        invalidation_level=46000.0,
        preferred_zone_low=47500.0,
        preferred_zone_high=48500.0,
        break_level=48000.0,
    )
    wm.active_watches.append(watch)

    # Candle price dumps below 46000
    df_1h = _make_dummy_1h_df([49000, 48000, 45500])
    df_4h = _make_dummy_4h_df([49000, 48000, 45500])

    signals = wm.evaluate_active_watches(df_1h, df_4h, settings={})
    assert len(signals) == 0
    assert len(wm.active_watches) == 0
    assert len(wm.completed_watches) == 1
    assert wm.completed_watches[0].status == "INVALIDATED"
    assert "breached" in wm.completed_watches[0].rejection_reason


def test_watch_expiry_when_max_candles_exceeded():
    wm = WatchManager(default_expiry_candles=5)
    watch = WatchSetup(
        setup_id="test_expiry",
        symbol="SOL/USDT",
        direction="BUY",
        created_at="2026-01-01T00:00:00Z",
        created_candle_idx=0,
        expiry_candles=5,
        protected_level=100.0,
        invalidation_level=100.0,
        preferred_zone_low=110.0,
        preferred_zone_high=115.0,
        break_level=112.0,
    )
    wm.active_watches.append(watch)

    # Price stays above zone (120-130) for 8 candles
    df_1h = _make_dummy_1h_df([125, 126, 127, 128, 129, 130, 131, 132])
    df_4h = _make_dummy_4h_df([125, 126, 127, 128, 129, 130, 131, 132])

    signals = wm.evaluate_active_watches(df_1h, df_4h, settings={})
    assert len(signals) == 0
    assert len(wm.active_watches) == 0
    assert len(wm.completed_watches) == 1
    assert wm.completed_watches[0].status == "EXPIRED"


def test_pullback_to_zone_and_confirmation_triggers_entered_signal():
    wm = WatchManager(default_expiry_candles=24)
    watch = WatchSetup(
        setup_id="test_pullback_confirm",
        symbol="BTC/USDT",
        direction="BUY",
        created_at="2026-01-01T00:00:00Z",
        created_candle_idx=80,
        protected_level=46000.0,
        invalidation_level=46000.0,
        preferred_zone_low=47800.0,
        preferred_zone_high=48400.0,
        break_level=48000.0,
        setup_score=85.0,
        setup_grade="A",
    )
    wm.active_watches.append(watch)

    # 1. Price pulls back into zone (48100) and prints a bullish rejection candle
    # Create 100 candles of realistic history so 200/indicators/swings don't lack context
    warmup_prices = [45000 + i * 50 for i in range(80)]
    # Next: breakout to 54000, then pullback to 48100 with a green close
    test_prices = warmup_prices + [54000, 52500, 50800, 48100]
    df_1h = _make_dummy_1h_df(test_prices)
    # Set the last candle explicitly with a bullish rejection inside zone [47800, 48400]
    df_1h.loc[len(df_1h) - 1, "open"] = 47900.0
    df_1h.loc[len(df_1h) - 1, "low"] = 47850.0  # inside zone
    df_1h.loc[len(df_1h) - 1, "high"] = 48300.0
    df_1h.loc[len(df_1h) - 1, "close"] = 48250.0  # strong bullish close

    df_4h = _make_dummy_4h_df([45000 + i * 150 for i in range(50)])

    settings = {
        "minimum_rr": 1.0,
        "sl_atr_buffer_mult": 0.5,
        "max_structural_risk_pct": 8.0,
        "risk_per_trade_usd": 250.0,
        "capital": 25000.0,
    }

    signals = wm.evaluate_active_watches(df_1h, df_4h, settings=settings)
    assert len(signals) == 1
    sig = signals[0]
    assert sig["decision"] == "VALID"
    assert sig["entry_model"] == "B"
    assert sig["signal_type"] == "CONFIRMATION LONG"
    assert sig["entry"] == 48250.0
    assert sig["stop_loss"] < 46000.0  # protected_level - atr buffer
    assert len(wm.completed_watches) == 1
    assert wm.completed_watches[0].status == "ENTERED"


def test_watch_statistics_calculation():
    wm = WatchManager()
    w1 = WatchSetup("w1", "BTC/USDT", "BUY", "2026-01-01T00:00:00Z", 0, status="ENTERED", touch_candle_idx=3)
    w2 = WatchSetup("w2", "ETH/USDT", "BUY", "2026-01-01T00:00:00Z", 0, status="INVALIDATED", touch_candle_idx=2)
    w3 = WatchSetup("w3", "SOL/USDT", "BUY", "2026-01-01T00:00:00Z", 0, status="EXPIRED", touch_candle_idx=None)
    w4 = WatchSetup("w4", "BNB/USDT", "BUY", "2026-01-01T00:00:00Z", 0, status="WATCHING", touch_candle_idx=None)

    wm.completed_watches = [w1, w2, w3]
    wm.active_watches = [w4]

    stats = wm.get_statistics()
    assert stats["total_watches"] == 4
    assert stats["zone_touched_count"] == 2
    assert stats["trades_entered_count"] == 1
    assert stats["invalidated_count"] == 1
    assert stats["expired_count"] == 1
    assert stats["watch_to_zone_conversion_pct"] == 50.0
    assert stats["watch_to_trade_conversion_pct"] == 25.0
    assert stats["zone_to_trade_conversion_pct"] == 50.0
