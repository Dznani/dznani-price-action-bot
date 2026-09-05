"""
database.py — JSON persistence layer for Dznani Signals Bot.

Two files:
  dznani_signals.json  -> {"signals": [...], "trades": [...], "settings": {...}, "daily_stats": {...}}
  watchlist.json        -> ["BTC/USDT", "ETH/USDT", ...]

Writes are atomic (write to a temp file, then os.replace) so a crash or
power loss mid-write can never corrupt the store. A rolling backup of
the previous version is kept alongside each file.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dznani.database")

DEFAULT_SETTINGS = {
    # AUDIT NOTE (spec Part 22): these three keys belonged to the old
    # single-timeframe 5-rule indicator engine. That engine no longer
    # exists anywhere in the codebase — strategy.evaluate_symbol() never
    # reads them. Kept ONLY so a pre-existing dznani_signals.json settings
    # file from before this rebuild still loads without a KeyError; do not
    # add new logic that reads these. telegram_bot.py's /set command warns
    # the user if they try to change one of these (see _LEGACY_INERT_SETTINGS).
    "min_confirmations": 4,
    "trend_filter_enabled": True,
    "pro_filters_enabled": True,
    "capital": 25000.0,
    "daily_loss_limit_pct": 0.05,
    "scan_interval_minutes": 10,
    "duplicate_signal_hours": 6,
    "paused": False,
    "min_liquidity_usd": 500000.0,  # skip symbols with < $500k 24h quote volume — avoids thin/unreliable setups

    # --- Price Action Edition settings (spec section 27) ---
    "risk_per_trade_usd": 250.0,
    "aggressive_entry_pct": 0.30,
    "confirmation_add_pct": 0.70,
    "tp1_pct": 0.05, "tp1_sell_pct": 0.40,
    "tp2_pct": 0.10, "tp2_sell_pct": 0.40,
    "tp3_pct": 0.15, "tp3_sell_pct": 0.20,
    "minimum_rr": 2.0,
    "swing_left": 2,
    "swing_right": 2,
    "bos_min_displacement_pct": 0.15,
    "min_swing_significance_pct": 0.3,
    # --- Entry Extension / Chase Filter (Layer 5) ---
    "extended_chase_score": 40.0,       # chase_score >= this -> EXTENDED (wait for pullback)
    "overextended_chase_score": 70.0,   # chase_score >= this -> OVEREXTENDED (no chase at all)
    "displacement_full_scale_pct": 8.0, # % move since break considered "fully extended" on that axis
    "atr_full_scale_multiple": 3.0,     # move measured in ATRs considered "fully extended" on that axis
    "choch_min_displacement_pct": 0.15,
    "liquidity_equal_tolerance_pct": 0.15,
    "sweep_confirm_candles": 3,
    "retest_tolerance_pct": 0.3,
    "min_setup_score": 50,
    "sl_atr_buffer_mult": 0.5,           # ATR buffer added to structural invalidation level for SL
    "max_watch_candles": 36,             # Max 1H candles to keep a pending watch active before expiry
    "indicator_weighting": None,   # None -> strategy.DEFAULT_WEIGHTS
    "send_no_trade_cards": False,  # True sends the NO TRADE card for every scanned symbol, not just VALID setups
}


def _atomic_write_json(path: str, data: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    if os.path.exists(path):
        shutil.copy2(path, f"{path}.bak")
    os.replace(tmp_path, path)


class Database:
    def __init__(self, signals_path: str = "dznani_signals.json", watchlist_path: str = "watchlist.json"):
        self.signals_path = signals_path
        self.watchlist_path = watchlist_path
        self._lock = threading.RLock()
        self._data = self._load_signals_file()
        self._watchlist = self._load_watchlist_file()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_signals_file(self) -> Dict[str, Any]:
        if os.path.exists(self.signals_path):
            try:
                with open(self.signals_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data.setdefault("signals", [])
                data.setdefault("trades", [])
                data.setdefault("settings", dict(DEFAULT_SETTINGS))
                data.setdefault("daily_stats", {})
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Failed to load %s (%s) — trying backup", self.signals_path, e)
                backup = f"{self.signals_path}.bak"
                if os.path.exists(backup):
                    with open(backup, "r", encoding="utf-8") as f:
                        return json.load(f)
        return {"signals": [], "trades": [], "settings": dict(DEFAULT_SETTINGS), "daily_stats": {}}

    def _load_watchlist_file(self) -> List[str]:
        if os.path.exists(self.watchlist_path):
            try:
                with open(self.watchlist_path, "r", encoding="utf-8") as f:
                    wl = json.load(f)
                if isinstance(wl, list):
                    return wl
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Failed to load %s: %s", self.watchlist_path, e)
        return []

    # ------------------------------------------------------------------ #
    # Persisting
    # ------------------------------------------------------------------ #
    def _save_signals_file(self) -> None:
        with self._lock:
            _atomic_write_json(self.signals_path, self._data)

    def _save_watchlist_file(self) -> None:
        with self._lock:
            tmp_path = f"{self.watchlist_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._watchlist, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.watchlist_path):
                shutil.copy2(self.watchlist_path, f"{self.watchlist_path}.bak")
            os.replace(tmp_path, self.watchlist_path)

    # ------------------------------------------------------------------ #
    # Settings
    # ------------------------------------------------------------------ #
    def get_settings(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data["settings"])

    def set_setting(self, key: str, value: Any) -> None:
        with self._lock:
            self._data["settings"][key] = value
            self._save_signals_file()

    # ------------------------------------------------------------------ #
    # Signals
    # ------------------------------------------------------------------ #
    def add_signal(self, signal: Dict[str, Any]) -> None:
        with self._lock:
            self._data["signals"].append(signal)
            self._save_signals_file()

    def last_signal_time(self, symbol: str) -> Optional[datetime]:
        """Most recent signal timestamp for a symbol, or None if never signaled."""
        with self._lock:
            matches = [s for s in self._data["signals"] if s.get("symbol") == symbol]
            if not matches:
                return None
            latest = max(matches, key=lambda s: s["timestamp"])
            ts = datetime.fromisoformat(latest["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts

    def get_recent_signals(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return list(reversed(self._data["signals"][-limit:]))

    # ------------------------------------------------------------------ #
    # Trades (funded account)
    # ------------------------------------------------------------------ #
    def open_trade(self, symbol: str, entry_price: float, size_usd: float, direction: str = "BUY") -> Dict[str, Any]:
        with self._lock:
            trade = {
                "id": len(self._data["trades"]) + 1,
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": None,
                "size": size_usd,
                "pnl": None,
                "status": "open",
                "open_date": datetime.now(timezone.utc).isoformat(),
                "close_date": None,
            }
            self._data["trades"].append(trade)
            self._save_signals_file()
            return trade

    def close_trade(self, symbol: str, exit_price: float, pnl_fn) -> Optional[Dict[str, Any]]:
        """
        Closes the oldest open trade for `symbol`. `pnl_fn(entry, exit, size, direction)`
        is injected (see risk.calculate_pnl) to keep this module free of risk-logic imports.
        """
        with self._lock:
            for trade in self._data["trades"]:
                if trade["symbol"] == symbol and trade["status"] == "open":
                    trade["exit_price"] = exit_price
                    trade["pnl"] = pnl_fn(trade["entry_price"], exit_price, trade["size"], trade["direction"])
                    trade["status"] = "closed"
                    trade["close_date"] = datetime.now(timezone.utc).isoformat()
                    self._save_signals_file()
                    self._update_daily_stats(trade["pnl"])
                    return trade
            return None

    def get_open_trades(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [t for t in self._data["trades"] if t["status"] == "open"]

    def get_all_trades(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._data["trades"])

    # ------------------------------------------------------------------ #
    # Daily stats / loss limit
    # ------------------------------------------------------------------ #
    def _today_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _update_daily_stats(self, pnl: float) -> None:
        with self._lock:
            key = self._today_key()
            stats = self._data["daily_stats"].setdefault(key, {"date": key, "total_pnl": 0.0, "trade_count": 0})
            stats["total_pnl"] = round(stats["total_pnl"] + pnl, 2)
            stats["trade_count"] += 1
            self._save_signals_file()

    def get_today_pnl(self) -> float:
        with self._lock:
            return self._data["daily_stats"].get(self._today_key(), {}).get("total_pnl", 0.0)

    def get_daily_stats(self, date_key: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            key = date_key or self._today_key()
            return dict(self._data["daily_stats"].get(key, {"date": key, "total_pnl": 0.0, "trade_count": 0}))

    # ------------------------------------------------------------------ #
    # Watchlist
    # ------------------------------------------------------------------ #
    def get_watchlist(self) -> List[str]:
        with self._lock:
            return list(self._watchlist)

    def add_to_watchlist(self, symbol: str) -> bool:
        with self._lock:
            if symbol in self._watchlist:
                return False
            self._watchlist.append(symbol)
            self._save_watchlist_file()
            return True

    def remove_from_watchlist(self, symbol: str) -> bool:
        with self._lock:
            if symbol not in self._watchlist:
                return False
            self._watchlist.remove(symbol)
            self._save_watchlist_file()
            return True
