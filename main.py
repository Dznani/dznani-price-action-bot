"""
main.py — entry point for Dznani Signals Bot.

Runs the Telegram bot (command handlers) and the background scan loop
concurrently under one asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Optional

import yaml

import risk
import strategy
from database import Database
from exchange import BinanceExchange
from notion_exporter import NotionExporter
from telegram_bot import DznaniTelegramBot
from utils import setup_logging
from watch_engine import WatchManager

CONFIG_PATH = os.environ.get("DZNANI_CONFIG", "config.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found at '{path}'. Copy config.yaml.example to config.yaml and fill in your keys."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def scan_loop(
    db: Database,
    exchange: BinanceExchange,
    bot: DznaniTelegramBot,
    config: dict,
    logger: logging.Logger,
    notion: Optional[NotionExporter] = None,
) -> None:
    """Runs forever: every scan_interval_minutes, evaluate the watchlist
    (or the top-200 USDT pairs if the watchlist is empty) and push any
    signals that clear the confluence threshold and duplicate-signal window."""
    while True:
        settings = db.get_settings()
        interval_minutes = int(settings.get("scan_interval_minutes", config.get("scan_interval_minutes", 10)))

        if settings.get("paused"):
            logger.info("Bot is paused — skipping this scan cycle.")
            await asyncio.sleep(interval_minutes * 60)
            continue

        if not exchange.is_available():
            logger.error("Binance unreachable — skipping this scan cycle.")
            await asyncio.sleep(interval_minutes * 60)
            continue

        today_pnl = db.get_today_pnl()
        capital = settings.get("capital", risk.DEFAULT_CAPITAL)
        loss_limit_pct = settings.get("daily_loss_limit_pct", risk.DEFAULT_DAILY_LOSS_LIMIT_PCT)
        if risk.daily_loss_limit_breached(today_pnl, capital, loss_limit_pct):
            logger.warning("Daily loss limit breached (P&L $%.2f) — no new trades this cycle.", today_pnl)
            await asyncio.sleep(interval_minutes * 60)
            continue

        symbols = db.get_watchlist()
        if not symbols:
            fallback_limit = int(config.get("watchlist_fallback_limit", 200))
            logger.info("Watchlist empty — falling back to top %d Binance USDT pairs.", fallback_limit)
            try:
                symbols = exchange.fetch_top_usdt_pairs(limit=fallback_limit)
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to fetch top USDT pairs: %s", e)
                await asyncio.sleep(interval_minutes * 60)
                continue

        logger.info("Scanning %d symbols…", len(symbols))
        signal_count = 0
        ohlcv_limit_1h = int(config.get("ohlcv_limit_1h", 300))
        ohlcv_limit_4h = int(config.get("ohlcv_limit_4h", 150))
        duplicate_hours = int(settings.get("duplicate_signal_hours", 6))
        send_no_trade_cards = bool(settings.get("send_no_trade_cards", False))

        # One bulk ticker fetch per cycle (not per symbol) to build the
        # liquidity map used by the min_liquidity_usd filter in strategy.py.
        volume_map: dict = {}
        try:
            tickers = exchange.fetch_all_tickers()
            volume_map = {sym: (t.get("quoteVolume") or 0.0) for sym, t in tickers.items()}
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not fetch bulk tickers for liquidity filter this cycle: %s", e)

        watch_mgr = getattr(scan_loop, "watch_mgr", None)
        if watch_mgr is None:
            watch_mgr = WatchManager(default_expiry_candles=int(settings.get("max_watch_candles", 36)))
            scan_loop.watch_mgr = watch_mgr

        for symbol in symbols:
            try:
                df_1h = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=ohlcv_limit_1h)
                df_4h = exchange.fetch_ohlcv(symbol, timeframe="4h", limit=ohlcv_limit_4h)
                quote_volume = volume_map.get(symbol)

                # 1. Evaluate any existing active watches for this symbol
                confirmed_watch_signals = watch_mgr.evaluate_active_watches(df_1h, df_4h, settings)
                for confirmed_sig in confirmed_watch_signals:
                    db.add_signal(confirmed_sig)
                    if notion is not None:
                        notion.export_signal(confirmed_sig)
                    await bot.send_signal_card(confirmed_sig)
                    signal_count += 1
                    logger.info("Confirmed Watch Pullback Signal sent: %s %s (%s)", symbol, confirmed_sig["signal_type"], confirmed_sig["decision"])

                # 2. Evaluate current candle for new structural setups
                signal = strategy.evaluate_symbol(symbol, df_4h, df_1h, settings, quote_volume_24h=quote_volume)
                if not signal:
                    continue  # liquidity/data-length guard rejected it outright — nothing to say

                # If setup requires waiting for pullback/retest, register persistent watch
                if signal.get("decision") == "WAIT" and signal.get("direction") == "BUY":
                    watch_mgr.create_watch_from_signal(
                        signal=signal,
                        candle_idx=len(df_1h) - 1,
                        timestamp=df_1h["timestamp"].iloc[-1],
                        expiry_candles=int(settings.get("max_watch_candles", 36)),
                    )

                is_tradeable = signal["decision"] == "VALID"
                if not is_tradeable and not send_no_trade_cards:
                    if notion is not None:
                        notion.export_signal(signal)
                    continue

                last_time = db.last_signal_time(symbol)
                if not strategy.should_send_signal(symbol, last_time, duplicate_hours):
                    logger.debug("Skipping duplicate signal for %s (< %dh since last)", symbol, duplicate_hours)
                    continue

                db.add_signal(signal)
                if notion is not None:
                    notion.export_signal(signal)
                await bot.send_signal_card(signal)
                signal_count += 1
                logger.info("Signal sent: %s %s %s (%s)", symbol, signal.get("direction"), signal["signal_type"], signal["decision"])

            except Exception as e:  # noqa: BLE001
                logger.warning("Scan failed for %s: %s", symbol, e)

        logger.info("Scan cycle complete — %d signal(s) sent. Active watches: %d. Sleeping %d minutes.",
                    signal_count, len(watch_mgr.active_watches), interval_minutes)
        await asyncio.sleep(interval_minutes * 60)


async def run() -> None:
    config = load_config()
    logger = setup_logging(config.get("log_dir", "logs"), config.get("log_level", "INFO"))
    logger.info("Starting Dznani Signals Bot (Binance-only, Frankfurt VPS build)…")

    db = Database(
        signals_path=config.get("signals_path", "dznani_signals.json"),
        watchlist_path=config.get("watchlist_path", "watchlist.json"),
    )

    exchange = BinanceExchange(
        api_key=config.get("binance", {}).get("api_key"),
        api_secret=config.get("binance", {}).get("api_secret"),
    )

    telegram_cfg = config.get("telegram", {})
    bot = DznaniTelegramBot(
        token=telegram_cfg["bot_token"],
        chat_id=str(telegram_cfg["chat_id"]),
        db=db,
        exchange=exchange,
    )
    application = bot.build()

    notion_cfg = config.get("notion", {}) or {}
    notion = NotionExporter(
        api_key=notion_cfg.get("api_key"),
        database_id=notion_cfg.get("database_id"),
        enabled=bool(notion_cfg.get("enabled", False)),
    )

    async with application:
        await application.start()
        await application.updater.start_polling()
        logger.info("Telegram bot polling started.")
        try:
            await scan_loop(db, exchange, bot, config, logger, notion=notion)
        finally:
            await application.updater.stop()
            await application.stop()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Shutting down…")
        sys.exit(0)


if __name__ == "__main__":
    main()
