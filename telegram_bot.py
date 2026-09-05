"""
telegram_bot.py — Telegram command handlers and message formatting
for Dznani Signals Bot. Built on python-telegram-bot v20+ (async).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import indicators as ind
import liquidity
import risk
import strategy
import structure as struct_engine
from database import Database
from exchange import BinanceExchange
from utils import fetch_fear_greed, format_price, format_time_hhmm_utc

logger = logging.getLogger("dznani.telegram")


def _split_symbol(symbol: str) -> tuple[str, str]:
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
    else:
        base, quote = symbol, "USDT"
    return base.upper(), quote.upper()


def build_exchange_links_keyboard(symbol: str, extra_rows: Optional[list] = None) -> InlineKeyboardMarkup:
    base, quote = _split_symbol(symbol)
    binance_url = f"https://www.binance.com/en/trade/{base}_{quote}"
    tradingview_url = f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{base}{quote}"
    rows = list(extra_rows) if extra_rows else []
    rows.append([
        InlineKeyboardButton("📊 Binance", url=binance_url),
        InlineKeyboardButton("📈 TradingView", url=tradingview_url),
    ])
    return InlineKeyboardMarkup(rows)


# --------------------------------------------------------------------------- #
# Signal card formatting
# --------------------------------------------------------------------------- #
def _check(ok: bool) -> str:
    return "✅" if ok else "❌"


def _decision_icon(decision: str) -> str:
    return {"VALID": "🟢", "WAIT": "🟡", "NO TRADE": "⚪"}.get(decision, "⚪")


def build_signal_card(signal: dict) -> str:
    """
    Price-action signal card (spec section 20). Works for every decision
    state — VALID / WAIT / NO TRADE — because a NO TRADE card with a clear
    reason is mandatory output, not an edge case (spec section 21).
    """
    decision = signal.get("decision", "NO TRADE")
    if decision == "NO TRADE":
        return build_no_trade_card(signal)

    ind_ = signal["indicators"]
    targets = {t["label"]: t for t in signal.get("targets", [])}
    direction = signal["direction"]
    icon = _decision_icon(decision)
    fng = fetch_fear_greed()
    struct4h = signal["structure_4h"]
    struct1h = signal["structure_1h"]
    loc = signal["location"]
    location_assessment = loc.get("assessment", {})
    confirmation = signal.get("entry_confirmation") or {}
    rr = signal.get("rr", {})
    pos = signal.get("position_plan", {})
    ext_info = signal.get("extension")

    tp_conflict = signal.get("targets_structure_conflict") or {}
    rr_to_targets = signal.get("rr_to_targets") or {}
    tp_flags = tp_conflict.get("conflicts", {})

    lines = [
        f"{icon} {signal['signal_type']} | {signal['symbol']} ({signal.get('entry_model','-') or '-'}-model)",
        "",
        "[ MARKET CONTEXT ]",
        f"┌ 4H Trend: {struct4h['trend']} (conf {struct4h['confidence']})",
        f"└ 1H Execution Structure: {struct1h['trend']} (conf {struct1h['confidence']})",
        "",
        "[ STRUCTURE ]",
        f"┌ Protected HL/LH: {struct1h.get('protected_low') or struct1h.get('protected_high') or '—'}",
        f"├ 1H Structure: {struct1h['trend']} (conf {struct1h['confidence']})",
        f"├ CHoCH: {_check(bool(signal.get('choch')))} {signal['choch']['direction'] if signal.get('choch') else '—'}",
        f"├ BOS: {_check(bool(signal.get('bos')))} {signal['bos']['direction'] if signal.get('bos') else '—'}",
        f"└ Retest: {_check(bool(signal.get('retest') and signal['retest'].get('held')))} "
        f"{'held' if signal.get('retest') and signal['retest'].get('held') else ('occurred, not held' if signal.get('retest') and signal['retest'].get('occurred') else '—')}",
        "",
        "[ LIQUIDITY ]",
        f"┌ Sweep: {_check(bool(signal['liquidity'].get('sweep')))} "
        f"{signal['liquidity']['sweep']['liquidity_type'] + ' ' + str(signal['liquidity']['sweep']['sweep_direction']) if signal['liquidity'].get('sweep') else '—'}",
        f"└ Pools tracked: {signal['liquidity'].get('pools', '—')}",
        "",
        "[ LOCATION ]",
        f"┌ Premium/Discount: {loc['premium_discount']['zone']} ({loc['premium_discount']['pct_of_range']*100:.0f}% of range)",
        f"├ Grade: {location_assessment.get('location_grade', '—')} | score {location_assessment.get('location_score', 0)}/100 | zone {location_assessment.get('zone_type') or '—'}",
        f"├ Distance to zone: {location_assessment.get('distance_to_zone', '—')}%",
        f"└ Confluence: supporting {loc.get('supporting_score', loc.get('zones_near_price', 0))} | opposing {loc.get('opposing_score', 0)}",
        "",
    ]
    if ext_info:
        chase_icon = {"EARLY": "🟢", "EXTENDED": "🟡", "OVEREXTENDED": "🔴"}.get(ext_info.get("label"), "")
        lines += [
            "[ ENTRY QUALITY ]",
            f"┌ Extension: {chase_icon} {ext_info.get('label', '—')}",
            f"├ Distance from break: {ext_info.get('displacement_pct', 0)}% | ATR extension: {ext_info.get('atr_multiple', 0)}x",
            f"└ Chase score: {ext_info.get('chase_score', 0)}/100",
        ]
        if ext_info.get("label") != "EARLY" and ext_info.get("preferred_entry_low"):
            lines.append(f"  📍 Preferred zone: {format_price(ext_info['preferred_entry_low'])}–{format_price(ext_info['preferred_entry_high'])}")
        lines.append("")
    lines += [
        "[ ENTRY CONFIRMATION ]",
        f"└ Status: {confirmation.get('status', '—')} | quality {confirmation.get('quality', '—')} — {confirmation.get('reason', '—')}",
        "",
    ]
    lines += [
        "[ ENTRY & RISK ]",
        f"┌ ▶ Entry: {format_price(signal['entry'])}",
        f"├ Structural invalidation: {format_price(signal.get('structural_invalidation') or signal['stop_loss'])}",
        f"├ ⛔ Final SL: {format_price(signal['stop_loss'])} ({signal['stop_loss_pct']}% risk)",
        f"├ 💰 Max position: ${pos.get('max_position_usd', 0):,.0f} | 30% now: ${pos.get('aggressive_usd', 0):,.0f} | 70% on confirm: ${pos.get('confirmation_usd', 0):,.0f}",
        f"├ 📏 {'Nearest resistance' if direction == 'BUY' else 'Nearest support'}: "
        f"{format_price(signal.get('nearest_resistance') or signal.get('nearest_support') or 0) if (signal.get('nearest_resistance') or signal.get('nearest_support')) else '—'} "
        f"(room {signal.get('available_room_pct', 0)}%)",
        f"└ ⚖️ R:R: {rr.get('rr', 0)} {'✅' if rr.get('passes_minimum') else '❌'}",
        "",
        "[ TARGETS ]" + (" ⚠ structure conflict" if tp_conflict.get("has_conflict") else ""),
    ]
    for label in ("TP1", "TP2", "TP3"):
        t = targets.get(label)
        if not t:
            continue
        flag = " ⚠" if label in tp_flags else ""
        rr_to_t = rr_to_targets.get(label)
        rr_note = f", R:R {rr_to_t}" if rr_to_t is not None else ""
        lines.append(f"  • {label}: {format_price(t['price'])} (+{int(t['pct']*100)}%, sell {int(t['sell_fraction']*100)}%{rr_note}){flag}")

    lines += [
        "",
        "[ INDICATORS — secondary context only ]",
        f"┌ RSI Divergence: {_check(bool(ind_['rsi_divergence']))} {ind_['rsi_divergence'] or '—'}",
        f"├ MFI (14): {ind_['mfi']} | ADX (14): {ind_['adx']} | EMA stack: {ind_['ema_stack']}",
        f"└ Volume: {ind_['volume_spike_ratio']}x avg | Momentum: {ind_['momentum_state'] or '—'}",
        "",
        "[ DIAGNOSIS ]",
        f"┌ Setup quality: {signal['setup_grade']} ({signal['setup_score']}/100)",
        f"├ Trade quality: {signal['trade_quality_grade']} ({signal['trade_quality_score']}/100)",
        f"├ Decision: {decision} — {signal['reason']}",
        f"└ 😱 Sentiment: {fng['value']}/100 — {fng['label']}",
        "",
        f"🕐 Issued at {format_time_hhmm_utc()}",
        "📊 Spot-only: scale in per plan, never add the 70% just because the 30% is losing.",
    ]
    return "\n".join(lines)


def build_no_trade_card(signal: dict) -> str:
    """Mandatory NO TRADE card (spec section 21) — shown even for an A+ setup
    when R:R or room fails, so the reasoning is never hidden."""
    struct4h = signal.get("structure_4h", {})
    struct1h = signal.get("structure_1h", {})
    rr = signal.get("rr", {})
    lines = [
        f"⚪ {signal.get('signal_type', 'NO TRADE')} | {signal['symbol']}",
        "",
        f"Setup quality: {signal.get('setup_grade', '—')} ({signal.get('setup_score', 0)}/100)",
        f"Trade quality: {signal.get('trade_quality_grade', '—')} ({signal.get('trade_quality_score', 0)}/100)",
        "",
        f"4H trend: {struct4h.get('trend', '—')} | 1H structure: {struct1h.get('trend', '—')}",
        f"CHoCH: {_check(bool(signal.get('choch')))}  BOS: {_check(bool(signal.get('bos')))}  "
        f"Retest: {_check(bool(signal.get('retest') and signal['retest'].get('held')))}",
    ]
    if rr:
        obstacle_label = "Nearest resistance" if signal.get("direction") == "BUY" else "Nearest support"
        lines.append(f"{obstacle_label}: {signal.get('available_room_pct', 0)}% away | R:R: {rr.get('rr', 0)} "
                     f"{'✅' if rr.get('passes_minimum') else '❌'}")
    lines += [
        "",
        f"Decision: {signal.get('decision', 'NO TRADE')}",
        f"Reason: {signal.get('reason', '—')}",
        "",
        f"🕐 Issued at {format_time_hhmm_utc()}",
    ]
    return "\n".join(lines)

# --------------------------------------------------------------------------- #
# Bot wiring
# --------------------------------------------------------------------------- #
class DznaniTelegramBot:
    def __init__(self, token: str, chat_id: str, db: Database, exchange: BinanceExchange):
        self.token = token
        self.chat_id = chat_id
        self.db = db
        self.exchange = exchange
        self.app: Optional[Application] = None

    def build(self) -> Application:
        app = Application.builder().token(self.token).build()

        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_start))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("scan", self.cmd_scan))
        app.add_handler(CommandHandler("watchlist", self.cmd_watchlist))
        app.add_handler(CommandHandler("addwatch", self.cmd_addwatch))
        app.add_handler(CommandHandler("removewatch", self.cmd_removewatch))
        app.add_handler(CommandHandler("watchscan", self.cmd_watchscan))
        app.add_handler(CommandHandler("pending", self.cmd_pending))
        app.add_handler(CommandHandler("performance", self.cmd_performance))
        app.add_handler(CommandHandler("risk", self.cmd_risk))
        app.add_handler(CommandHandler("history", self.cmd_history))
        app.add_handler(CommandHandler("analyze", self.cmd_analyze))
        app.add_handler(CommandHandler("structure", self.cmd_structure))
        app.add_handler(CommandHandler("liquidity", self.cmd_liquidity))
        app.add_handler(CommandHandler("setup", self.cmd_setup))
        app.add_handler(CommandHandler("settings", self.cmd_settings))
        app.add_handler(CommandHandler("set", self.cmd_set))
        app.add_handler(CommandHandler("pause", self.cmd_pause))
        app.add_handler(CommandHandler("slguide", self.cmd_slguide))
        app.add_handler(CommandHandler("alert", self.cmd_alert))
        app.add_handler(CommandHandler("trade", self.cmd_trade))
        app.add_handler(CommandHandler("funded_status", self.cmd_funded_status))
        app.add_handler(CommandHandler("journal", self.cmd_journal))
        app.add_handler(CommandHandler("export", self.cmd_export))
        app.add_handler(CallbackQueryHandler(self.on_button))
        # Bare ticker text (e.g. "SOL") triggers a quick scan
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_plain_ticker))

        self.app = app
        return app

    async def send_message(self, text: str) -> None:
        if self.app is None:
            raise RuntimeError("Bot not built yet — call build() first")
        await self.app.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_signal_card(self, signal: dict) -> None:
        text = build_signal_card(signal)
        keyboard = build_exchange_links_keyboard(signal["symbol"])
        # No code block – send as plain text
        await self.app.bot.send_message(
            chat_id=self.chat_id, text=text, reply_markup=keyboard
        )

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 Dznani Signals Bot — Professional Price Action Edition\n\n"
            "Send a ticker (e.g. SOL) for a quick scan, or use:\n"
            "/analyze COIN /structure COIN /liquidity COIN /setup COIN /risk COIN\n"
            "/status /scan COIN /watchlist /addwatch COIN /removewatch COIN\n"
            "/watchscan /pending /performance /risk /history /settings /set key value\n"
            "/pause /slguide COIN ENTRY WICK RISK /alert COIN PRICE\n"
            "/trade COIN PRICE BUY SIZE /trade COIN PRICE SELL\n"
            "/funded_status /journal /export\n\n"
            "Price action (structure, liquidity, CHoCH/BOS, location, R:R) drives every "
            "decision. Indicators are shown as secondary context only — never the trigger."
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        settings = self.db.get_settings()
        available = self.exchange.is_available()
        open_trades = self.db.get_open_trades()
        # FIXED (audit, spec Part 22): this used to display min_confirmations
        # /trend_filter_enabled/pro_filters_enabled — all dead settings the
        # new price-action engine never reads. Showing them here implied
        # they still governed behavior. Replaced with the settings that
        # actually drive strategy.evaluate_symbol().
        text = (
            f"🤖 Bot Status — Price Action Edition\n"
            f"Exchange (Binance): {'🟢 online' if available else '🔴 unreachable'}\n"
            f"Paused: {'yes' if settings.get('paused') else 'no'}\n"
            f"Minimum R:R: {settings.get('minimum_rr', 2.0)}\n"
            f"Max structural risk: {settings.get('max_structural_risk_pct', 8.0)}%\n"
            f"Risk per trade: ${settings.get('risk_per_trade_usd', 250.0):,.0f}\n"
            f"Aggressive/Confirmation split: {int(settings.get('aggressive_entry_pct', 0.30)*100)}% / {int(settings.get('confirmation_add_pct', 0.70)*100)}%\n"
            f"Open trades: {len(open_trades)}\n"
            f"Watchlist size: {len(self.db.get_watchlist())}"
        )
        await update.message.reply_text(text)

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /scan COIN (e.g. /scan SOL)")
            return
        await self._quick_scan(update, context.args[0])

    async def on_plain_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (update.message.text or "").strip()
        if not text or " " in text or len(text) > 12:
            return
        await self._quick_scan(update, text)

    async def _fetch_both_timeframes(self, symbol: str):
        df_1h = self.exchange.fetch_ohlcv(symbol, timeframe="1h", limit=300)
        df_4h = self.exchange.fetch_ohlcv(symbol, timeframe="4h", limit=150)
        return df_4h, df_1h

    async def _quick_scan(self, update: Update, raw_symbol: str):
        symbol = self._normalize_symbol(raw_symbol)
        try:
            df_4h, df_1h = await self._fetch_both_timeframes(symbol)
        except Exception as e:
            await update.message.reply_text(f"Couldn't fetch {symbol}: {e}")
            return

        settings = self.db.get_settings()
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            quote_volume = ticker.get("quoteVolume")
        except Exception:
            quote_volume = None

        signal = strategy.evaluate_symbol(symbol, df_4h, df_1h, settings, quote_volume_24h=quote_volume)
        if signal is None:
            await update.message.reply_text(f"{symbol}: not enough history yet, or below the min-liquidity filter.")
            return

        keyboard = build_exchange_links_keyboard(
            symbol,
            extra_rows=[[InlineKeyboardButton("❌ Dismiss", callback_data=f"dismiss:{symbol}")]]
            if signal["decision"] != "VALID" else None,
        )
        await update.message.reply_text(build_signal_card(signal), reply_markup=keyboard)

    async def on_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data
        if data and data.startswith("addwatch:"):
            symbol = data.split(":", 1)[1]
            added = self.db.add_to_watchlist(symbol)
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(f"{'Added' if added else 'Already on'} watchlist: {symbol}")
        elif data and data.startswith("dismiss:"):
            # Delete the message
            try:
                await query.delete_message()
            except Exception:
                await query.edit_message_reply_markup(reply_markup=None)

    async def cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        wl = self.db.get_watchlist()
        await update.message.reply_text("Watchlist:\n" + ("\n".join(wl) if wl else "(empty — using top 200 USDT pairs)"))

    async def cmd_addwatch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /addwatch COIN")
            return
        symbol = self._normalize_symbol(context.args[0])
        added = self.db.add_to_watchlist(symbol)
        await update.message.reply_text(f"{'Added' if added else 'Already on'} watchlist: {symbol}")

    async def cmd_removewatch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /removewatch COIN")
            return
        symbol = self._normalize_symbol(context.args[0])
        removed = self.db.remove_from_watchlist(symbol)
        await update.message.reply_text(f"{'Removed' if removed else 'Not on'} watchlist: {symbol}")

    async def cmd_watchscan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        wl = self.db.get_watchlist()
        if not wl:
            await update.message.reply_text("Watchlist is empty — nothing to scan.")
            return
        await update.message.reply_text(f"Scanning {len(wl)} watchlist symbols…")
        settings = self.db.get_settings()
        hits = 0
        for symbol in wl:
            try:
                df_4h, df_1h = await self._fetch_both_timeframes(symbol)
                signal = strategy.evaluate_symbol(symbol, df_4h, df_1h, settings)
                if signal and signal["decision"] == "VALID":
                    hits += 1
                    await self.send_signal_card(signal)
            except Exception as e:
                logger.warning("watchscan failed for %s: %s", symbol, e)
        await update.message.reply_text(f"Watchlist scan complete — {hits} valid setup(s) found.")

    _GRADE_WEIGHT = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1}
    _PENDING_SIGNAL_TYPES = {"VALID", "NO CHASE", "WAIT FOR RETEST"}

    def _priority_score(self, signal: dict) -> float:
        """Mirrors the Notion dashboard's Priority Score formula
        (grade_weight * R:R / 2) so /pending and the Notion board agree —
        computed independently here since this command never reads back
        from Notion (one-way export only, no two-way sync)."""
        grade_weight = self._GRADE_WEIGHT.get(signal.get("setup_grade"), 0)
        rr = (signal.get("rr") or {}).get("rr") or 0.0
        return round(grade_weight * (rr / 2), 2)

    async def cmd_pending(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Live watchlist scan (not a read from Notion — this bot never
        reads Notion back) showing the highest-priority actionable/
        near-actionable setups right now: VALID, NO CHASE, WAIT FOR RETEST."""
        wl = self.db.get_watchlist()
        if not wl:
            await update.message.reply_text("Watchlist is empty — nothing to show.")
            return
        settings = self.db.get_settings()
        candidates = []
        for symbol in wl:
            try:
                df_4h, df_1h = await self._fetch_both_timeframes(symbol)
                signal = strategy.evaluate_symbol(symbol, df_4h, df_1h, settings)
                if signal and signal.get("signal_type") in self._PENDING_SIGNAL_TYPES:
                    candidates.append(signal)
            except Exception as e:
                logger.warning("pending scan failed for %s: %s", symbol, e)

        if not candidates:
            await update.message.reply_text("📋 DZNANI WATCHLIST\n\nNothing pending right now.")
            return

        candidates.sort(key=self._priority_score, reverse=True)
        icon = {"VALID": "🟢", "NO CHASE": "🔴", "WAIT FOR RETEST": "🟡"}
        lines = ["📋 DZNANI WATCHLIST", ""]
        for i, sig in enumerate(candidates[:10], 1):
            ext = sig.get("extension") or {}
            zone = ""
            if ext.get("preferred_entry_low") is not None and ext.get("preferred_entry_high") is not None:
                zone = f"\n   Zone: {format_price(ext['preferred_entry_low'])}–{format_price(ext['preferred_entry_high'])}"
            rr = (sig.get("rr") or {}).get("rr")
            lines.append(
                f"{i}. {icon.get(sig['signal_type'], '⚪')} {sig['symbol']}\n"
                f"   {sig.get('setup_grade', '—')} | R:R {rr if rr is not None else '—'}\n"
                f"   {sig['signal_type']}{zone}"
            )
        await update.message.reply_text("\n".join(lines))

    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        trades = self.db.get_all_trades()
        closed = [t for t in trades if t["status"] == "closed"]
        wins = [t for t in closed if t["pnl"] and t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] and t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in closed if t["pnl"] is not None)
        win_rate = (len(wins) / len(closed) * 100) if closed else 0.0
        await update.message.reply_text(
            f"📊 Performance\n"
            f"Closed trades: {len(closed)}\n"
            f"Wins: {len(wins)} | Losses: {len(losses)}\n"
            f"Win rate: {win_rate:.1f}%\n"
            f"Total P&L: ${total_pnl:,.2f}"
        )

    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if context.args:
            await self._coin_risk(update, context.args[0])
            return
        settings = self.db.get_settings()
        capital = settings.get("capital", risk.DEFAULT_CAPITAL)
        limit_pct = settings.get("daily_loss_limit_pct", risk.DEFAULT_DAILY_LOSS_LIMIT_PCT)
        today_pnl = self.db.get_today_pnl()
        buffer_remaining = risk.daily_loss_buffer_remaining(today_pnl, capital, limit_pct)
        await update.message.reply_text(
            f"⚖️ Risk\n"
            f"Capital: ${capital:,.0f}\n"
            f"Daily loss limit: ${capital * limit_pct:,.0f} ({limit_pct*100:.0f}%)\n"
            f"Today's P&L: ${today_pnl:,.2f}\n"
            f"Buffer remaining: ${buffer_remaining:,.2f}\n"
            f"Breached: {'YES — no new trades' if risk.daily_loss_limit_breached(today_pnl, capital, limit_pct) else 'no'}"
        )

    async def _coin_risk(self, update: Update, raw_symbol: str):
        """`/risk COIN` — structural SL, dollar risk, and the 30/70 position
        plan for that symbol right now (spec section 26/12)."""
        symbol = self._normalize_symbol(raw_symbol)
        try:
            df_4h, df_1h = await self._fetch_both_timeframes(symbol)
        except Exception as e:
            await update.message.reply_text(f"Couldn't fetch {symbol}: {e}")
            return
        settings = self.db.get_settings()
        signal = strategy.evaluate_symbol(symbol, df_4h, df_1h, settings)
        if signal is None or signal.get("stop_loss") is None:
            await update.message.reply_text(f"{symbol}: no structural stop loss to show yet — no directional bias confirmed.")
            return
        pos = signal.get("position_plan", {})
        await update.message.reply_text(
            f"⚖️ Structural Risk — {symbol}\n"
            f"Entry: {format_price(signal['entry'])}\n"
            f"Structural SL: {format_price(signal['stop_loss'])} ({signal['stop_loss_pct']}%)\n"
            f"Max position: ${pos.get('max_position_usd', 0):,.0f}\n"
            f"30% initial: ${pos.get('aggressive_usd', 0):,.0f} | 70% on confirmation: ${pos.get('confirmation_usd', 0):,.0f}\n"
            f"R:R: {signal['rr']['rr']} | Available room: {signal['available_room_pct']}%"
        )

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        signals = self.db.get_recent_signals(10)
        if not signals:
            await update.message.reply_text("No signal history yet.")
            return
        lines = ["🕓 Recent Signals:"]
        for s in signals:
            lines.append(f"{s.get('timestamp','?')[:16]} | {s.get('symbol')} {s.get('direction')} {s.get('strength')}/5")
        await update.message.reply_text("\n".join(lines))

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Full signal card for one coin — same output the scanner would send if it fired now."""
        if not context.args:
            await update.message.reply_text("Usage: /analyze COIN")
            return
        await self._quick_scan(update, context.args[0])

    async def cmd_setup(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Alias for /analyze — named to match the spec's setup-quality terminology."""
        await self.cmd_analyze(update, context)

    async def cmd_structure(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /structure COIN")
            return
        symbol = self._normalize_symbol(context.args[0])
        try:
            df_4h, df_1h = await self._fetch_both_timeframes(symbol)
        except Exception as e:
            await update.message.reply_text(f"Couldn't fetch {symbol}: {e}")
            return
        settings = self.db.get_settings()
        state_4h = struct_engine.analyze_structure(df_4h, settings.get("swing_left", 2), settings.get("swing_right", 2))
        state_1h = struct_engine.analyze_structure_1h(df_1h, settings.get("swing_left", 2), settings.get("swing_right", 2),
                                                        bos_min_displacement_pct=settings.get("bos_min_displacement_pct", 0.15),
                                                        min_swing_significance_pct=settings.get("min_swing_significance_pct", 0.3))
        lines = [
            f"📐 Structure — {symbol}",
            f"4H: {state_4h.trend} (confidence {state_4h.confidence})",
            f"  last HH: {round(state_4h.last_HH.price,4) if state_4h.last_HH else '—'} | last HL: {round(state_4h.last_HL.price,4) if state_4h.last_HL else '—'}",
            f"  last LH: {round(state_4h.last_LH.price,4) if state_4h.last_LH else '—'} | last LL: {round(state_4h.last_LL.price,4) if state_4h.last_LL else '—'}",
            f"1H: {state_1h.trend} (confidence {state_1h.confidence})",
        ]
        for e in state_1h.events[-5:]:
            lines.append(f"  • {e.kind} ({e.direction}) at {format_price(e.level)}, disp {e.displacement_pct}%")
        await update.message.reply_text("\n".join(lines))

    async def cmd_liquidity(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /liquidity COIN")
            return
        symbol = self._normalize_symbol(context.args[0])
        try:
            _, df_1h = await self._fetch_both_timeframes(symbol)
        except Exception as e:
            await update.message.reply_text(f"Couldn't fetch {symbol}: {e}")
            return
        settings = self.db.get_settings()
        state_1h = struct_engine.analyze_structure_1h(df_1h, settings.get("swing_left", 2), settings.get("swing_right", 2))
        pools = liquidity.detect_liquidity_pools(df_1h, state_1h, equal_tolerance_pct=settings.get("liquidity_equal_tolerance_pct", 0.15))
        sweep = liquidity.best_recent_sweep(df_1h, pools, sweep_confirm_candles=settings.get("sweep_confirm_candles", 3))
        lines = [f"💧 Liquidity — {symbol}", f"Pools tracked: {len(pools)}"]
        for p in pools[:6]:
            lines.append(f"  • {p.liquidity_type} @ {format_price(p.level)} (touches {p.touches}, {p.freshness_index} candles ago)")
        if sweep:
            lines.append(f"\nMost recent sweep: {sweep.liquidity_type} {sweep.sweep_direction or '(accepted beyond level)'} "
                          f"— depth {sweep.sweep_depth_pct}%, rejection {sweep.rejection_strength}, confidence {sweep.confidence}")
        else:
            lines.append("\nNo recent sweep detected.")
        await update.message.reply_text("\n".join(lines))

    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        settings = self.db.get_settings()
        lines = ["⚙️ Settings:"] + [f"{k} = {v}" for k, v in settings.items()]
        await update.message.reply_text("\n".join(lines))

    async def cmd_set(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if len(context.args) < 2:
            await update.message.reply_text("Usage: /set key value  (e.g. /set minimum_rr 2.5, /set risk_per_trade_usd 300)")
            return
        key, raw_value = context.args[0], " ".join(context.args[1:])
        value = self._coerce_setting_value(key, raw_value)

        key_map = {"pro_filters": "pro_filters_enabled", "trend_filter": "trend_filter_enabled"}
        key = key_map.get(key, key)

        self.db.set_setting(key, value)
        # FIXED (audit, spec Part 22): min_confirmations/trend_filter_enabled/
        # pro_filters_enabled are legacy keys the new price-action engine
        # never reads — setting them used to silently "succeed" with zero
        # effect, which is misleading. Now flagged explicitly.
        if key in self._LEGACY_INERT_SETTINGS:
            await update.message.reply_text(
                f"Set {key} = {value}\n"
                f"⚠️ Note: '{key}' is a legacy setting from the old indicator-confluence engine and has "
                f"no effect on the current price-action strategy. Relevant settings: minimum_rr, "
                f"risk_per_trade_usd, max_structural_risk_pct, bos_min_displacement_pct, retest_tolerance_pct."
            )
        else:
            await update.message.reply_text(f"Set {key} = {value}")

    _LEGACY_INERT_SETTINGS = {"min_confirmations", "trend_filter_enabled", "pro_filters_enabled"}

    @staticmethod
    def _coerce_setting_value(key: str, raw_value: str):
        if raw_value.lower() in ("on", "true", "yes"):
            return True
        if raw_value.lower() in ("off", "false", "no"):
            return False
        try:
            if "." in raw_value:
                return float(raw_value)
            return int(raw_value)
        except ValueError:
            return raw_value

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        settings = self.db.get_settings()
        new_state = not settings.get("paused", False)
        self.db.set_setting("paused", new_state)
        await update.message.reply_text(f"Bot {'paused ⏸' if new_state else 'resumed ▶️'}")

    async def cmd_slguide(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if len(context.args) < 4:
            await update.message.reply_text("Usage: /slguide COIN ENTRY WICK_PRICE RISK_USD")
            return
        coin, entry_s, wick_s, risk_usd_s = context.args[:4]
        try:
            entry = float(entry_s)
            wick = float(wick_s)
            risk_usd = float(risk_usd_s)
        except ValueError:
            await update.message.reply_text("ENTRY, WICK, and RISK must be numbers.")
            return

        sl_distance = abs(entry - wick)
        sl_pct = sl_distance / entry if entry else 0
        position_size = (risk_usd / sl_pct) if sl_pct else 0
        await update.message.reply_text(
            f"🧮 SL Guide — {coin.upper()}\n"
            f"Entry: ${entry:,.4f}\n"
            f"Stop (wick): ${wick:,.4f}\n"
            f"SL distance: {sl_pct*100:.2f}%\n"
            f"Suggested position size for ${risk_usd:,.0f} risk: ${position_size:,.0f}"
        )

    async def cmd_alert(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if len(context.args) < 2:
            await update.message.reply_text("Usage: /alert COIN PRICE")
            return
        coin, price_s = context.args[0], context.args[1]
        await update.message.reply_text(
            f"🔔 Alert set for {self._normalize_symbol(coin)} at ${price_s}.\n"
            f"(Price-alert triggering runs in the main scan loop — see main.py.)"
        )

    async def cmd_trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if len(context.args) < 3:
            await update.message.reply_text("Usage:\n/trade COIN PRICE BUY SIZE\n/trade COIN PRICE SELL")
            return
        coin = self._normalize_symbol(context.args[0])
        try:
            price = float(context.args[1])
        except ValueError:
            await update.message.reply_text("PRICE must be a number.")
            return
        action = context.args[2].upper()

        if action == "BUY":
            if len(context.args) < 4:
                await update.message.reply_text("Usage: /trade COIN PRICE BUY SIZE")
                return
            try:
                size = float(context.args[3])
            except ValueError:
                await update.message.reply_text("SIZE must be a number.")
                return
            trade = self.db.open_trade(coin, price, size, "BUY")
            await update.message.reply_text(f"✅ Opened {coin} BUY @ ${price:,.2f} size ${size:,.0f} (id {trade['id']})")

        elif action == "SELL":
            closed = self.db.close_trade(coin, price, risk.calculate_pnl)
            if closed:
                pnl = closed["pnl"]
                emoji = "🟢" if pnl >= 0 else "🔴"
                await update.message.reply_text(f"{emoji} Closed {coin} @ ${price:,.2f} — P&L: ${pnl:,.2f}")
            else:
                await update.message.reply_text(f"No open trade found for {coin}.")
        else:
            await update.message.reply_text("Third argument must be BUY or SELL.")

    async def cmd_funded_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        settings = self.db.get_settings()
        capital = settings.get("capital", risk.DEFAULT_CAPITAL)
        limit_pct = settings.get("daily_loss_limit_pct", risk.DEFAULT_DAILY_LOSS_LIMIT_PCT)
        open_trades = self.db.get_open_trades()
        all_trades = self.db.get_all_trades()
        closed = [t for t in all_trades if t["status"] == "closed"]
        total_pnl = sum(t["pnl"] for t in closed if t["pnl"] is not None)
        today_pnl = self.db.get_today_pnl()
        buffer_remaining = risk.daily_loss_buffer_remaining(today_pnl, capital, limit_pct)

        lines = [
            "💰 Funded Account Status",
            f"Capital: ${capital:,.0f}",
            f"Total realized P&L: ${total_pnl:,.2f}",
            f"Today's P&L: ${today_pnl:,.2f}",
            f"Daily loss buffer remaining: ${buffer_remaining:,.2f}",
            f"Open positions: {len(open_trades)}",
        ]
        for t in open_trades:
            lines.append(f"  • {t['symbol']} {t['direction']} @ ${t['entry_price']:,.2f} size ${t['size']:,.0f}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_journal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        trades = [t for t in self.db.get_all_trades() if t["status"] == "closed" and t["pnl"] is not None]
        if not trades:
            await update.message.reply_text("No closed trades yet.")
            return
        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        best = max(trades, key=lambda t: t["pnl"])
        worst = min(trades, key=lambda t: t["pnl"])
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss else float("inf")

        await update.message.reply_text(
            "📔 Trade Journal\n"
            f"Total trades: {len(trades)}\n"
            f"Avg win: ${avg_win:,.2f} | Avg loss: ${avg_loss:,.2f}\n"
            f"Best: {best['symbol']} ${best['pnl']:,.2f} | Worst: {worst['symbol']} ${worst['pnl']:,.2f}\n"
            f"Profit factor: {profit_factor:.2f}"
        )

    async def cmd_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        import csv
        import io

        trades = self.db.get_all_trades()
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["id", "symbol", "direction", "entry_price", "exit_price", "size", "pnl", "status", "open_date", "close_date"],
        )
        writer.writeheader()
        for t in trades:
            writer.writerow(t)
        buf.seek(0)

        await update.message.reply_document(
            document=io.BytesIO(buf.getvalue().encode("utf-8")),
            filename=f"dznani_trades_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_symbol(raw: str) -> str:
        raw = raw.upper().strip()
        if "/" in raw:
            return raw
        return f"{raw}/USDT"
