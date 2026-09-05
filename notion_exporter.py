"""
notion_exporter.py — one-way Notion dashboard export for Dznani Signals Bot.

STRICTLY ISOLATED FROM TRADING LOGIC: this module only ever reads an
already-fully-formed signal dict (as produced by strategy.evaluate_symbol)
and writes it to Notion as a new page. It never influences structure,
CHoCH/BOS, extension, location, scoring, sizing, or the 30/70 model in any
way, and it never reads status/anything back from Notion (one-way,
create-only — never updates or deletes an existing page, so a person's
manual Status changes in Notion are never touched by this bot).

SAFETY CONTRACT: export_signal() must NEVER raise. Any failure — missing
package, bad credentials, network error, malformed signal — is caught,
logged as a warning, and the caller proceeds exactly as if Notion didn't
exist. The trading/scanning loop must never stop or slow meaningfully
because Notion is unavailable.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("dznani.notion")

try:
    from notion_client import Client as _NotionClient
    from notion_client.errors import APIResponseError as _NotionAPIError
    _NOTION_CLIENT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via the "package missing" test
    _NotionClient = None
    _NotionAPIError = Exception
    _NOTION_CLIENT_AVAILABLE = False


_PREMIUM_DISCOUNT_MAP = {"premium": "Premium", "discount": "Discount"}


def _map_premium_discount(zone: Optional[str]) -> str:
    return _PREMIUM_DISCOUNT_MAP.get((zone or "").lower(), "Neutral")


def _format_preferred_zone(extension: Optional[Dict[str, Any]]) -> str:
    if not extension:
        return ""
    low = extension.get("preferred_entry_low")
    high = extension.get("preferred_entry_high")
    if low is None or high is None:
        return ""
    # Reuse the shared price formatter (utils.format_price) rather than
    # re-deriving formatting rules locally — avoids exactly the kind of
    # bug caught by this module's own tests (an earlier version used an
    # f-string with conflicting comma+significant-digits specifiers that
    # silently produced scientific notation like "$6.31e+04" for prices
    # >= $10k). Imported from utils, not telegram_bot, to keep this module
    # lightweight and decoupled from the Telegram presentation layer.
    from utils import format_price
    return f"{format_price(low)} – {format_price(high)}"


def signal_to_notion_properties(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps a signal dict (any decision/signal_type — VALID, NO CHASE, WAIT
    FOR RETEST, INVALIDATED, BEARISH CONTEXT, DIP-BUY PLAN, WATCHLIST, or
    any future type) into Notion property payloads. Handles every field
    missing/None safely — nothing here raises on incomplete data.

    Deliberately does NOT set "Priority Score" or "Status": Priority Score
    is a Notion FORMULA property (computed server-side from other
    properties — the API rejects attempts to write to formula properties),
    and Status is left for Notion's page-creation default ("Pending") so
    a person's manual changes are the only thing that ever sets it.
    Signal Type is passed through as whatever the bot's own string is
    (not remapped to a fixed enum) so new signal types the bot introduces
    later show up automatically rather than being silently dropped/broken.
    """
    symbol = signal.get("symbol") or "UNKNOWN"
    signal_type = signal.get("signal_type") or "UNKNOWN"
    direction = signal.get("direction")
    entry_price = signal.get("entry")
    if entry_price is None:
        entry_price = signal.get("current_price")
    setup_grade = signal.get("setup_grade")
    rr = (signal.get("rr") or {}).get("rr")
    extension = signal.get("extension") or {}
    chase_score = extension.get("chase_score")
    preferred_zone = _format_preferred_zone(extension)
    premium_discount = _map_premium_discount((signal.get("location") or {}).get("premium_discount", {}).get("zone"))
    timestamp = signal.get("timestamp") or datetime.now(timezone.utc).isoformat()

    properties: Dict[str, Any] = {
        "Symbol": {"title": [{"text": {"content": str(symbol)}}]},
        "Signal Type": {"select": {"name": str(signal_type)}},
        "Premium/Discount": {"select": {"name": premium_discount}},
        "Preferred Zone": {"rich_text": [{"text": {"content": preferred_zone}}]},
        "Timestamp": {"date": {"start": timestamp}},
    }
    if direction in ("BUY", "SELL"):
        properties["Direction"] = {"select": {"name": direction}}
    if entry_price is not None:
        properties["Entry Price"] = {"number": round(float(entry_price), 8)}
    if setup_grade:
        properties["Setup Grade"] = {"select": {"name": str(setup_grade)}}
    if rr is not None:
        properties["R:R"] = {"number": round(float(rr), 3)}
    if chase_score is not None:
        properties["Chase Score"] = {"number": round(float(chase_score), 1)}

    return properties


def _dedup_key(signal: Dict[str, Any]) -> str:
    """
    Coarse dedup key: same symbol + signal_type + direction + entry/current
    price bucketed to the hour. Deliberately loose — the goal is only to
    stop the exact same evaluation cycle's signal from being written twice
    (e.g. a retry path calling export twice), not to build a perfect
    signal-history model.
    """
    price = signal.get("entry") or signal.get("current_price") or 0.0
    ts = signal.get("timestamp") or ""
    hour_bucket = ts[:13] if len(ts) >= 13 else ts  # "YYYY-MM-DDTHH"
    return f"{signal.get('symbol')}|{signal.get('signal_type')}|{signal.get('direction')}|{round(float(price), 6)}|{hour_bucket}"


class NotionExporter:
    """One-way exporter: signal dict -> new Notion page. Create-only, never
    updates or reads back existing pages."""

    def __init__(
        self,
        api_key: Optional[str],
        database_id: Optional[str],
        enabled: bool = True,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
        dedup_window_hours: float = 1.0,
    ):
        self.enabled = bool(enabled and api_key and database_id)
        self.database_id = database_id
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.dedup_window_hours = dedup_window_hours
        self._recent_exports: Dict[str, datetime] = {}
        self._auth_failed = False  # once True, stop retrying — a bad key won't fix itself mid-run
        self.client = None

        if not enabled:
            logger.info("Notion export disabled by configuration.")
            return
        if not (api_key and database_id):
            logger.info("Notion export disabled — api_key/database_id not configured.")
            self.enabled = False
            return
        if not _NOTION_CLIENT_AVAILABLE:
            logger.warning(
                "Notion export enabled in config but the 'notion-client' package isn't installed "
                "(pip install notion-client) — export disabled for this run."
            )
            self.enabled = False
            return
        try:
            self.client = _NotionClient(auth=api_key)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to initialize Notion client — export disabled for this run: %s", e)
            self.enabled = False

    def _is_duplicate(self, key: str) -> bool:
        now = datetime.now(timezone.utc)
        last = self._recent_exports.get(key)
        if last is not None and (now - last).total_seconds() < self.dedup_window_hours * 3600:
            return True
        self._recent_exports[key] = now
        # Cheap unbounded-growth guard — drop stale entries occasionally
        # rather than maintaining a precise LRU for what's a best-effort cache.
        if len(self._recent_exports) > 5000:
            cutoff = now.timestamp() - self.dedup_window_hours * 3600
            self._recent_exports = {k: v for k, v in self._recent_exports.items() if v.timestamp() >= cutoff}
        return False

    def export_signal(self, signal: Optional[Dict[str, Any]]) -> bool:
        """
        Exports one signal as a new Notion page. Returns True if a page
        was created, False for every other outcome (disabled, duplicate,
        missing data, or any failure) — never raises.
        """
        if not self.enabled or signal is None:
            return False
        if self._auth_failed:
            return False  # already know the credentials are bad this run — don't hammer the API

        try:
            key = _dedup_key(signal)
            if self._is_duplicate(key):
                logger.debug("Notion export skipped (duplicate within %sh): %s", self.dedup_window_hours, key)
                return False

            properties = signal_to_notion_properties(signal)
        except Exception as e:  # noqa: BLE001
            logger.warning("Notion export: failed to build properties from signal — skipping: %s", e)
            return False

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.client.pages.create(parent={"database_id": self.database_id}, properties=properties)
                logger.info("Notion export: %s %s recorded.", signal.get("symbol"), signal.get("signal_type"))
                return True
            except _NotionAPIError as e:  # noqa: BLE001
                last_err = e
                status = getattr(e, "status", None)
                if status in (401, 403):
                    logger.warning("Notion export: authentication/permission error — disabling further attempts this run: %s", e)
                    self._auth_failed = True
                    return False
                if status == 429 or (isinstance(status, int) and 500 <= status < 600):
                    wait = self.retry_delay_seconds * attempt
                    logger.warning("Notion export: retryable error (attempt %d/%d), waiting %.1fs: %s", attempt, self.max_retries, wait, e)
                    time.sleep(wait)
                    continue
                logger.warning("Notion export: non-retryable API error — skipping this signal: %s", e)
                return False
            except Exception as e:  # noqa: BLE001 — absolute safety net; export must never crash the bot
                last_err = e
                wait = self.retry_delay_seconds * attempt
                logger.warning("Notion export: unexpected error (attempt %d/%d), waiting %.1fs: %s", attempt, self.max_retries, wait, e)
                time.sleep(wait)

        logger.warning("Notion export: exhausted retries for %s — giving up for this signal: %s", signal.get("symbol"), last_err)
        return False
