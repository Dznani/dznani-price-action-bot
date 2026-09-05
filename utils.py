"""
utils.py — shared helpers for Dznani Signals Bot.

Logging setup, UTC time formatting, and the Fear & Greed index fetch.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from datetime import datetime, timezone
from typing import Optional

import requests

FNG_URL = "https://api.alternative.me/fng/"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Configure root logging for the bot:
      - logs/bot.log    -> INFO and above, rotating (5 x 5MB)
      - logs/error.log  -> ERROR and above, rotating (5 x 5MB)
      - console         -> INFO and above
    Safe to call once at startup. Returns the "dznani" logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("dznani")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        # already configured (e.g. re-imported) — don't double-attach handlers
        return logger

    fmt = logging.Formatter(
        "%(asctime)s UTC | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt.converter = lambda *args: datetime.now(timezone.utc).timetuple()

    bot_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "bot.log"), maxBytes=5 * 1024 * 1024, backupCount=5
    )
    bot_handler.setLevel(logging.INFO)
    bot_handler.setFormatter(fmt)

    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "error.log"), maxBytes=5 * 1024 * 1024, backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger.addHandler(bot_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    return logger


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_time_hhmm_utc(dt: Optional[datetime] = None) -> str:
    dt = dt or utc_now()
    return dt.strftime("%H:%M UTC")


def format_date(dt: Optional[datetime] = None) -> str:
    dt = dt or utc_now()
    return dt.strftime("%Y-%m-%d")


def iso_now() -> str:
    return utc_now().isoformat()


# --------------------------------------------------------------------------- #
# Price formatting (shared — used by telegram_bot.py and notion_exporter.py)
# --------------------------------------------------------------------------- #
def format_price(price: float) -> str:
    price = float(price)
    if price == 0:
        return "$0.00"
    if price >= 1:
        decimals = 2
    elif price >= 0.1:
        decimals = 4
    elif price >= 0.01:
        decimals = 5
    elif price >= 0.0001:
        decimals = 6
    else:
        decimals = 8
    return f"${price:,.{decimals}f}"


# --------------------------------------------------------------------------- #
# Fear & Greed Index
# --------------------------------------------------------------------------- #
def fetch_fear_greed(timeout: float = 6.0) -> dict:
    """
    Fetch the current crypto Fear & Greed index.
    Returns {"value": int, "label": str}. Falls back to a neutral
    placeholder if the request fails, so a network hiccup never
    breaks signal-card rendering.
    """
    logger = logging.getLogger("dznani")
    try:
        resp = requests.get(FNG_URL, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()["data"][0]
        return {"value": int(data["value"]), "label": data["value_classification"]}
    except Exception as e:  # noqa: BLE001 — this must never raise upstream
        logger.warning("Fear & Greed fetch failed, using neutral fallback: %s", e)
        return {"value": 50, "label": "Neutral"}
