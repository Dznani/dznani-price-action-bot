"""
exchange.py — Binance-only CCXT wrapper for Dznani Signals Bot.

NOTE ON THE OLD FALLBACK CHAIN
-------------------------------
Earlier versions of this bot fell back Kraken -> Coinbase -> Bitstamp
because Binance returned HTTP 451 (geo-blocked) from Algerian and US
IPs. The bot now runs on a Hosteons EU KVM VPS in Frankfurt, Germany,
which reaches Binance directly with no geo-blocking — so that fallback
chain has been removed. This module talks to Binance only.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

import ccxt
import pandas as pd

logger = logging.getLogger("dznani.exchange")

_TIMEFRAME_HOURS = {"1h": 1, "4h": 4, "1d": 24}


def _drop_unclosed_last_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    FIXED (audit, spec Part 25): Binance/ccxt's fetch_ohlcv commonly
    returns the currently-forming candle as the last row. If that row is
    fed into structure/CHoCH/BOS/indicator calculations as if it were
    closed, every live signal is built on lookahead-tainted data — the
    exact bug the spec calls out as CRITICAL for the 4H timeframe, but it
    equally applies to 1H live scans. This drops the last row whenever its
    open time + timeframe duration is still in the future relative to now.
    """
    if df.empty:
        return df
    hours = _TIMEFRAME_HOURS.get(timeframe)
    if hours is None:
        return df  # unknown timeframe — don't guess, leave as-is
    last_open = df["timestamp"].iloc[-1]
    if hasattr(last_open, "to_pydatetime"):
        last_open = last_open.to_pydatetime()
    close_time = last_open + pd.Timedelta(hours=hours)
    now = datetime.now(timezone.utc)
    if close_time > now:
        return df.iloc[:-1].reset_index(drop=True)
    return df


class BinanceExchange:
    """Thin, retrying wrapper around ccxt.binance for market data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
        use_futures: bool = False,
    ):
        options = {
            "defaultType": "future" if use_futures else "spot",
            "fetchMarkets": ["future"] if use_futures else ["spot"],
        }
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key or "",
                "secret": api_secret or "",
                "enableRateLimit": True,
                "options": options,
            }
        )
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._markets_cache: Optional[dict] = None

    # ------------------------------------------------------------------ #
    def _ensure_valid_spot_symbol(self, symbol: str) -> None:
        """
        FIXED (support ticket — "binance requires apiKey credential" for
        THE/USDT): this wrapper NEVER requires real credentials for public
        market data — api_key/api_secret default to empty strings and no
        signed-only option is ever set (see __init__). That cryptic ccxt
        error is what ccxt raises internally when it can't resolve a
        symbol to a known market and falls through into a code path that
        assumes signing is needed, instead of a clean "symbol not found"
        error. Most common real causes: a typo, a delisted pair, or a
        symbol that's margin/futures-only and isn't actually a Binance
        SPOT market. This checks against ccxt's own loaded market list
        first and raises a clear, specific error instead of letting that
        confusing exception surface — it is NOT an authentication problem.
        """
        if self._markets_cache is None:
            self._markets_cache = self._with_retry(self.exchange.load_markets)

        market = self._markets_cache.get(symbol)
        if market is None:
            raise ValueError(
                f"{symbol} is not a recognized Binance market (check the ticker/quote currency — "
                f"this is a symbol-lookup problem, not an API-key problem; no credentials are required "
                f"for public OHLCV data)."
            )
        if not market.get("spot", False):
            raise ValueError(
                f"{symbol} exists on Binance but is not a SPOT market (likely futures/margin-only) — "
                f"this bot is spot-only and cannot fetch OHLCV for it."
            )
        if not market.get("active", True):
            raise ValueError(f"{symbol} is listed on Binance spot but is currently inactive/delisted.")

    # ------------------------------------------------------------------ #
    def _with_retry(self, fn, *args, **kwargs):
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except ccxt.RateLimitExceeded as e:
                last_err = e
                wait = self.retry_delay_seconds * attempt * 2
                logger.warning("Rate limited (attempt %d/%d), waiting %.1fs: %s", attempt, self.max_retries, wait, e)
                time.sleep(wait)
            except ccxt.NetworkError as e:
                last_err = e
                logger.warning("Network error (attempt %d/%d): %s", attempt, self.max_retries, e)
                time.sleep(self.retry_delay_seconds * attempt)
            except ccxt.AuthenticationError as e:
                # This wrapper never sets real credentials for public data —
                # if ccxt still raised an auth error, it's almost always a
                # symbol-resolution problem (see _ensure_valid_spot_symbol),
                # not an actual missing-API-key problem. Don't burn retries
                # on it; surface a clearer message immediately.
                logger.error("Auth-shaped error from ccxt (not a real credentials issue — retrying won't help): %s", e)
                raise ValueError(
                    f"Binance rejected this request in a way that looks like an auth error but almost "
                    f"certainly isn't (no real credentials are configured). Original ccxt error: {e}"
                ) from e
            except ccxt.ExchangeError as e:
                # Bad symbol, invalid params, etc. — retrying won't help.
                logger.error("Exchange error (not retrying): %s", e)
                raise
        logger.error("Exhausted retries: %s", last_err)
        raise last_err  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        try:
            self.exchange.fetch_time()
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Binance unreachable: %s", e)
            return False

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 260) -> pd.DataFrame:
        """Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
        Always drops a still-forming last candle (see _drop_unclosed_last_candle) —
        callers get only CLOSED candles, matching spec Part 25."""
        self._ensure_valid_spot_symbol(symbol)
        raw = self._with_retry(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return _drop_unclosed_last_candle(df, timeframe)

    def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str = "1h",
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        page_limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Paginates fetch_ohlcv to pull a full historical range — Binance
        caps a single call at 1000 candles, which is only ~41 days of 1H
        data, not enough for a meaningful backtest. Used by backtest.py.
        """
        self._ensure_valid_spot_symbol(symbol)
        timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000
        until_ms = until_ms or self.exchange.milliseconds()
        since = since_ms if since_ms is not None else until_ms - 180 * 86400 * 1000

        all_rows: list = []
        while since < until_ms:
            batch = self._with_retry(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, since=since, limit=page_limit)
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = batch[-1][0]
            next_since = last_ts + timeframe_ms
            if next_since <= since:
                break  # safety valve against a pathological non-advancing response
            since = next_since
            if len(batch) < page_limit:
                break  # exchange ran out of candles before we hit until_ms
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        df = df[df["timestamp"] <= until_ms]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        # If until_ms is close to real "now" (e.g. backtest.py fetching up
        # to the current moment), the last row can still be a forming
        # candle — drop it for the same reason fetch_ohlcv does.
        return _drop_unclosed_last_candle(df, timeframe)

    def fetch_ticker(self, symbol: str) -> dict:
        return self._with_retry(self.exchange.fetch_ticker, symbol)

    def fetch_all_tickers(self) -> dict:
        """
        Fetches all tickers in one call. Used once per scan cycle to build
        a symbol -> 24h quote volume map for the liquidity filter, instead
        of one ticker request per symbol.
        """
        return self._with_retry(self.exchange.fetch_tickers)

    def fetch_top_usdt_pairs(self, limit: int = 200, quote: str = "USDT") -> List[str]:
        """
        Returns the top `limit` USDT spot pairs on Binance ranked by 24h
        quote volume. Used as the scan fallback when watchlist.json is empty.
        """
        markets = self._with_retry(self.exchange.load_markets)
        tickers = self._with_retry(self.exchange.fetch_tickers)

        candidates = []
        for symbol, market in markets.items():
            if not market.get("spot", True):
                continue
            if market.get("quote") != quote:
                continue
            if not market.get("active", True):
                continue
            ticker = tickers.get(symbol)
            if not ticker:
                continue
            quote_volume = ticker.get("quoteVolume") or 0
            candidates.append((symbol, quote_volume))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in candidates[:limit]]

    def fetch_24h_change_pct(self, symbol: str) -> float:
        ticker = self.fetch_ticker(symbol)
        return float(ticker.get("percentage") or 0.0)
