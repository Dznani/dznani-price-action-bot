"""
tests/test_exchange_symbol_validation.py

Support-ticket regression test: "binance requires 'apiKey' credential" for
THE/USDT while scanning with public-only access (no real API keys
configured). Confirmed by code inspection that BinanceExchange never sets
real credentials or any signed-only option for public OHLCV data — the
error was ccxt's own confusing failure mode when it can't resolve a
symbol. These tests mock ccxt entirely (no network) to verify the new
_ensure_valid_spot_symbol() guard produces a clear, correct error instead.
"""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exchange import BinanceExchange  # noqa: E402


def _make_exchange_with_markets(markets: dict) -> BinanceExchange:
    ex = BinanceExchange(api_key=None, api_secret=None)
    ex.exchange.load_markets = MagicMock(return_value=markets)
    return ex


def test_no_credentials_ever_configured_for_public_data():
    """The wrapper must never pass real credentials for plain OHLCV
    scanning — confirms the "requires apiKey" error can't be coming from
    this code intentionally requiring auth."""
    ex = BinanceExchange(api_key=None, api_secret=None)
    assert ex.exchange.apiKey == ""
    assert ex.exchange.secret == ""


def test_unknown_symbol_raises_clear_not_authentication_error():
    ex = _make_exchange_with_markets({"BTC/USDT": {"spot": True, "active": True}})
    try:
        ex._ensure_valid_spot_symbol("THE/USDT")
        assert False, "expected a ValueError for an unrecognized symbol"
    except ValueError as e:
        msg = str(e)
        assert "not a recognized Binance market" in msg
        assert "not an API-key problem" in msg


def test_futures_only_symbol_raises_clear_spot_only_error():
    ex = _make_exchange_with_markets({"XYZ/USDT": {"spot": False, "active": True}})
    try:
        ex._ensure_valid_spot_symbol("XYZ/USDT")
        assert False, "expected a ValueError for a non-spot market"
    except ValueError as e:
        assert "not a SPOT market" in str(e)


def test_delisted_symbol_raises_clear_inactive_error():
    ex = _make_exchange_with_markets({"OLD/USDT": {"spot": True, "active": False}})
    try:
        ex._ensure_valid_spot_symbol("OLD/USDT")
        assert False, "expected a ValueError for an inactive market"
    except ValueError as e:
        assert "inactive/delisted" in str(e)


def test_valid_active_spot_symbol_passes_without_error():
    ex = _make_exchange_with_markets({"BTC/USDT": {"spot": True, "active": True}})
    ex._ensure_valid_spot_symbol("BTC/USDT")  # must not raise


def test_markets_cache_only_loaded_once():
    ex = _make_exchange_with_markets({"BTC/USDT": {"spot": True, "active": True}})
    ex._ensure_valid_spot_symbol("BTC/USDT")
    ex._ensure_valid_spot_symbol("BTC/USDT")
    assert ex.exchange.load_markets.call_count == 1, "load_markets should be cached, not re-fetched every call"
