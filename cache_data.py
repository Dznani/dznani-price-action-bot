import json
import os
import pandas as pd
from exchange import BinanceExchange
import backtest
from database import DEFAULT_SETTINGS

CANDIDATES = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT", 
    "DOT/USDT", "UNI/USDT", "NEAR/USDT", "SUI/USDT", "GRT/USDT", 
    "VET/USDT", "BCH/USDT", "OP/USDT", "ARB/USDT", "ALGO/USDT", "INJ/USDT"
]

CACHE_DIR = "ohlcv_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_and_cache(days=180):
    ex = BinanceExchange()
    until_ms = ex.exchange.milliseconds()
    since_ms_1h = until_ms - days * 86400 * 1000
    since_ms_4h = until_ms - (days + 30) * 86400 * 1000

    data = {}
    for sym in CANDIDATES:
        clean_sym = sym.replace("/", "_")
        f_1h = os.path.join(CACHE_DIR, f"{clean_sym}_1h.csv")
        f_4h = os.path.join(CACHE_DIR, f"{clean_sym}_4h.csv")
        
        if os.path.exists(f_1h) and os.path.exists(f_4h):
            df_1h = pd.read_csv(f_1h, parse_dates=["timestamp"])
            df_4h = pd.read_csv(f_4h, parse_dates=["timestamp"])
        else:
            print(f"Fetching {sym}...")
            df_1h = ex.fetch_ohlcv_range(sym, "1h", since_ms_1h, until_ms)
            df_4h = ex.fetch_ohlcv_range(sym, "4h", since_ms_4h, until_ms)
            df_1h.to_csv(f_1h, index=False)
            df_4h.to_csv(f_4h, index=False)
        data[sym] = (df_4h, df_1h)
    return data

if __name__ == "__main__":
    data = fetch_and_cache(180)
    print("Fetched and cached", len(data), "symbols.")
