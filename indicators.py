"""
indicators.py — technical indicator functions for Dznani Signals Bot.

All functions take/return pandas Series or DataFrames. Written against
pandas 3.x (no deprecated .append / fillna(method=) / iteritems usage).

Expected OHLCV DataFrame columns: timestamp, open, high, low, close, volume
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# RSI
# --------------------------------------------------------------------------- #
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Classic Wilder RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral during warm-up / zero-loss stretches
    return rsi


# --------------------------------------------------------------------------- #
# ATR
# --------------------------------------------------------------------------- #
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


# --------------------------------------------------------------------------- #
# MFI
# --------------------------------------------------------------------------- #
def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    raw_money_flow = typical_price * df["volume"]

    direction = typical_price.diff()
    positive_flow = raw_money_flow.where(direction > 0, 0.0)
    negative_flow = raw_money_flow.where(direction < 0, 0.0)

    pos_sum = positive_flow.rolling(period, min_periods=period).sum()
    neg_sum = negative_flow.rolling(period, min_periods=period).sum()

    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    mfi = mfi.fillna(50)
    return mfi


# --------------------------------------------------------------------------- #
# ADX
# --------------------------------------------------------------------------- #
def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    atr = calculate_atr(df, period)
    atr_safe = atr.replace(0, np.nan)

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_safe)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_safe)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx.fillna(0)


# --------------------------------------------------------------------------- #
# Squeeze Momentum (Bollinger Bands vs Keltner Channels)
# --------------------------------------------------------------------------- #
def calculate_sqz_mom(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (squeeze_on, momentum):
      squeeze_on : bool Series — True while Bollinger Bands sit inside the
                   Keltner Channel (volatility compressed / "ON").
      momentum   : float Series — linear-regression style momentum value,
                   positive/rising favors bullish continuation.
    """
    close, high, low = df["close"], df["high"], df["low"]

    basis = close.rolling(bb_period).mean()
    dev = bb_mult * close.rolling(bb_period).std(ddof=0)
    bb_upper = basis + dev
    bb_lower = basis - dev

    kc_basis = close.rolling(kc_period).mean()
    atr = calculate_atr(df, kc_period)
    kc_upper = kc_basis + kc_mult * atr
    kc_lower = kc_basis - kc_mult * atr

    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    highest_high = high.rolling(kc_period).max()
    lowest_low = low.rolling(kc_period).min()
    sma_close = close.rolling(kc_period).mean()
    avg_range = (highest_high + lowest_low) / 2
    donchian_mid = (avg_range + sma_close) / 2
    source = close - donchian_mid

    def _linreg_last(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        return slope * (len(y) - 1) + intercept

    momentum = source.rolling(kc_period).apply(_linreg_last, raw=True)
    return squeeze_on.fillna(False), momentum.fillna(0.0)


# --------------------------------------------------------------------------- #
# EMA Stack (4EME)
# --------------------------------------------------------------------------- #
def check_ema_stack(df: pd.DataFrame) -> str:
    """Returns 'Bullish', 'Bearish', or 'Neutral' based on EMA(7,25,70,200) order."""
    close = df["close"]
    if len(close) < 200:
        return "Neutral"

    ema7 = close.ewm(span=7, adjust=False).mean().iloc[-1]
    ema25 = close.ewm(span=25, adjust=False).mean().iloc[-1]
    ema70 = close.ewm(span=70, adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]

    if ema7 > ema25 > ema70 > ema200:
        return "Bullish"
    if ema7 < ema25 < ema70 < ema200:
        return "Bearish"
    return "Neutral"


def ema200_trend_pass(df: pd.DataFrame, direction: str, breakout_volume_mult: float = 3.0) -> bool:
    """
    Trend filter: blocks BUY below the 200 EMA and SELL above it, unless
    volume on the current candle exceeds `breakout_volume_mult` x the
    20-period average (a strong-enough breakout is allowed through).
    """
    close = df["close"]
    if len(close) < 200:
        return True  # not enough history to filter — don't block

    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
    last_close = close.iloc[-1]
    avg_vol20 = df["volume"].rolling(20).mean().iloc[-1]
    last_vol = df["volume"].iloc[-1]
    breakout = avg_vol20 > 0 and last_vol > breakout_volume_mult * avg_vol20

    if direction == "BUY":
        return bool(last_close > ema200 or breakout)
    if direction == "SELL":
        return bool(last_close < ema200 or breakout)
    return True


# --------------------------------------------------------------------------- #
# RSI Divergence
# --------------------------------------------------------------------------- #
def detect_divergence(df: pd.DataFrame, rsi_series: pd.Series, lookback: int = 6) -> Optional[str]:
    """
    Looks for classic divergence between price and RSI over the last
    `lookback` candles:
      Bullish: price makes a lower low, RSI makes a higher low.
      Bearish: price makes a higher high, RSI makes a lower high.
    Returns 'Bullish', 'Bearish', or None.
    """
    if len(df) < lookback:
        return None

    window_close = df["close"].iloc[-lookback:]
    window_rsi = rsi_series.iloc[-lookback:]

    price_low_idx = window_close.idxmin()
    price_high_idx = window_close.idxmax()

    # Compare the most recent candle against the window's extreme
    last_close = window_close.iloc[-1]
    last_rsi = window_rsi.iloc[-1]

    prior_low = window_close.drop(index=window_close.index[-1]).min()
    prior_low_rsi = window_rsi.loc[window_close.drop(index=window_close.index[-1]).idxmin()]

    prior_high = window_close.drop(index=window_close.index[-1]).max()
    prior_high_rsi = window_rsi.loc[window_close.drop(index=window_close.index[-1]).idxmax()]

    if last_close < prior_low and last_rsi > prior_low_rsi:
        return "Bullish"
    if last_close > prior_high and last_rsi < prior_high_rsi:
        return "Bearish"
    return None


# --------------------------------------------------------------------------- #
# Price Action Patterns
# --------------------------------------------------------------------------- #
def detect_price_action(df: pd.DataFrame) -> Optional[str]:
    """
    Detects, on the current (last) candle:
      - Bullish Engulfing / Bearish Engulfing (needs previous candle)
      - Hammer (lower wick > 2x body)
      - Shooting Star (upper wick > 2x body)
    Returns the pattern name or None.
    """
    if len(df) < 2:
        return None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    body = abs(curr["close"] - curr["open"])
    upper_wick = curr["high"] - max(curr["close"], curr["open"])
    lower_wick = min(curr["close"], curr["open"]) - curr["low"]

    prev_body_low = min(prev["open"], prev["close"])
    prev_body_high = max(prev["open"], prev["close"])

    # Engulfing patterns
    curr_bullish = curr["close"] > curr["open"]
    curr_bearish = curr["close"] < curr["open"]
    prev_bearish = prev["close"] < prev["open"]
    prev_bullish = prev["close"] > prev["open"]

    if curr_bullish and prev_bearish and curr["open"] <= prev_body_low and curr["close"] >= prev_body_high:
        return "Bullish Engulfing"
    if curr_bearish and prev_bullish and curr["open"] >= prev_body_high and curr["close"] <= prev_body_low:
        return "Bearish Engulfing"

    # Hammer / Shooting Star (body must be non-zero to avoid div-by-zero)
    if body > 0:
        if lower_wick > 2 * body and upper_wick < body:
            return "Hammer"
        if upper_wick > 2 * body and lower_wick < body:
            return "Shooting Star"

    return None


# --------------------------------------------------------------------------- #
# Momentum (simple rule-5 check)
# --------------------------------------------------------------------------- #
def calculate_momentum_state(df: pd.DataFrame, period: int = 10) -> Optional[str]:
    """Returns 'Positive' if close[last] > close[last-period], 'Negative' if <, else None."""
    if len(df) <= period:
        return None
    last_close = df["close"].iloc[-1]
    past_close = df["close"].iloc[-1 - period]
    if last_close > past_close:
        return "Positive"
    if last_close < past_close:
        return "Negative"
    return None
