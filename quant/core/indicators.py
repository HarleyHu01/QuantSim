"""指标库：策略与 LLM 声明式 DSL 共用的白名单指标。

所有函数输入/输出均为 pandas Series/DataFrame（index 对齐），
LLM 策略生成时的指标白名单与此保持一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD：列 = [macd, signal, hist]。"""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd_line, "signal": signal_line,
                         "hist": macd_line - signal_line})


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI（Wilder 平滑）。无下跌（avg_loss=0）时为 100。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0)


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """布林带：列 = [up, mid, low]。"""
    mid = sma(close, period)
    std = close.rolling(period).std()
    return pd.DataFrame({"up": mid + num_std * std, "mid": mid, "low": mid - num_std * std})


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR（Wilder 平滑），输入需含 high/low/close。"""
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def donchian(close: pd.Series, period: int) -> pd.DataFrame:
    """唐奇安通道（shift(1)：用前 period 根，避免用当前 bar 突破自身）。列 = [high, low]。"""
    return pd.DataFrame({
        "high": close.rolling(period).max().shift(1),
        "low": close.rolling(period).min().shift(1),
    })


def roc(close: pd.Series, window: int) -> pd.Series:
    """变动率（动量）。"""
    return close.pct_change(window)


def zscore(close: pd.Series, window: int) -> pd.Series:
    """Z-Score（均值回归）。"""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / std.replace(0, np.nan)
