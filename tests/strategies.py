"""测试共享：合成行情、可注册的测试策略。"""
from __future__ import annotations

from typing import ClassVar

import pandas as pd
from pydantic import BaseModel

from quant.strategies.base import BaseStrategy
from quant.strategies.registry import register


def make_bars(prices: list[float], start: str = "2024-01-01", freq: str = "D") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [10000.0] * len(prices),
    }, index=idx)


@register
class StepStrategy(BaseStrategy):
    """测试策略：前 hold_bars 根做多，之后平仓（确定性，用于引擎/模拟盘验证）。"""

    key: ClassVar[str] = "test_step"
    name: ClassVar[str] = "测试阶梯策略"
    description: ClassVar[str] = "前 N 根 K 线做多，之后空仓"

    class Params(BaseModel):
        hold_bars: int = 3
        long_only: bool = True

    def generate_signals(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        n = len(bars)
        target = pd.Series(0.0, index=bars.index)
        target.iloc[: min(self.params.hold_bars, n)] = 1.0
        reason = pd.Series("", index=bars.index)
        reason.iloc[0] = "测试入场"
        return pd.DataFrame({"target_weight": target, "reason": reason, "confidence": 1.0})


@register
class HalfStepStrategy(BaseStrategy):
    """测试策略：前 hold_bars 根半仓（0.5），之后空仓（capital_split 多策略用）。"""

    key: ClassVar[str] = "test_half"
    name: ClassVar[str] = "测试半仓阶梯"
    description: ClassVar[str] = "前 N 根 K 线半仓，之后空仓"

    class Params(BaseModel):
        hold_bars: int = 3

    def generate_signals(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        n = len(bars)
        target = pd.Series(0.5, index=bars.index)
        target.iloc[min(self.params.hold_bars, n):] = 0.0
        reason = pd.Series("", index=bars.index)
        reason.iloc[0] = "测试半仓入场"
        return pd.DataFrame({"target_weight": target, "reason": reason, "confidence": 1.0})
