"""MACD 策略：金叉做多，死叉平仓（可选做空）。"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.core import indicators
from quant.strategies.base import BaseStrategy, StrategyContext
from quant.strategies.registry import register


@register
class MacdStrategy(BaseStrategy):
    key: ClassVar[str] = "macd"
    name: ClassVar[str] = "MACD 策略"
    display_name: ClassVar[str] = "MACD 策略 (Moving Average Convergence Divergence)"
    description: ClassVar[str] = "MACD 金叉做多，死叉平仓（可选做空）"

    class Params(BaseModel):
        fast: int = Field(12, ge=2, le=100)
        slow: int = Field(26, ge=3, le=200)
        signal: int = Field(9, ge=2, le=50)
        only_long: bool = True

    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        close = bars["close"]
        m = indicators.macd(close, self.params.fast, self.params.slow, self.params.signal)
        cross_up = (m["macd"] > m["signal"]) & (m["macd"].shift(1) <= m["signal"].shift(1))
        cross_dn = (m["macd"] < m["signal"]) & (m["macd"].shift(1) >= m["signal"].shift(1))

        n = len(bars)
        target = np.zeros(n)
        reason = [""] * n
        pos = 0
        for i in range(n):
            if pos == 0 and bool(cross_up.iloc[i]):
                pos = 1
                reason[i] = f"MACD金叉 {m['macd'].iloc[i]:.3f}>{m['signal'].iloc[i]:.3f}"
            elif pos == 1 and bool(cross_dn.iloc[i]):
                pos = 0 if self.params.only_long else -1
                reason[i] = "MACD死叉平仓" if self.params.only_long else f"MACD死叉做空 {m['macd'].iloc[i]:.3f}"
            elif pos == -1 and bool(cross_up.iloc[i]):
                pos = 0
                reason[i] = "MACD金叉平空"
            target[i] = pos
        return pd.DataFrame({"target_weight": target, "reason": reason, "confidence": 1.0},
                            index=bars.index)
