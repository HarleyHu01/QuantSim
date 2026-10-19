"""RSI 超买超卖策略：脱离超卖区买入，脱离超买区卖出（可选做空）。"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.core import indicators
from quant.strategies.base import BaseStrategy, StrategyContext
from quant.strategies.registry import register


@register
class RsiStrategy(BaseStrategy):
    key: ClassVar[str] = "rsi"
    name: ClassVar[str] = "RSI 策略"
    display_name: ClassVar[str] = "RSI 策略 (Relative Strength Index)"
    description: ClassVar[str] = "RSI 脱离超卖区买入，脱离超买区卖出"

    class Params(BaseModel):
        period: int = Field(14, ge=2, le=100)
        oversold: float = Field(30.0, ge=1.0, le=50.0)
        overbought: float = Field(70.0, ge=50.0, le=99.0)
        only_long: bool = True

    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        close = bars["close"]
        r = indicators.rsi(close, self.params.period)
        # 脱离超卖区（上穿 oversold）→ 买入；脱离超买区（下穿 overbought）→ 卖出/做空
        cross_up_oversold = (r > self.params.oversold) & (r.shift(1) <= self.params.oversold)
        cross_dn_overbought = (r < self.params.overbought) & (r.shift(1) >= self.params.overbought)

        n = len(bars)
        target = np.zeros(n)
        reason = [""] * n
        pos = 0
        for i in range(n):
            if pos == 0 and bool(cross_up_oversold.iloc[i]):
                pos = 1
                reason[i] = f"RSI 上穿超卖 {r.iloc[i]:.1f}"
            elif pos == 1 and bool(cross_dn_overbought.iloc[i]):
                pos = 0 if self.params.only_long else -1
                reason[i] = "RSI 脱离超买" if self.params.only_long else f"RSI 下穿超买做空 {r.iloc[i]:.1f}"
            elif pos == -1 and bool(cross_up_oversold.iloc[i]):
                pos = 0
                reason[i] = "RSI 上穿超卖平空"
            target[i] = pos
        return pd.DataFrame({"target_weight": target, "reason": reason, "confidence": 1.0},
                            index=bars.index)
