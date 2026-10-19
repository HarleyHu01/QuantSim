"""布林带策略：跌破下轨买入，回到中轨/突破上轨卖出（可选做空）。"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.core import indicators
from quant.strategies.base import BaseStrategy, StrategyContext
from quant.strategies.registry import register


@register
class BollingerStrategy(BaseStrategy):
    key: ClassVar[str] = "bollinger"
    name: ClassVar[str] = "布林带策略"
    display_name: ClassVar[str] = "布林带策略 (Bollinger Bands)"
    description: ClassVar[str] = "跌破下轨买入，回到中轨或突破上轨卖出"

    class Params(BaseModel):
        period: int = Field(20, ge=2, le=200)
        num_std: float = Field(2.0, ge=0.5, le=5.0)
        exit_at_mid: bool = True
        only_long: bool = True

    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        close = bars["close"]
        b = indicators.bollinger(close, self.params.period, self.params.num_std)
        cross_below_low = (close < b["low"]) & (close.shift(1) >= b["low"].shift(1))
        cross_above_high = (close > b["up"]) & (close.shift(1) <= b["up"].shift(1))
        exit_long_mask = (close > b["mid"]) if self.params.exit_at_mid else cross_above_high

        n = len(bars)
        target = np.zeros(n)
        reason = [""] * n
        pos = 0
        exit_short_mask = (close < b["mid"]) if self.params.exit_at_mid else cross_below_low
        for i in range(n):
            if pos == 0 and bool(cross_below_low.iloc[i]):
                pos = 1
                reason[i] = f"跌破下轨 {b['low'].iloc[i]:.2f}"
            elif pos == 0 and not self.params.only_long and bool(cross_above_high.iloc[i]):
                pos = -1
                reason[i] = f"突破上轨做空 {b['up'].iloc[i]:.2f}"
            elif pos == 1 and bool(exit_long_mask.iloc[i]):
                if not self.params.only_long and bool(cross_above_high.iloc[i]):
                    pos = -1
                    reason[i] = f"突破上轨做空 {b['up'].iloc[i]:.2f}"
                else:
                    pos = 0
                    reason[i] = "回到中轨" if self.params.exit_at_mid else "突破上轨平仓"
            elif pos == -1 and bool(exit_short_mask.iloc[i]):
                pos = 0
                reason[i] = "回到中轨平空"
            target[i] = pos
        return pd.DataFrame({"target_weight": target, "reason": reason, "confidence": 1.0},
                            index=bars.index)
