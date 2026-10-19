"""动量策略：ROC 为正做多，转负或持有到期平仓（可选做空）。"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.core import indicators
from quant.strategies.base import BaseStrategy, StrategyContext
from quant.strategies.registry import register


@register
class MomentumStrategy(BaseStrategy):
    key: ClassVar[str] = "momentum"
    name: ClassVar[str] = "动量策略"
    display_name: ClassVar[str] = "动量策略 (Momentum)"
    description: ClassVar[str] = "ROC(lookback) 为正做多，转负或持有 hold_bars 根后平仓"

    class Params(BaseModel):
        lookback: int = Field(60, ge=5, le=500)
        hold_bars: int = Field(10, ge=1, le=200)
        threshold: float = Field(0.0, ge=0.0, le=1.0)
        only_long: bool = True

    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        close = bars["close"]
        r = indicators.roc(close, self.params.lookback)

        n = len(bars)
        target = np.zeros(n)
        reason = [""] * n
        pos = 0
        count = 0
        th = self.params.threshold
        for i in range(n):
            roc_i = r.iloc[i]
            if np.isnan(roc_i):
                target[i] = pos
                continue
            if pos == 0 and roc_i > th:
                pos = 1
                count = self.params.hold_bars
                reason[i] = f"动量转正 {roc_i:.2%}"
            elif pos == 1:
                count -= 1
                if roc_i < -th or count <= 0:
                    pos = 0 if self.params.only_long else -1
                    reason[i] = "动量转负" if roc_i < -th else "持有到期"
                    if not self.params.only_long:
                        reason[i] = f"动量转负做空 {roc_i:.2%}"
            elif pos == -1:
                count -= 1
                if roc_i > th or count <= 0:
                    pos = 0
                    reason[i] = "动量转正平空"
            target[i] = pos
        return pd.DataFrame({"target_weight": target, "reason": reason, "confidence": 1.0},
                            index=bars.index)
