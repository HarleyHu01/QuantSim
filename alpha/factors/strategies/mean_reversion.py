"""均值回归策略：Z-Score 低于 -entry_z 买入，回到 -exit_z 平仓（可选做空）。"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.core import indicators
from quant.strategies.base import BaseStrategy, StrategyContext
from quant.strategies.registry import register


@register
class MeanReversionStrategy(BaseStrategy):
    key: ClassVar[str] = "mean_reversion"
    name: ClassVar[str] = "均值回归策略"
    display_name: ClassVar[str] = "均值回归策略 (Mean Reversion)"
    description: ClassVar[str] = "Z-Score 低于 -entry_z 买入，回升到 -exit_z 平仓"

    class Params(BaseModel):
        lookback: int = Field(20, ge=5, le=200)
        entry_z: float = Field(2.0, ge=0.5, le=5.0)
        exit_z: float = Field(0.0, ge=-5.0, le=5.0)
        only_long: bool = True

    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        close = bars["close"]
        z = indicators.zscore(close, self.params.lookback)

        n = len(bars)
        target = np.zeros(n)
        reason = [""] * n
        pos = 0
        for i in range(n):
            z_i = z.iloc[i]
            if np.isnan(z_i):
                target[i] = pos
                continue
            if pos == 0 and z_i < -self.params.entry_z:
                pos = 1
                reason[i] = f"Z 低于 {self.params.entry_z:.1f} 买入 ({z_i:.2f})"
            elif pos == 1 and z_i > -self.params.exit_z:
                pos = 0 if self.params.only_long else -1
                reason[i] = "回归均值平仓" if self.params.only_long else f"突破上沿做空 ({z_i:.2f})"
            elif pos == -1 and z_i < self.params.exit_z:
                pos = 0
                reason[i] = "回归均值平空"
            target[i] = pos
        return pd.DataFrame({"target_weight": target, "reason": reason, "confidence": 1.0},
                            index=bars.index)
