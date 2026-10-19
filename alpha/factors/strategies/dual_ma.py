"""双均线策略（第 1 个内置策略，提前实现用于引擎验证）。

金叉（快线上穿慢线）做多；死叉（快线下穿慢线）平多，可选反向做空。
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.strategies.base import BaseStrategy, StrategyContext
from quant.strategies.registry import register


@register
class DualMaStrategy(BaseStrategy):
    key: ClassVar[str] = "dual_ma"
    name: ClassVar[str] = "双均线策略"
    display_name: ClassVar[str] = "双均线策略 (Dual Moving Average)"
    description: ClassVar[str] = "快线上穿慢线做多，下穿平仓（可选做空）"

    class Params(BaseModel):
        fast: int = Field(20, ge=2, le=500)
        slow: int = Field(60, ge=3, le=1000)
        only_long: bool = False

    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        close = bars["close"]
        fast = close.rolling(self.params.fast).mean()
        slow = close.rolling(self.params.slow).mean()

        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        target = pd.Series(np.nan, index=bars.index)
        target[cross_up] = 1.0
        target[cross_dn] = 0.0 if self.params.only_long else -1.0
        target = target.ffill().fillna(0.0)

        reason = pd.Series("", index=bars.index, dtype=object)
        reason[cross_up] = "金叉 " + fast[cross_up].round(2).astype(str) + ">" + slow[cross_up].round(2).astype(str)
        reason[cross_dn] = "死叉平仓" if self.params.only_long else "死叉做空"

        return pd.DataFrame({
            "target_weight": target,
            "reason": reason,
            "confidence": 1.0,
        })
