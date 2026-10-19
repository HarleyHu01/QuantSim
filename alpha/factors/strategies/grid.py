"""网格交易策略：以首根收盘价为锚点，价格每下跌一个网格加仓，每上涨一个网格减仓。

边界行为（v1 语义）：
- 锚点 = 首根收盘价（固定，不随价格移动；v2 计划：整轮回补后重新锚定）
- 间距 = grid_pct × 锚点（默认 2%）
- 跳多档：目标 = clip(base_position + step×(1/levels), 0或−1, 1)，一次跨 N 档直接到位
- 越界：价格远高于锚点 → 饱和到 0（清仓）；远低于 → 饱和到 1（满仓）

模型（位置式网格）：
    step = (anchor − close) / anchor / grid_pct        # 低于锚点的网格数（可为负）
    target = clip(base_position + step × (1/levels), 0, 1)（only_long）或 (-1, 1)
价格在网格间波动时目标仓位阶梯式变化，由引擎按目标变化撮合。
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.strategies.base import BaseStrategy, StrategyContext
from quant.strategies.registry import register


@register
class GridStrategy(BaseStrategy):
    key: ClassVar[str] = "grid"
    name: ClassVar[str] = "网格策略"
    display_name: ClassVar[str] = "网格策略 (Grid Trading)"
    description: ClassVar[str] = "价格每跌一格加仓、每涨一格减仓（锚点 = 首根收盘价）"

    class Params(BaseModel):
        grid_pct: float = Field(0.02, ge=0.001, le=0.5)  # 网格间距（2%）
        levels: int = Field(10, ge=2, le=100)  # 网格层数（每层仓位 = 1/levels）
        base_position: float = Field(0.5, ge=0.0, le=1.0)  # 锚点处的基础仓位
        only_long: bool = True

    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        close = bars["close"]
        anchor = float(close.iloc[0]) if len(close) else 1.0
        per_grid = 1.0 / max(self.params.levels, 1)

        n = len(bars)
        target = np.zeros(n)
        reason = [""] * n
        last_steps = None
        for i in range(n):
            steps = (anchor - close.iloc[i]) / anchor / self.params.grid_pct
            t = self.params.base_position + steps * per_grid
            t = float(np.clip(t, 0.0 if self.params.only_long else -1.0, 1.0))
            target[i] = t
            if last_steps is None or int(steps) != int(last_steps):
                reason[i] = f"网格 {int(steps):+d} 格 @ {close.iloc[i]:.2f}"
            last_steps = steps
        return pd.DataFrame({"target_weight": target, "reason": reason, "confidence": 1.0},
                            index=bars.index)
