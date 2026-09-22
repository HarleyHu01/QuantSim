"""海龟交易策略（经典系统 1 完整版）。

规则：
- 入场：收盘价突破 entry_n 日唐奇安通道高点 → 做多 1 单位；跌破低点 → 做空 1 单位
- 加仓：每朝有利方向移动 add_unit_step × N（0.5N）加 1 单位，上限 max_units（4 单位）
- 止损：价格反向移动 stop_atr_mult × N（2N）→ 全部离场
- 离场：价格反向突破 exit_n 日通道 → 全部离场
- 再入场：离场后若突破仍有效，后续 bar 重新入场（"同 bar 再入场"为简化）
- 仓位：每单位目标权重 = risk_per_trade / (stop_atr_mult × N_frac)，随波动率缩放；
  经典参数（1% 风险、2N 止损、N≈2%）下 4 单位 ≈ 100% 仓位。
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.core import indicators
from quant.strategies.base import BaseStrategy, StrategyContext
from quant.strategies.registry import register


@register
class TurtleStrategy(BaseStrategy):
    key: ClassVar[str] = "turtle"
    name: ClassVar[str] = "海龟策略"
    display_name: ClassVar[str] = "海龟策略 (Turtle Trading)"
    description: ClassVar[str] = "唐奇安通道突破入场，0.5N 加仓、2N 止损、反向通道离场（经典系统1）"

    class Params(BaseModel):
        entry_n: int = Field(20, ge=5, le=200)  # 唐奇安入场通道周期
        exit_n: int = Field(10, ge=2, le=100)  # 唐奇安离场通道周期
        atr_period: int = Field(14, ge=2, le=100)  # N 值（ATR）周期
        risk_per_trade: float = Field(0.01, ge=0.001, le=0.2)  # 单笔风险（账户 %）
        stop_atr_mult: float = Field(2.0, ge=0.5, le=10.0)  # 止损距离（×N）
        add_unit_step: float = Field(0.5, ge=0.1, le=3.0)  # 加仓间隔（×N）
        max_units: int = Field(4, ge=1, le=20)  # 最大持仓单位
        only_long: bool = False

    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        close = bars["close"]
        dc_in = indicators.donchian(close, self.params.entry_n)
        dc_out = indicators.donchian(close, self.params.exit_n)
        n_frac = indicators.atr(bars, self.params.atr_period) / close  # 归一化 N

        n = len(bars)
        target = np.zeros(n)
        reason = [""] * n
        pos_units = 0  # 持仓单位（带符号）
        entry_price: float | None = None
        entry_n_frac: float | None = None  # 入场时锁定的 N（止损/加仓用固定 N）

        p = self.params
        for i in range(n):
            c = float(close.iloc[i])
            N = float(n_frac.iloc[i]) if not np.isnan(n_frac.iloc[i]) else 0.0
            unit_weight = p.risk_per_trade / (p.stop_atr_mult * N) if N > 0 else 0.0

            if pos_units == 0:
                if not np.isnan(dc_in["high"].iloc[i]) and c > dc_in["high"].iloc[i]:
                    pos_units = 1
                    entry_price = c
                    entry_n_frac = N
                    reason[i] = f"突破 {p.entry_n} 日高点入场 {dc_in['high'].iloc[i]:.2f}"
                elif not p.only_long and not np.isnan(dc_in["low"].iloc[i]) \
                        and c < dc_in["low"].iloc[i]:
                    pos_units = -1
                    entry_price = c
                    entry_n_frac = N
                    reason[i] = f"跌破 {p.entry_n} 日低点做空 {dc_in['low'].iloc[i]:.2f}"
            else:
                entry_atr = entry_n_frac * entry_price if entry_n_frac else 0.0
                stop_dist = p.stop_atr_mult * entry_atr
                step_dist = p.add_unit_step * entry_atr
                if pos_units > 0:
                    if c <= entry_price - stop_dist:
                        pos_units = 0
                        reason[i] = f"2N 止损离场 ({p.stop_atr_mult:.0f}N)"
                    elif c < dc_out["low"].iloc[i]:
                        pos_units = 0
                        reason[i] = f"跌破 {p.exit_n} 日通道离场 {dc_out['low'].iloc[i]:.2f}"
                    elif step_dist > 0 and c > entry_price:  # 仅朝有利方向加仓
                        units = 1 + int((c - entry_price) / step_dist)
                        pos_units = min(units, p.max_units)
                        if units > 1 and pos_units > 1:
                            reason[i] = f"0.5N 加仓至 {pos_units} 单位"
                else:  # 空头
                    if c >= entry_price + stop_dist:
                        pos_units = 0
                        reason[i] = f"2N 止损离场 ({p.stop_atr_mult:.0f}N)"
                    elif c > dc_out["high"].iloc[i]:
                        pos_units = 0
                        reason[i] = f"突破 {p.exit_n} 日通道平空 {dc_out['high'].iloc[i]:.2f}"
                    elif step_dist > 0 and c < entry_price:  # 仅朝有利方向加仓
                        units = 1 + int((entry_price - c) / step_dist)
                        pos_units = max(-min(units, p.max_units), -p.max_units)
                        if units > 1:
                            reason[i] = f"0.5N 加仓至 {abs(pos_units)} 单位"

            target[i] = float(np.clip(pos_units * unit_weight, -1.0, 1.0))
        return pd.DataFrame({"target_weight": target, "reason": reason, "confidence": 1.0},
                            index=bars.index)
