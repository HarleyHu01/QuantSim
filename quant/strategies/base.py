"""策略统一接口。

策略只输出目标仓位（-1..1），不直接下单；由组合器合成、引擎执行。
输出约定：generate_signals 返回 DataFrame，index=bars.index，
列 = [target_weight(-1..1), reason(str), confidence(0~1，默认 1.0)]。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd
from pydantic import BaseModel


class StrategyContext:
    """策略运行上下文（只读）。"""

    def __init__(self, symbol: str, timeframe: str, history: pd.DataFrame,
                 extra: dict | None = None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.history = history  # 截至当前 bar 的完整行情
        self.extra = extra or {}  # 引擎注入的附加信息（如 initial_capital）


class BaseStrategy(ABC):
    """策略基类。子类需定义：
    - key / name / description（类属性，注册表标识）
    - class Params(BaseModel)（参数模型，每个参数必须有默认值）
    - generate_signals(bars, ctx) -> pd.DataFrame
    """

    key: ClassVar[str] = "base"
    name: ClassVar[str] = "Base"
    description: ClassVar[str] = ""

    def __init__(self, params: dict | None = None):
        # params: 覆盖默认值的字典（UI 调参/LLM 生成）
        self.params: BaseModel = self.build_params(params)

    @classmethod
    def build_params(cls, params: dict | None) -> BaseModel:
        return cls.Params(**(params or {}))

    @classmethod
    def params_schema(cls) -> dict:
        """Pydantic schema（UI 动态渲染参数表单用）。"""
        return cls.Params.model_json_schema()

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        """返回目标仓位 DataFrame：列 = [target_weight, reason, confidence]。"""

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> tuple[float, str, float]:
        """增量版（模拟盘用）：默认取 generate_signals(ctx.history) 最新一行。

        返回 (target_weight, reason, confidence)。高级策略可覆写做增量指标缓存。
        """
        df = self.generate_signals(ctx.history, ctx)
        if df is None or df.empty:
            return 0.0, "无信号", 1.0
        row = df.iloc[-1]
        return (
            float(row["target_weight"]),
            str(row.get("reason", "")),
            float(row.get("confidence", 1.0)),
        )
