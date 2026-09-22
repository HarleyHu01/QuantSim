"""LLM / 手动表单共用的声明式策略 DSL（架构文档 §5）。

- 手动创建表单与未来的 LLM 生成共用同一套 schema 与校验链
- INDICATOR_WHITELIST / OPERATORS / POSITION_MODES 是 UI 表单的下拉白名单
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ---- 指标白名单（name → 中文标签 / 参数 / 输出命名规则） ----
# outputs 中 {as} 为用户指定的输出名前缀；多输出指标按后缀区分（如 boll_up/boll_mid/boll_low）
INDICATOR_WHITELIST: dict[str, dict] = {
    "sma": {
        "label": "简单移动平均线 (SMA)",
        "params": [{"name": "window", "type": "integer", "default": 20, "min": 2, "max": 500}],
        "outputs": ["{as}"],
    },
    "ema": {
        "label": "指数移动平均线 (EMA)",
        "params": [{"name": "window", "type": "integer", "default": 20, "min": 2, "max": 500}],
        "outputs": ["{as}"],
    },
    "macd": {
        "label": "MACD",
        "params": [
            {"name": "fast", "type": "integer", "default": 12, "min": 2, "max": 100},
            {"name": "slow", "type": "integer", "default": 26, "min": 3, "max": 200},
            {"name": "signal", "type": "integer", "default": 9, "min": 2, "max": 50},
        ],
        "outputs": ["{as}", "{as}_signal", "{as}_hist"],
    },
    "rsi": {
        "label": "相对强弱指标 (RSI)",
        "params": [{"name": "period", "type": "integer", "default": 14, "min": 2, "max": 100}],
        "outputs": ["{as}"],
    },
    "boll": {
        "label": "布林带 (BOLL)",
        "params": [
            {"name": "period", "type": "integer", "default": 20, "min": 2, "max": 200},
            {"name": "num_std", "type": "number", "default": 2.0, "min": 0.5, "max": 5.0},
        ],
        "outputs": ["{as}_up", "{as}_mid", "{as}_low"],
    },
    "atr": {
        "label": "平均真实波幅 (ATR)",
        "params": [{"name": "period", "type": "integer", "default": 14, "min": 2, "max": 100}],
        "outputs": ["{as}"],
    },
    "roc": {
        "label": "变动率 (ROC)",
        "params": [{"name": "window", "type": "integer", "default": 10, "min": 2, "max": 500}],
        "outputs": ["{as}"],
    },
    "zscore": {
        "label": "Z-Score",
        "params": [{"name": "window", "type": "integer", "default": 20, "min": 5, "max": 500}],
        "outputs": ["{as}"],
    },
    "donchian": {
        "label": "唐奇安通道 (Donchian)",
        "params": [{"name": "period", "type": "integer", "default": 20, "min": 5, "max": 200}],
        "outputs": ["{as}_up", "{as}_low"],
    },
    "volume_sma": {
        "label": "成交量均线 (Volume SMA)",
        "params": [{"name": "window", "type": "integer", "default": 20, "min": 2, "max": 500}],
        "outputs": ["{as}"],
    },
    "vwap": {
        "label": "成交量加权均价 (VWAP)",
        "params": [{"name": "window", "type": "integer", "default": 20, "min": 2, "max": 500}],
        "outputs": ["{as}"],
    },
}

# 内置可直接引用的价格/持仓字段
BUILTIN_REFS = ["close", "high", "low", "volume", "position"]
REF_LABELS = {
    "close": "收盘价 (close)",
    "high": "最高价 (high)",
    "low": "最低价 (low)",
    "volume": "成交量 (volume)",
    "position": "当前持仓 (position)",
}

OPERATORS = [">", "<", ">=", "<=", "==", "between", "not_between", "crosses_above", "crosses_below"]
OP_LABELS = {
    ">": "大于 (>)", "<": "小于 (<)", ">=": "大于等于 (>=)", "<=": "小于等于 (<=)",
    "==": "等于 (==)", "between": "介于 (between)", "not_between": "不介于 (not_between)",
    "crosses_above": "上穿 (crosses_above)", "crosses_below": "下穿 (crosses_below)",
}

POSITION_MODES = ["fixed_pct", "fixed_amount", "atr_pct", "strength_mapped"]
POSITION_LABELS = {
    "fixed_pct": "固定比例仓位",
    "fixed_amount": "固定金额仓位",
    "atr_pct": "ATR 动态仓位",
    "strength_mapped": "信号强度映射",
}


class IndicatorSpec(BaseModel):
    """指标声明：name（白名单）+ params + 输出名前缀 as。"""

    name: str
    params: dict = Field(default_factory=dict)
    as_: str = Field(..., alias="as")

    model_config = ConfigDict(populate_by_name=True)


class Condition(BaseModel):
    """单条条件：lhs 指标输出/内置引用，op 操作符，rhs 数值/引用/[lo,hi]。"""

    lhs: str
    op: str
    rhs: float | str | list[float]


class ConditionGroup(BaseModel):
    """条件组：内部 all（全部满足）/ any（任一满足）。"""

    logic: str = "all"
    conditions: list[Condition] = Field(default_factory=list)


class PositionSpec(BaseModel):
    mode: str = "fixed_pct"  # fixed_pct | fixed_amount | atr_pct | strength_mapped
    value: float = 1.0  # fixed_pct: 比例；fixed_amount: 金额；strength_mapped: 基准比例
    risk_per_trade: float = 0.01  # atr_pct：单笔风险
    stop_atr_mult: float = 2.0  # atr_pct：止损倍数（决定单位仓位）
    max_position: float = Field(1.0, ge=0.0, le=2.0)


class RuleParams(BaseModel):
    """可调参数（策略管理页 schema 表单渲染）。"""

    allow_short: bool = False
    only_long: bool = True
    max_hold_bars: int | None = Field(None, ge=1, le=10000)


class RuleStrategyConfig(BaseModel):
    """完整声明式策略配置（手动表单与 LLM 生成共用）。"""

    name: str = "未命名策略"
    description: str = ""
    indicators: list[IndicatorSpec] = Field(default_factory=list)
    entry_long: ConditionGroup | None = None
    exit_long: ConditionGroup | None = None
    entry_short: ConditionGroup | None = None
    exit_short: ConditionGroup | None = None
    position: PositionSpec = Field(default_factory=PositionSpec)
    params: RuleParams = Field(default_factory=RuleParams)
