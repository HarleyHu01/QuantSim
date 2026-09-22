"""声明式规则策略：按 RuleStrategyConfig 计算指标并评估买卖条件，输出目标仓位。

与内置策略相同接口（generate_signals → target_weight/reason/confidence），
因此可参与回测、组合、模拟盘；手动表单与 LLM 生成共用本类。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.strategies.base import BaseStrategy
from quant.strategies.llm.schema import (BUILTIN_REFS, INDICATOR_WHITELIST,
                                         OPERATORS, OP_LABELS, POSITION_LABELS,
                                         POSITION_MODES, REF_LABELS,
                                         Condition, ConditionGroup,
                                         RuleParams, RuleStrategyConfig)
from quant.strategies.llm.validator import validate_config
from quant.core import indicators as ind


class RuleStrategy(BaseStrategy):
    """规则策略：一组指标声明 + 买/卖/做空/平空条件组 + 仓位设置。

    传入 params（dict）：
    - 含 "indicators" 键 → 视为完整 RuleStrategyConfig（手动表单 / LLM 生成 / DB 加载）
    - 否则视为 RuleParams 可调参数（配合已加载的 self.config）
    """

    key: str = "rule"
    name: str = "规则策略"
    display_name: str = "规则策略 (Rule Strategy)"
    description: str = "声明式条件规则策略（手动创建或 LLM 生成）"

    class Params(BaseModel):
        allow_short: bool = False
        only_long: bool = True
        max_hold_bars: int | None = Field(None, ge=1, le=10000)

    def __init__(self, params: dict | None = None, config: dict | None = None):
        if config is not None:
            self.config = RuleStrategyConfig(**config)
        elif isinstance(params, dict) and "indicators" in params:
            self.config = RuleStrategyConfig(**params)
        else:
            self.config = None
        if self.config is not None:
            params = self.config.params.model_dump()
        super().__init__(params)
        self.params: RuleParams = RuleParams(**(params or {}))

    # ---- 指标计算 ----
    def _compute_indicators(self, bars: pd.DataFrame,
                            specs) -> dict[str, pd.Series]:
        close = bars["close"]
        env: dict[str, pd.Series] = {}
        for spec in specs:
            p = spec.params or {}
            asn = spec.as_
            name = spec.name
            if name == "sma":
                env[asn] = ind.sma(close, p["window"])
            elif name == "ema":
                env[asn] = ind.ema(close, p["window"])
            elif name == "macd":
                m = ind.macd(close, p.get("fast", 12), p.get("slow", 26), p.get("signal", 9))
                env[asn] = m["macd"]
                env[f"{asn}_signal"] = m["signal"]
                env[f"{asn}_hist"] = m["hist"]
            elif name == "rsi":
                env[asn] = ind.rsi(close, p["period"])
            elif name == "boll":
                b = ind.bollinger(close, p.get("period", 20), p.get("num_std", 2.0))
                env[f"{asn}_up"] = b["up"]
                env[f"{asn}_mid"] = b["mid"]
                env[f"{asn}_low"] = b["low"]
            elif name == "atr":
                env[asn] = ind.atr(bars, p["period"])
            elif name == "roc":
                env[asn] = ind.roc(close, p["window"])
            elif name == "zscore":
                env[asn] = ind.zscore(close, p["window"])
            elif name == "donchian":
                d = ind.donchian(close, p["period"])
                env[f"{asn}_up"] = d["high"]
                env[f"{asn}_low"] = d["low"]
            elif name == "volume_sma":
                env[asn] = bars["volume"].rolling(int(p["window"]), min_periods=1).mean()
            elif name == "vwap":
                vol = bars["volume"].rolling(int(p["window"]), min_periods=1).sum()
                env[asn] = (bars["close"] * bars["volume"]).rolling(
                    int(p["window"]), min_periods=1).sum() / vol.replace(0, np.nan)
        for ref in BUILTIN_REFS:
            if ref in bars.columns:
                env[ref] = bars[ref]
            else:
                env[ref] = pd.Series(0.0, index=bars.index)
        return env

    # ---- 条件求值 ----
    def _eval_condition(self, cond: Condition, env: dict[str, pd.Series]) -> pd.Series:
        lhs = env[cond.lhs]
        op = cond.op
        rhs = cond.rhs
        if op == "between":
            lo, hi = rhs  # type: ignore[misc]
            return (lhs >= lo) & (lhs <= hi)
        if op == "not_between":
            lo, hi = rhs  # type: ignore[misc]
            return ~((lhs >= lo) & (lhs <= hi))
        r = env[rhs] if isinstance(rhs, str) else float(rhs)  # type: ignore[arg-type]
        if op == "crosses_above":
            return (lhs > r) & (lhs.shift(1) <= r)
        if op == "crosses_below":
            return (lhs < r) & (lhs.shift(1) >= r)
        if op == ">":
            return lhs > r
        if op == "<":
            return lhs < r
        if op == ">=":
            return lhs >= r
        if op == "<=":
            return lhs <= r
        if op == "==":
            return lhs == r
        raise ValueError(f"未知操作符: {op}")

    def _group_mask(self, group: ConditionGroup | None,
                    env: dict[str, pd.Series]) -> pd.Series | None:
        if group is None or not group.conditions:
            return None
        masks = [self._eval_condition(c, env) for c in group.conditions]
        if group.logic == "all":
            out = masks[0]
            for m in masks[1:]:
                out = out & m
        else:
            out = masks[0]
            for m in masks[1:]:
                out = out | m
        return out.fillna(False)

    def _group_strength(self, group: ConditionGroup | None,
                        env: dict[str, pd.Series]) -> pd.Series:
        """条件满足比例（信号强度映射用）：满足条数 / 总条数。"""
        if group is None or not group.conditions:
            return pd.Series(1.0, index=env["close"].index)
        n = len(group.conditions)
        masks = [self._eval_condition(c, env).fillna(False) for c in group.conditions]
        total = masks[0].astype(int)
        for m in masks[1:]:
            total = total + m.astype(int)
        return total / n

    # ---- 仓位 ----
    def _position_weight(self, bars: pd.DataFrame, env: dict[str, pd.Series],
                         i: int, ctx, strength: float) -> float:
        cfg = self.config
        pos = cfg.position
        close = float(bars["close"].iloc[i])
        if pos.mode == "fixed_pct":
            w = pos.value
        elif pos.mode == "fixed_amount":
            capital = (getattr(ctx, "extra", None) or {}).get("initial_capital", 1_000_000)
            w = pos.value / capital
        elif pos.mode == "atr_pct":
            atr = ind.atr(bars, 14).iloc[i]
            w = pos.risk_per_trade / (pos.stop_atr_mult * atr / close) if atr and atr > 0 else pos.value
        else:  # strength_mapped
            w = pos.value * float(strength)
        return float(np.clip(w, 0.0, pos.max_position))

    # ---- 主逻辑 ----
    def generate_signals(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        n = len(bars)
        idx = bars.index
        cfg = self.config
        if cfg is None:
            return pd.DataFrame({"target_weight": 0.0, "reason": "", "confidence": 1.0},
                                index=idx)
        env = self._compute_indicators(bars, cfg.indicators)
        el = self._group_mask(cfg.entry_long, env)
        xl = self._group_mask(cfg.exit_long, env)
        es = self._group_mask(cfg.entry_short, env) if (cfg.entry_short and self.params.allow_short) else None
        xs = self._group_mask(cfg.exit_short, env) if (cfg.exit_short and self.params.allow_short) else None
        el_str = self._group_strength(cfg.entry_long, env)
        es_str = self._group_strength(cfg.entry_short, env) if cfg.entry_short else el_str

        target = np.zeros(n)
        reason: list[str] = [""] * n
        pos = 0
        entry_i = 0
        weight = 0.0
        max_hold = self.params.max_hold_bars

        for i in range(n):
            if pos == 0:
                if el is not None and bool(el.iloc[i]):
                    pos = 1
                    entry_i = i
                    weight = self._position_weight(bars, env, i, ctx, el_str.iloc[i])
                    target[i] = weight
                    reason[i] = "买入条件触发"
                elif es is not None and bool(es.iloc[i]):
                    pos = -1
                    entry_i = i
                    weight = -self._position_weight(bars, env, i, ctx, es_str.iloc[i])
                    target[i] = weight
                    reason[i] = "做空条件触发"
            elif pos == 1:
                held_out = bool(xl is not None and xl.iloc[i])
                if held_out or (max_hold and (i - entry_i) >= max_hold):
                    pos = 0
                    weight = 0.0
                    reason[i] = "卖出条件触发" if held_out else f"最大持仓到期（{max_hold} 根）"
                target[i] = weight
            else:  # pos == -1
                held_out = bool(xs is not None and xs.iloc[i])
                if held_out or (max_hold and (i - entry_i) >= max_hold):
                    pos = 0
                    weight = 0.0
                    reason[i] = "平空条件触发" if held_out else f"最大持仓到期（{max_hold} 根）"
                target[i] = weight

        return pd.DataFrame({"target_weight": target, "reason": reason, "confidence": 1.0},
                            index=idx)

    # ---- 元信息 ----
    @classmethod
    def config_schema(cls) -> dict:
        """表单元信息：指标白名单 / 操作符 / 仓位模式 / 引用字段。"""
        return {
            "indicators": [
                {"key": k, "label": v["label"], "params": v["params"], "outputs": v["outputs"]}
                for k, v in INDICATOR_WHITELIST.items()
            ],
            "operators": [
                {"key": k, "label": OP_LABELS.get(k, k)} for k in OPERATORS
            ],
            "refs": [{"key": k, "label": REF_LABELS[k]} for k in BUILTIN_REFS],
            "position_modes": [
                {"key": k, "label": POSITION_LABELS.get(k, k)} for k in POSITION_MODES
            ],
        }
