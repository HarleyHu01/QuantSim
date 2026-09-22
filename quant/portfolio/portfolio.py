"""StrategyPortfolio：策略组合与权重合成。

三种模式（公式见 docs/BACKTEST_ENGINE_DESIGN.md §1.4）：
- signal_merge（默认）：combined = Σ wᵢ·sᵢ·confᵢ / Σ|wᵢ·confᵢ|，clip 到 [-1,1]
- equal：wᵢ 全相等特例
- capital_split：每个策略独立子账户（资金按 |wᵢ| 比例分配，引擎侧执行）

策略输出信号列 = [target_weight, reason, confidence]。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant.core.models import CombinedSignal, PortfolioConfig
from quant.strategies.base import BaseStrategy

SIGNAL_COLUMNS = ["target_weight", "reason", "confidence"]


@dataclass
class SubPortfolio:
    """capital_split 的一个子账户配置（signal_merge/equal 时只有一个组合子账户）。"""

    key: str  # 策略 key（signal_merge 用 "__combined__"）
    weight: float
    capital_pct: float  # 子账户资金占比
    strategy_keys: list[str] = field(default_factory=list)


class StrategyPortfolio:
    def __init__(self, config: PortfolioConfig):
        self.config = config

    # ---- 拆分子组合 ----
    def split(self, strategies: list[BaseStrategy]) -> list[SubPortfolio]:
        mode = self.config.mode
        if mode == "capital_split":
            weights = {s.key: self.config.weights.get(s.key, 0.0) for s in strategies}
            total_abs = sum(abs(w) for w in weights.values())
            if total_abs <= 0:
                return []
            subs = []
            for s in strategies:
                w = weights[s.key]
                if w == 0:
                    continue
                subs.append(SubPortfolio(s.key, w, abs(w) / total_abs, strategy_keys=[s.key]))
            return subs
        # signal_merge / equal：单一组合子账户，包含全部策略
        return [SubPortfolio("__combined__", 1.0, 1.0, strategy_keys=[s.key for s in strategies])]

    # ---- 权重解析 ----
    def _weights_for(self, strategy_keys: list[str]) -> dict[str, float]:
        # equal 或未配置权重 → 等权
        if self.config.mode == "equal" or not self.config.weights:
            return {k: 1.0 for k in strategy_keys}
        return {k: self.config.weights.get(k, 0.0) for k in strategy_keys}

    # ---- 单 bar 合成（事件驱动引擎 / UI 信号分解） ----
    def combine_at(
        self,
        signals: dict[str, pd.DataFrame],
        ts: pd.Timestamp,
        strategy_keys: list[str] | None = None,
    ) -> CombinedSignal:
        """signals: strategy_key → 该策略的信号 DataFrame（与 symbol bars 对齐）。

        取 ts 时刻各策略的 (target_weight, confidence) 合成。strategy_keys 缺省取 signals 键。
        """
        keys = strategy_keys if strategy_keys is not None else list(signals.keys())
        weights = self._weights_for(keys)
        vals: dict[str, tuple[float, float]] = {}
        reasons: list[str] = []
        for k in keys:
            df = signals.get(k)
            if df is None or df.empty or ts not in df.index:
                continue
            row = df.loc[ts]
            tw = float(row.get("target_weight", 0.0))
            cf = float(row.get("confidence", 1.0))
            vals[k] = (tw, cf)
            r = str(row.get("reason", ""))
            if r:
                reasons.append(r)

        if not vals:
            return CombinedSignal(0.0, weights, {}, 1.0, "")

        use_conf = self.config.confidence_weighted
        num = 0.0
        den = 0.0
        contrib: dict[str, float] = {}
        for k, (tw, cf) in vals.items():
            w = weights.get(k, 0.0)
            factor = (w * cf) if use_conf else w
            num += factor * tw
            den += abs(factor)
            contrib[k] = factor * tw
        if den <= 0:
            return CombinedSignal(0.0, weights, {}, 1.0, "")
        target = max(-1.0, min(1.0, num / den))
        contrib = {k: (v / den if den else 0.0) for k, v in contrib.items()}
        # 合成置信度：按 |wᵢ| 加权平均
        wsum = sum(abs(weights.get(k, 0.0)) for k in vals)
        conf = sum(abs(weights.get(k, 0.0)) * cf for k, (_, cf) in vals.items()) / wsum if wsum else 1.0
        reason = "; ".join(reasons) if reasons else ""
        return CombinedSignal(target, weights, contrib, conf, reason)

    # ---- 全序列合成（向量化引擎） ----
    def combine_series(self, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """signals: strategy_key → 信号 DataFrame（同一 index）。返回合成序列。"""
        keys = list(signals.keys())
        if not keys:
            return pd.DataFrame(columns=SIGNAL_COLUMNS)
        weights = self._weights_for(keys)
        use_conf = self.config.confidence_weighted

        target = pd.Series(0.0, index=signals[keys[0]].index)
        confs = pd.Series(1.0, index=target.index)
        reasons = pd.Series("", index=target.index, dtype=object)

        den_series = pd.Series(0.0, index=target.index)
        num_series = pd.Series(0.0, index=target.index)
        for k in keys:
            df = signals[k]
            if df is None or df.empty:
                continue
            tw = df["target_weight"].reindex(target.index).fillna(0.0)
            cf = df["confidence"].reindex(target.index).fillna(1.0)
            w = weights.get(k, 0.0)
            factor = (w * cf) if use_conf else w
            num_series = num_series + factor * tw
            den_series = den_series + abs(factor)
            # reason 拼接（非空时）
            r = df["reason"].reindex(target.index).fillna("").astype(str)
            mask = r != ""
            reasons = reasons.mask(mask, reasons[mask].str.cat(r[mask], sep="; ") if reasons[mask].any() else r[mask])
        den_series = den_series.replace(0.0, pd.NA)
        target = (num_series / den_series).clip(-1.0, 1.0).fillna(0.0)

        wsum = sum(abs(weights.get(k, 0.0)) for k in keys)
        if wsum > 0:
            conf_sum = pd.Series(0.0, index=target.index)
            for k in keys:
                df = signals[k]
                if df is None or df.empty:
                    continue
                cf = df["confidence"].reindex(target.index).fillna(1.0)
                conf_sum = conf_sum + abs(weights.get(k, 0.0)) * cf
            confs = (conf_sum / wsum).clip(0.0, 1.0)

        return pd.DataFrame({
            "target_weight": target,
            "reason": reasons,
            "confidence": confs,
        })
