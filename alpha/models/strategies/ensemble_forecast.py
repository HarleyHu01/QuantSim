"""多模型集成预测策略：多个预测模型加权投票 + 硬性门控决定仓位。

和单模型策略（如 ``kronos_forecast``）的区别在于**它要求模型之间达成共识**：

- 单模型说涨 → 就做多，模型错了你无从得知
- 集成说涨、且四个模型方向一致、分歧小于阈值 → 才做多

## 为什么阈值要按波动率归一

绝对百分比阈值跨周期不成立：5 分钟线 0.1% 已是大波动，日线 0.5% 才算。
所以门槛是 ``min_return_sigma × 近 60 根单根收益标准差``，
1 分钟线和月线用同一套默认参数都说得通。

## 为什么要有硬性门控

加权平均有个陷阱：+3%、-2%、+0.1%、-5% 平均后可能接近 0，
看似「中性」，真相是**模型们完全不知道**。``max_dispersion`` 用模型间
标准差识别这种情形并强制空仓 —— 宁可不下注，也不要让平均值蒙混过关。

## 成本

每次预测点要跑 N 个模型。默认权重只开 ``kronos + ewma + momentum``
（其余权重为 0），Kronos 单次约 0.7 秒，所以默认配置下每个预测点约 1 秒。
回测 1000 根 5 分钟线、``rebalance_every=20`` 即 50 个预测点 ≈ 1 分钟。
把 ``weights`` 里 kronos 调 0 就退回纯统计基线，毫秒级。
"""
from __future__ import annotations

import json
import logging
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.strategies.base import BaseStrategy, StrategyContext
from quant.strategies.registry import register

log = logging.getLogger(__name__)


def _default_weights_json() -> str:
    """默认权重的 JSON 文本（UI 里当文本框初值，也便于用户直接改）。"""
    from alpha.models.ensemble import DEFAULT_WEIGHTS

    return json.dumps(DEFAULT_WEIGHTS, ensure_ascii=False)


@register
class EnsembleForecastStrategy(BaseStrategy):
    key: ClassVar[str] = "ensemble_forecast"
    name: ClassVar[str] = "多模型集成策略"
    display_name: ClassVar[str] = "多模型集成策略 (Ensemble Forecast)"
    description: ClassVar[str] = ("多个预测模型（Kronos / 统计基线 / 外部时序模型）加权投票，"
                                  "通过一致性、分歧度等硬性门控后才建仓")

    class Params(BaseModel):
        # ---- 模型权重 ----
        weights_json: str = Field(
            default_factory=_default_weights_json,
            description='各模型权重 JSON，如 {"kronos":1.0,"ewma":1.0,"momentum":0.5}；'
                        '权重 0 = 不启用该模型')
        # ---- 硬性门控 ----
        min_return_sigma: float = Field(
            0.3, ge=0.0, le=10.0,
            description="集成预测幅度至少是近 N 根波动率的几倍才动手（按波动率归一，跨周期通用）")
        min_abs_return: float = Field(
            0.0, ge=0.0, le=0.5,
            description="预测幅度的绝对下限（0 = 不限，纯靠波动率归一）")
        min_agreement: float = Field(
            0.6, ge=0.0, le=1.0,
            description="同向模型的权重占比下限，低于此值视为「模型分歧大」→ 空仓")
        max_dispersion: float = Field(
            2.0, ge=0.0, le=20.0,
            description="模型之间预测的标准差不得超过波动率的几倍（超过视为没人知道 → 空仓）")
        min_models: int = Field(
            2, ge=1, le=9,
            description="至少几个模型给出有效预测才认可结果")
        # ---- 仓位 ----
        position: float = Field(1.0, ge=0.0, le=1.0, description="触发信号时的目标仓位")
        allow_short: bool = Field(False, description="允许做空（否则只做多）")
        # ---- 运行参数 ----
        context: int = Field(240, ge=32, le=2048,
                             description="喂给模型的上下文 K 线根数")
        pred_len: int = Field(12, ge=1, le=120, description="预测未来多少根 K 线")
        rebalance_every: int = Field(20, ge=1, le=500,
                                     description="每多少根 K 线重新预测一次（越大越快）")
        max_forecasts: int = Field(150, ge=1, le=5000,
                                   description="单次回测的预测次数上限（超出自动加大步长）")

    # ---- 集成器缓存 ----
    def _ensure_ensemble(self):
        from alpha.models.ensemble import EnsembleForecaster, EnsembleSpec

        cached = getattr(self, "_ens_cache", None)
        if cached is not None:
            return cached[0]

        p = self.params
        try:
            weights = json.loads(p.weights_json or "{}")
            if not isinstance(weights, dict):
                weights = {}
        except json.JSONDecodeError as e:
            log.warning("权重 JSON 解析失败，改用默认权重：%s", e)
            from alpha.models.ensemble import DEFAULT_WEIGHTS

            weights = dict(DEFAULT_WEIGHTS)

        spec = EnsembleSpec(
            weights={str(k): float(v) for k, v in weights.items()
                     if isinstance(v, (int, float))},
            min_abs_return=p.min_abs_return,
            min_return_sigma=p.min_return_sigma,
            min_agreement=p.min_agreement,
            max_dispersion=p.max_dispersion,
            min_models=p.min_models,
            position=p.position,
            allow_short=p.allow_short,
            pred_len=p.pred_len,
            context=p.context,
        )
        ens = EnsembleForecaster(spec)
        self._ens_cache = (ens, spec)
        return ens

    def _predict(self, window: pd.DataFrame):
        """跑一次集成预测。返回 EnsembleResult 或 None（失败时）。"""
        try:
            ens = self._ensure_ensemble()
            return ens.predict(window, pred_len=self.params.pred_len)
        except Exception as e:  # noqa: BLE001 - 单点失败不该让整段回测崩
            log.warning("集成预测失败（%d 根）：%s", len(window), e)
            return None

    @staticmethod
    def _sigma(history: pd.DataFrame, lookback: int = 60) -> float:
        from alpha.models.base import return_sigma

        return return_sigma(history["close"], lookback=lookback)

    # ---- BaseStrategy 契约 ----
    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        if bars is None or bars.empty:
            return pd.DataFrame({"target_weight": [], "reason": [], "confidence": []})

        n = len(bars)
        p = self.params
        warmup = max(p.context, 2)
        if n <= warmup:
            return pd.DataFrame({
                "target_weight": pd.Series(0.0, index=bars.index),
                "reason": f"K 线不足（需 > {warmup} 根，实际 {n} 根）",
                "confidence": 0.0,
            })

        start = warmup - 1
        span = n - start
        step = max(1, int(p.rebalance_every))
        if span / step > p.max_forecasts:
            step = int(np.ceil(span / p.max_forecasts))
            log.info("集成预测次数超过上限 %d，步长自动调整为 %d 根", p.max_forecasts, step)
        points = list(range(start, n, step))

        target = pd.Series(np.nan, index=bars.index)
        reason = pd.Series("", index=bars.index, dtype=object)
        conf = pd.Series(np.nan, index=bars.index)

        for i in points:
            window = bars.iloc[: i + 1]
            res = self._predict(window)
            if res is None:
                target.iloc[i] = 0.0
                reason.iloc[i] = "集成预测不可用"
                conf.iloc[i] = 0.0
                continue
            target.iloc[i] = float(res.target_weight)
            reason.iloc[i] = res.reason
            conf.iloc[i] = float(res.confidence)

        target = target.ffill().fillna(0.0).astype(float)
        conf = conf.ffill().fillna(0.0).astype(float)
        reason = reason.replace("", np.nan).ffill().fillna("等待预测")
        reason.iloc[:start] = "预热期（K 线不足）"

        return pd.DataFrame({"target_weight": target, "reason": reason,
                             "confidence": conf})

    # ---- 模拟盘：每根新 bar 只预测一次 ----
    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> tuple[float, str, float]:
        history = ctx.history
        if history is None or len(history) < self.params.context:
            return 0.0, f"预热中（需 {self.params.context} 根）", 0.0

        stamp = history.index[-1]
        cache = getattr(self, "_bar_cache", None)
        if cache is not None and cache[0] == stamp:
            return cache[1]

        window = history.iloc[-self.params.context:]
        res = self._predict(window)
        if res is None:
            out = (0.0, "集成预测不可用", 0.0)
        else:
            out = (float(res.target_weight), res.reason, float(res.confidence))
        self._bar_cache = (stamp, out)
        return out
