"""Kronos 预测策略：用 Kronos 基础模型预测未来 K 线，据预测涨跌幅决定仓位。

Kronos（AAAI 2026，arXiv:2508.02739）是面向金融 K 线序列的开源基础模型，
对 OHLCV 做多步自回归预测。本策略把「预测未来 N 根的收盘价」翻译成目标仓位：

    预期收益 = 预测末根 close / 当前 close - 1

- 高于 ``long_threshold`` → 做多 ``position``
- 低于 ``short_threshold``（且允许做空）→ 做空 ``position``
- 其间 → 空仓

## 关键设计：滚动预测，不用未来数据

回测时 ``generate_signals`` 拿到的是整段 bars。若在每根 bar 上预测，
计算量会爆（Kronos CPU 约 0.7s/次，6000 根日线即 70 分钟）。
所以：

1. 从第 ``context`` 根开始，每 ``rebalance_every`` 根预测一次，
   **每次只用 ``bars[:i+1]``**，无前视；
2. 预测点数量受 ``max_forecasts`` 限制，超出时自动加大步长并写日志；
3. 预测结果 ``ffill`` 到下一预测点之间，形成台阶式仓位。

## 依赖是可选的

本策略依赖 ``torch``（CPU 轮子约 200MB）。未安装时策略仍可注册、可在 UI 里
看到，但运行会抛 ``KronosUnavailable`` 并给出安装命令——不影响其他策略。

## 成本提示

按默认参数（context=240、pred_len=12、rebalance_every=12）跑 1000 根 5 分钟线，
约 80 次预测，CPU 上约 1 分钟。日线策略请把 ``rebalance_every`` 设大一些（如 20）。
"""
from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.strategies.base import BaseStrategy, StrategyContext
from quant.strategies.registry import register

log = logging.getLogger(__name__)


@register
class KronosForecastStrategy(BaseStrategy):
    key: ClassVar[str] = "kronos_forecast"
    name: ClassVar[str] = "Kronos 预测策略"
    display_name: ClassVar[str] = "Kronos 预测策略 (Kronos Forecast)"
    description: ClassVar[str] = "用 Kronos 基础模型预测未来 K 线，按预测涨跌幅做多/做空"

    class Params(BaseModel):
        context: int = Field(240, ge=32, le=2048,
                             description="喂给模型的上下文 K 线根数（越多越稳，越慢）")
        pred_len: int = Field(12, ge=1, le=120,
                              description="预测未来多少根 K 线")
        rebalance_every: int = Field(12, ge=1, le=500,
                                     description="每多少根 K 线重新预测一次（越大越快）")
        threshold_sigma: float = Field(0.5, ge=0.0, le=10.0,
                                       description="预测收益达到近 N 根波动率的几倍才动手"
                                                   "（按波动率归一，不同周期通用；0.5 = 半个标准差）")
        min_move: float = Field(0.0, ge=0.0, le=0.2,
                                description="绝对涨跌幅下限，过滤微小波动（0 = 不设下限）")
        position: float = Field(1.0, ge=0.0, le=1.0,
                                description="触发信号时的目标仓位")
        only_long: bool = Field(True, description="只做多，不做空")
        max_forecasts: int = Field(200, ge=1, le=5000,
                                   description="单次回测的预测次数上限（超出自动加大步长）")
        model: str = Field("Kronos-small", description="模型：Kronos-mini / Kronos-small / Kronos-base")
        sample_count: int = Field(1, ge=1, le=8,
                                  description="采样路径数（多条取平均更稳，耗时线性增加）")

    # ---- 内部：预测器（进程内缓存，避免每次回测重复加载权重）----
    def _forecaster(self):
        from alpha.models import KronosForecaster, KronosUnavailable, available
        from alpha.models.kronos import KronosSpec, MAX_CONTEXT

        if not available():
            from alpha.models.kronos import install_hint, missing_deps

            raise KronosUnavailable(
                f"Kronos 策略依赖未安装：{', '.join(missing_deps())}。安装命令：{install_hint()}"
            )

        spec = KronosSpec(
            model=self.params.model,
            max_context=min(MAX_CONTEXT.get(self.params.model, 512), max(self.params.context, 32)),
            pred_len=self.params.pred_len,
            sample_count=self.params.sample_count,
        )
        try:
            return KronosForecaster(spec)
        except Exception as e:  # noqa: BLE001
            raise KronosUnavailable(f"Kronos 初始化失败：{e}") from e

    # ---- 核心：给定一段 bars，预测未来收益 ----
    def _expected_return(self, history: pd.DataFrame) -> float | None:
        """用 history（截至当前 bar，含当前）预测 pred_len 根后的收益。"""
        from alpha.models import KronosUnavailable

        n = len(history)
        if n < self.params.context:
            return None
        window = history.iloc[-self.params.context:]
        try:
            pred = self._forecaster().forecast(window, pred_len=self.params.pred_len)
        except KronosUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 - 单次预测失败不该让整个回测崩掉
            log.warning("Kronos 预测失败（%d 根）：%s", n, e)
            return None
        if pred is None or pred.empty or "close" not in pred.columns:
            return None
        last_close = float(window["close"].iloc[-1])
        pred_close = float(pred["close"].iloc[-1])
        if not np.isfinite(last_close) or not np.isfinite(pred_close) or last_close <= 0:
            return None
        return pred_close / last_close - 1.0

    def _weight_from_return(self, ret: float | None, sigma: float) -> tuple[float, str]:
        """把预测收益转成仓位。

        阈值按「近 N 根的实际波动率」归一（threshold_sigma 倍标准差），
        这样同一套默认参数在 5 分钟线和日线上都合理——绝对百分比阈值
        在 5 分钟线（0.1% 就算大波动）和日线（0.5% 才算）上是矛盾的。
        """
        p = self.params
        if ret is None:
            return 0.0, "预测不可用"
        pct = ret * 100
        need = max(sigma * p.threshold_sigma, p.min_move)
        if ret >= need and need > 0:
            return p.position, f"预测未来{p.pred_len}根 {pct:+.2f}%（阈值 {need * 100:+.2f}%）→ 做多"
        if not p.only_long and ret <= -need and need > 0:
            return -p.position, f"预测未来{p.pred_len}根 {pct:+.2f}%（阈值 {-need * 100:+.2f}%）→ 做空"
        return 0.0, f"预测未来{p.pred_len}根 {pct:+.2f}%（未达阈值 {need * 100:.2f}%）→ 空仓"

    @staticmethod
    def _sigma(history: pd.DataFrame, lookback: int = 60) -> float:
        """近 lookback 根的单根收益标准差（无量纲，用于阈值归一）。"""
        close = history["close"].astype(float)
        if len(close) < 3:
            return 0.0
        rets = close.pct_change().iloc[-lookback:]
        rets = rets[np.isfinite(rets)]
        if rets.empty:
            return 0.0
        return float(rets.std() or 0.0)

    # ---- BaseStrategy 契约 ----
    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        if bars is None or bars.empty:
            return pd.DataFrame({"target_weight": [], "reason": [], "confidence": []})

        n = len(bars)
        p = self.params
        warmup = max(p.context, 2)
        if n <= warmup:
            target = pd.Series(0.0, index=bars.index)
            return pd.DataFrame({
                "target_weight": target,
                "reason": f"K 线不足（需 > {warmup} 根，实际 {n} 根）",
                "confidence": 0.0,
            })

        # 预测点：warmup-1 起步，每 step 根一次；用 max_forecasts 反推最小步长
        start = warmup - 1
        span = n - start
        step = max(1, int(p.rebalance_every))
        if span / step > p.max_forecasts:
            step = int(np.ceil(span / p.max_forecasts))
            log.info("Kronos 预测次数超过上限 %d，步长自动调整为 %d 根", p.max_forecasts, step)

        points = list(range(start, n, step))

        # 逐点滚动预测（每次只用截至该点的数据，无前视）
        target = pd.Series(np.nan, index=bars.index)
        reason = pd.Series("", index=bars.index, dtype=object)
        conf = pd.Series(np.nan, index=bars.index)
        for i in points:
            window = bars.iloc[: i + 1]
            ret = self._expected_return(window)
            sigma = self._sigma(window)
            w, why = self._weight_from_return(ret, sigma)
            target.iloc[i] = w
            reason.iloc[i] = why
            # 置信度 = 预测幅度相对阈值的倍数，最多 1.0
            need = max(sigma * p.threshold_sigma, p.min_move, 1e-6)
            # 只有真的触发仓位时才算「有信心的信号」；空仓时置信度为 0，
            # 否则 UI 会出现「置信度 0.8 但目标仓位 0」的误导性组合
            conf.iloc[i] = 0.0 if (ret is None or w == 0.0) else float(np.clip(abs(ret) / (need * 3.0), 0.0, 1.0))

        # 台阶式持仓：预测点之间沿用上一次的结论
        target = target.ffill().fillna(0.0).astype(float)
        conf = conf.ffill().fillna(0.0).astype(float)
        reason = reason.replace("", np.nan).ffill().fillna("等待预测")
        # 预热期明确标注，避免被误读为「空仓信号」
        reason.iloc[:start] = "预热期（K 线不足）"

        return pd.DataFrame({
            "target_weight": target,
            "reason": reason,
            "confidence": conf,
        })

    # ---- 模拟盘：每根新 bar 只预测一次，避免重复跑整段 ----
    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> tuple[float, str, float]:
        history = ctx.history
        if history is None or len(history) < self.params.context:
            return 0.0, f"预热中（需 {self.params.context} 根）", 0.0

        # 同一根 bar 只算一次（模拟盘同一 tick 内可能被多次调用）
        stamp = history.index[-1]
        cache = getattr(self, "_bar_cache", None)
        if cache is not None and cache[0] == stamp:
            return cache[1]

        ret = self._expected_return(history)
        sigma = self._sigma(history)
        w, why = self._weight_from_return(ret, sigma)
        p = self.params
        need = max(sigma * p.threshold_sigma, p.min_move, 1e-6)
        c = 0.0 if (ret is None or w == 0.0) else float(np.clip(abs(ret) / (need * 3.0), 0.0, 1.0))
        self._bar_cache = (stamp, (w, why, c))
        return w, why, c
