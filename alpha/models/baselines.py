"""零依赖的统计基线预测器。

为什么要有这一层：Kronos 这类深度模型很酷，但**基线的存在让结论可信**。
如果某个深度模型跑不赢「随机游走」或「动量外推」，那它的复杂度就是白付的成本。
这些基线全部只用 pandas/numpy，毫秒级完成，不下载任何权重，
所以：

- 没装 torch 的机器也能用集成预测；
- 它们天然作为集成里的「对照组」，可以给权重 0 关掉。

四个模型覆盖四种经典假设：

===============  ==========================================  ==================
模型              假设                                        一句话
===============  ==========================================  ==================
``random_walk``   明天 = 今天（有效市场）                      最诚实的基线
``drift``         历史平均漂移会延续                          趋势外推
``momentum``      最近的涨跌会短期延续                        追涨杀跌
``reversion``     偏离均线太远会回归                          均值回归
``ewma``          漂移的指数加权估计更贴合近期                半衰期加权的 drift
===============  ==========================================  ==================
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from alpha.models.base import (BaseForecaster, future_index, path_to_ohlcv,
                           prepare_bars, return_sigma)


def _log_return(close: pd.Series) -> pd.Series:
    """对数收益率（比简单收益率更适合做可加的外推）。"""
    return np.log(close.astype(float)).diff()


def _project(last_close: float, last_volume: float, mu: float, n: int,
             index: pd.DatetimeIndex, sigma: float, damping: float = 0.0) -> pd.DataFrame:
    """按每根漂移 `mu` 外推收盘路径，再补出 OHLCV。

    `damping` 让漂移随预测步数衰减：``mu_t = mu * (1-damping)**t``。
    真实行情里「趋势能延续 N 根」这个假设随 N 增大迅速失效，
    所以默认给一点衰减（见各子类的 `damping` 参数）。
    """
    if damping > 0:
        growth = np.cumprod(np.full(n, 1.0 - damping))
        drift = mu * growth
    else:
        drift = np.full(n, mu)
    # 对数收益可加：累计求和后一次性取指数，避免逐步连乘累积浮点误差
    closes = pd.Series(last_close * np.exp(np.cumsum(drift)), index=index, dtype=float)
    return path_to_ohlcv(closes, last_close=last_close, sigma=sigma,
                         last_volume=last_volume)


class _StatForecaster(BaseForecaster):
    """统计基线的公共部分：取数 → 估漂移 → 外推。"""

    kind = "stat"
    #: 估漂移用的回看根数上限
    lookback: int = 120
    #: 漂移衰减速率（0 = 不衰减，越接近 1 衰减越快）
    damping: float = 0.0

    def _prepare(self, bars: pd.DataFrame, params: dict[str, Any] | None = None
                 ) -> tuple[pd.DataFrame, int, int]:
        p = {**self.params, **(params or {})}
        lookback = int(p.get("lookback", self.lookback))
        ctx = prepare_bars(bars, min_bars=3)
        ctx = ctx.iloc[-lookback:] if lookback > 0 else ctx
        return ctx, lookback, int(p.get("_pred_len", 0) or 0)

    def _estimate(self, ctx: pd.DataFrame) -> float:
        """返回每根 K 线的对数漂移估计。子类覆写。"""
        raise NotImplementedError

    def forecast_path(self, bars: pd.DataFrame, pred_len: int) -> pd.Series:
        """只要收盘路径（集成层用，省掉 OHLC 撑开）。"""
        ctx = prepare_bars(bars, min_bars=3)
        lb = int(self.params.get("lookback", self.lookback))
        if lb > 0:
            ctx = ctx.iloc[-lb:]
        mu = self._estimate(ctx)
        n = int(pred_len)
        idx = future_index(ctx.index, n)
        damp = float(self.params.get("damping", self.damping))
        growth = np.cumprod(np.full(n, 1.0 - damp)) if damp > 0 else np.ones(n)
        last = float(ctx["close"].iloc[-1])
        return pd.Series(last * np.exp(np.cumsum(mu * growth)), index=idx, dtype=float)

    def forecast(self, bars: pd.DataFrame, pred_len: int) -> pd.DataFrame:
        n = int(pred_len)
        if n <= 0:
            raise ValueError(f"pred_len 必须为正整数，收到 {pred_len}")
        ctx = prepare_bars(bars, min_bars=3)
        lb = int(self.params.get("lookback", self.lookback))
        if lb > 0:
            ctx = ctx.iloc[-lb:]
        mu = self._estimate(ctx)
        sigma = return_sigma(ctx["close"])
        last_close = float(ctx["close"].iloc[-1])
        last_vol = float(ctx["volume"].iloc[-1]) if "volume" in ctx else 0.0
        idx = future_index(ctx.index, n)
        damp = float(self.params.get("damping", self.damping))
        return _project(last_close, last_vol, mu, n, idx, sigma, damping=damp)

    def status(self) -> dict[str, Any]:
        return {
            "key": self.key, "label": self.label, "kind": self.kind,
            "available": True, "loaded": True, "requires": [],
            "missing": [], "device": "cpu", "load_seconds": None,
            "params": dict(self.params),
            "summary": self.summary(),
        }

    @classmethod
    def summary(cls) -> str:
        return ""


class RandomWalkForecaster(_StatForecaster):
    """随机游走：预测收盘 = 当前收盘（漂移为 0）。

    这是金融时序里最难打败的基线 —— 大量论文的结论是大部分模型都跑不赢它。
    放进集成里当「锚」：任何看多/看空预测都得先说服这个零假设。
    """

    key = "random_walk"
    label = "随机游走（基线）"
    lookback = 0

    def _estimate(self, ctx: pd.DataFrame) -> float:
        return 0.0

    @classmethod
    def summary(cls) -> str:
        return "假设有效市场：未来收盘 = 当前收盘。集成里的零假设锚点。"


class DriftForecaster(_StatForecaster):
    """历史平均漂移外推：``mu = mean(log_return)``。"""

    key = "drift"
    label = "历史漂移外推"
    lookback = 250
    damping = 0.0

    def _estimate(self, ctx: pd.DataFrame) -> float:
        lr = _log_return(ctx["close"]).dropna()
        if lr.empty:
            return 0.0
        mu = float(lr.mean())
        return mu if math.isfinite(mu) else 0.0

    @classmethod
    def summary(cls) -> str:
        return "用全部历史（最多 250 根）的平均对数收益外推未来。"


class MomentumForecaster(_StatForecaster):
    """动量：最近 N 根的收益线性外推，并带衰减。"""

    key = "momentum"
    label = "动量外推"
    lookback = 20
    damping = 0.03

    def _estimate(self, ctx: pd.DataFrame) -> float:
        close = ctx["close"].astype(float)
        if len(close) < 3:
            return 0.0
        span = min(len(close) - 1, int(self.params.get("span", 20)))
        lr = _log_return(close).dropna().iloc[-span:]
        if lr.empty:
            return 0.0
        mu = float(lr.mean())
        return mu if math.isfinite(mu) else 0.0

    @classmethod
    def summary(cls) -> str:
        return "最近 20 根的平均收益外推，按 3%/根衰减（趋势不能无限延续）。"


class ReversionForecaster(_StatForecaster):
    """均值回归：预测向 N 日均线收敛，收敛速度可调。

    ``mu = -speed * log(close / ma) / horizon``
    —— 当前价高于均线则预测下跌（mu 为负），反之为正。
    """

    key = "reversion"
    label = "均值回归"
    lookback = 120
    damping = 0.0

    def _estimate(self, ctx: pd.DataFrame) -> float:
        close = ctx["close"].astype(float)
        window = int(self.params.get("ma_window", 20))
        if len(close) < window + 1:
            return 0.0
        ma = float(close.rolling(window).mean().iloc[-1])
        last = float(close.iloc[-1])
        if ma <= 0 or last <= 0:
            return 0.0
        gap = math.log(last / ma)          # >0 表示高于均线
        speed = float(self.params.get("speed", 0.15))
        horizon = max(1, int(self.params.get("horizon", 10)))
        return -speed * gap / horizon

    @classmethod
    def summary(cls) -> str:
        return "偏离 20 日均线越远，预测回归越强（默认每根收敛 15% 的偏离量）。"


class EwmaForecaster(_StatForecaster):
    """指数加权漂移：近期收益权重更高，比等权平均更跟得上状态切换。"""

    key = "ewma"
    label = "指数加权漂移"
    lookback = 120
    damping = 0.02

    def _estimate(self, ctx: pd.DataFrame) -> float:
        lr = _log_return(ctx["close"]).dropna()
        if lr.empty:
            return 0.0
        halflife = float(self.params.get("halflife", 12))
        mu = float(lr.ewm(halflife=max(1.0, halflife), adjust=False).mean().iloc[-1])
        return mu if math.isfinite(mu) else 0.0

    @classmethod
    def summary(cls) -> str:
        return "对数收益的指数加权均值（半衰期 12 根），比等权更贴近最近的节奏。"


#: 所有零依赖基线（顺序即 UI 展示顺序）
STAT_FORECASTERS: dict[str, type[_StatForecaster]] = {
    cls.key: cls
    for cls in (
        RandomWalkForecaster,
        DriftForecaster,
        MomentumForecaster,
        ReversionForecaster,
        EwmaForecaster,
    )
}
