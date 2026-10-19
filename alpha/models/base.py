"""预测器公共层：行情预处理、未来时间索引、统一错误类型。

这一层的目的，是让三类完全不同的预测器共用同一套调用约定：

- ``Kronos``（K 线基础模型，torch 权重，慢但表达力强）
- ``ewma / drift / momentum / reversion``（纯 pandas 统计基线，零依赖、毫秒级）
- ``Chronos / TimesFM / Moirai``（外部时序基础模型，各自需要额外安装）

统一约定::

    forecaster.status() -> dict              # 只查状态，绝不触发加载
    forecaster.forecast(bars, pred_len)       # -> DataFrame[open, high, low, close, volume, amount]
    forecaster.available -> bool              # 依赖是否齐备

为什么要 `status()` 与加载分离：Kronos 首次加载含权重下载要 4~50 秒，
如果「看一眼页面」就触发加载，UI 会卡死；所以列表页只查状态。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# 输入必须含的四列；volume 可缺失（按 0 处理）
PRICE_COLS = ("open", "high", "low", "close")
# 统一输出列（与 Kronos 上游一致，便于画图和对比）
OUT_COLS = ["open", "high", "low", "close", "volume", "amount"]


class ForecasterUnavailable(RuntimeError):
    """依赖缺失 / 模型未启用 / 权重加载失败。

    统一用这个异常，API 层好统一转成「可读提示 + 安装命令」，
    而不用在调用处分别 catch ImportError / RuntimeError / OSError。
    """


def return_sigma(close: pd.Series, lookback: int = 60) -> float:
    """近 `lookback` 根的单根收益率标准差（波动率代理）。

    集成层用它把「预测幅度」归一成「几个标准差」，这样同一套阈值
    在 1 分钟线和月线上都说得通 —— 绝对百分比阈值做不到这点：
    5 分钟线 0.1% 已是大波动，日线 0.5% 才算。
    """
    if close is None or len(close) < 3:
        return 0.0
    rets = pd.Series(close).astype(float).pct_change().iloc[-int(lookback):]
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty:
        return 0.0
    sigma = float(rets.std())
    return sigma if np.isfinite(sigma) and sigma > 0 else 0.0


def prepare_bars(
    bars: pd.DataFrame,
    *,
    tz: str = "Asia/Shanghai",
    max_context: int | None = None,
    min_bars: int = 2,
) -> pd.DataFrame:
    """校验并规整行情，返回 ``[open, high, low, close, volume, amount]``。

    做的事（顺序有讲究）：

    1. 索引必须是 ``DatetimeIndex``；naive 视为本地时间 `tz`。
    2. 价格列缺失直接报错 —— 缺 OHLC 就没法预测，早报比晚报好。
    3. `amount` 由 ``volume × 均价`` 现算：上游模型（Kronos）要它，
       但本地行情表里没有这一列。
    4. 价格列含 NaN 的行丢弃（停牌/新股常见），而不是填 0 ——
       填 0 会把模型带偏。
    5. 超过 `max_context` 只保留尾部：模型上下文有上限，截早了会丢最新信息。
    """
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise ValueError("bars 为空，无法预测")

    missing = [c for c in PRICE_COLS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars 缺少列：{missing}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars 的索引必须是 DatetimeIndex")

    df = bars.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["amount"] = df["volume"].astype(float) * df[list(PRICE_COLS)].mean(axis=1)

    df = df[OUT_COLS].astype(float)
    df = df.dropna(subset=list(PRICE_COLS))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df) < int(min_bars):
        raise ValueError(f"有效 K 线不足：仅 {len(df)} 根，至少需要 {min_bars} 根")
    if max_context and len(df) > int(max_context):
        df = df.iloc[-int(max_context):]
    return df


def future_index(last_index: pd.DatetimeIndex, n: int,
                 freq: str | pd.Timedelta | None = None) -> pd.DatetimeIndex:
    """外推未来 n 个时间戳。

    默认用**已有索引间隔的中位数**，而不是硬编码「1 天」：
    同一套代码要服务 1m/5m/日线/月线，中位数是唯一不写死周期的办法。
    真实交易日历（节假日/停牌）这里不处理 —— 预测只是走势示意，
    时间轴偏一天不影响任何仓位计算。
    """
    n = int(n)
    if n <= 0:
        return pd.DatetimeIndex([])
    if freq is not None:
        step = pd.Timedelta(freq) if not isinstance(freq, pd.Timedelta) else freq
    elif len(last_index) >= 2:
        deltas = pd.Series(last_index).diff().dropna()
        step = deltas.median()
        if pd.isna(step) or step <= pd.Timedelta(0):
            step = pd.Timedelta(days=1)
    else:
        step = pd.Timedelta(days=1)
    return pd.DatetimeIndex([last_index[-1] + step * (i + 1) for i in range(n)])


def path_to_ohlcv(pred_close: pd.Series, *, last_close: float,
                  sigma: float = 0.0, last_volume: float = 0.0) -> pd.DataFrame:
    """由「收盘价路径」补出 OHLCV，供统计基线使用。

    基线模型只会给一条收盘价曲线。为了让前端能用同一张图画出来，
    这里按历史波动率撑开 high/low：``high = close*(1+sigma)``。

    注意 `open` 用**上一根的收盘**，不是本根收盘 —— 否则每根都是
    「开=收」，画 K 线会是无效的十字星，回测里按 open 成交也会失真。
    """
    close = pd.Series(pred_close).astype(float)
    if close.empty:
        return pd.DataFrame(columns=OUT_COLS)

    vol = max(float(sigma), 1e-6)
    prev = [float(last_close)] + [float(x) for x in close.iloc[:-1]]
    opens = pd.Series(prev, index=close.index, dtype=float)
    high = pd.concat([opens, close], axis=1).max(axis=1) * (1.0 + vol)
    low = pd.concat([opens, close], axis=1).min(axis=1) * (1.0 - vol)
    vol_col = pd.Series([float(last_volume)] * len(close), index=close.index, dtype=float)
    amount = vol_col * (high + low + close * 2.0) / 4.0

    return pd.DataFrame({
        "open": opens,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol_col,
        "amount": amount,
    })


class BaseForecaster:
    """预测器基类：子类实现 `status()` 与 `forecast()`。"""

    key: str = "base"
    label: str = "基类"
    kind: str = "stat"  # stat（零依赖）| dl（本地权重）| external（外部模型）

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = dict(params or {})

    # ---- 能力 ----
    def status(self) -> dict[str, Any]:
        raise NotImplementedError

    def forecast(self, bars: pd.DataFrame, pred_len: int) -> pd.DataFrame:
        raise NotImplementedError

    # ---- 便捷 ----
    def expected_return(self, bars: pd.DataFrame, pred_len: int) -> float | None:
        """预测末端收盘相对当前收盘的涨跌幅。

        集成层的核心量：不同模型输出维度不同（OHLC 路径 / 分位数 / 采样），
        统一折算成「一个数」才能加权。失败返回 None，不让单个模型拖垮整个集成。
        """
        pred = self.forecast(bars, pred_len)
        if pred is None or pred.empty:
            return None
        try:
            last = float(bars["close"].astype(float).iloc[-1])
            end = float(pred["close"].astype(float).iloc[-1])
        except Exception:  # noqa: BLE001
            return None
        if last <= 0 or not np.isfinite(end):
            return None
        return end / last - 1.0
