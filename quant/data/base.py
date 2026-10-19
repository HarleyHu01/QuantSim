"""DataProvider 抽象层：统一 OHLCV 格式、健康检查、fallback 链。

统一数据格式约定：
- 列: datetime / open / high / low / close / volume / adj_factor
- index: DatetimeIndex（UTC，tz-aware），bar 时间 = 开盘时间
- 已排序、去重、无 NaN 关键列
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import pandas as pd

from quant.core.models import ALL_COLUMNS
from quant.core.utils import ensure_utc_index, validate_ohlcv


@dataclass
class ProviderHealth:
    """数据源健康状态（独立健康检查结果）。

    latency_ms 始终记录真实网络请求耗时（命中 TTL 缓存时保留上次真实值，
    通过 cached / cached_age_seconds / checked_at 标注）。
    """

    healthy: bool
    latency_ms: float | None = None
    error: str | None = None
    source: str | None = None  # 实际使用的通道（如 A股：东方财富/腾讯/新浪）
    checked_at: float | None = None  # 检查完成时间（epoch 秒）
    cached: bool = False  # True = 本次为 TTL 缓存结果
    cached_age_seconds: float | None = None  # 缓存已存在秒数
    # 通道级明细（只有多通道的 provider 才填，如 akshare 的 东财/腾讯/新浪）。
    # 每项：{name, healthy, latency_ms, error, breaker, preferred}
    # 关键：latency_ms 是**该通道自己**的耗时，不是累计耗时——否则
    # 「先试东财失败（等满超时）再试新浪成功」会把东财的超时算到新浪头上。
    channels: list[dict] | None = None

    def to_dict(self) -> dict:
        return {"healthy": self.healthy, "latency_ms": self.latency_ms,
                "error": self.error, "source": self.source,
                "checked_at": self.checked_at, "cached": self.cached,
                "cached_age_seconds": self.cached_age_seconds,
                "channels": self.channels}


def empty_ohlcv() -> pd.DataFrame:
    """统一的空 OHLCV DataFrame。"""
    idx = pd.DatetimeIndex([], tz="UTC", name="datetime")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in ALL_COLUMNS}, index=idx)


def _split_a_symbol(symbol: str) -> tuple[str | None, str | None]:
    """拆 A股 symbol → (6 位代码, 市场后缀 SH/SZ/BJ 或 None)；非 A股格式返回 (None, None)。"""
    s = symbol.strip().upper()
    if "." in s:
        code, suffix = s.split(".")[:2]
        if code.isdigit() and len(code) == 6 and suffix in ("SH", "SZ", "BJ"):
            return code, suffix
        return None, None
    if s.isdigit() and len(s) == 6:
        return s, None
    return None, None


def _is_etf_code(code: str) -> bool:
    """ETF 代码段：51/56/58 开头（沪市）、15/16 开头（深市）。"""
    return code.startswith(("51", "56", "58", "15", "16"))


def _is_index_code(code: str, suffix: str | None) -> bool:
    """指数代码段：399xxx（深证系列）或 000xxx（上证系列，但 000xxx.SZ 是深主板股票）。"""
    if code.startswith("399"):
        return True
    return code.startswith("000") and suffix != "SZ"


def infer_symbol_type(symbol: str) -> tuple[str, str]:
    """根据 symbol 格式推断 (asset_type, source)。

    - 6 位 A股代码（可带 .SH/.SZ）：
      - 51/56/58（沪）、15/16（深）开头 → etf / baostock
      - 399xxx 或 000xxx（非 .SZ）→ index / baostock（指数仅日/周/月线）
      - 其他 → stock / baostock
    - 其他（AAPL、TSLA、NVDA、SPY）→ stock / yfinance

    说明：已移除加密货币（ccxt）支持。含 "/" 的代码（如 BTC/USDT）不再被识别，
    会落到 yfinance 分支并因查不到标的而失败。
    """
    s = symbol.strip().upper()
    code, suffix = _split_a_symbol(s)
    if code is None:
        return "stock", "yfinance"
    if _is_etf_code(code):
        return "etf", "baostock"
    if _is_index_code(code, suffix):
        return "index", "baostock"
    return "stock", "baostock"


def is_domestic(asset_type: str, source: str | None, symbol: str) -> bool:
    """是否为国内 A股体系（决定走 baostock→akshare→腾讯 还是 yfinance）。

    判断依据优先级：
    1. 显式 source：``yfinance`` → 国外；``baostock``/``akshare``/``tencent`` → 国内
    2. symbol 本身是 6 位 A股代码 → 国内
    3. 其余看 asset_type：stock/etf/index 都按国内链（自定义数据源如测试用 ``fake``
       依赖这条，否则会被错误地送进 us_stock 链而丢掉已注册的数据源）
    """
    if source == "yfinance":
        return False
    if source in ("baostock", "akshare", "tencent"):
        return True
    code, _ = _split_a_symbol(symbol)
    if code is not None:
        return True
    return asset_type in ("stock", "etf", "index")


class DataProvider(ABC):
    """数据源抽象接口。

    所有实现必须：
    - 声明 name / display_name / asset_types
    - fetch_history 返回统一格式 OHLCV（UTC DatetimeIndex，
      列 open/high/low/close/volume/adj_factor，已排序去重）
    """

    name: ClassVar[str] = "base"
    display_name: ClassVar[str] = "Base"
    asset_types: ClassVar[tuple[str, ...]] = ()

    # ---- 健康检查（每个数据源独立） ----
    def health_check(self) -> ProviderHealth:
        t0 = time.perf_counter()
        try:
            self._ping()
            return ProviderHealth(healthy=True, latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as e:  # noqa: BLE001
            return ProviderHealth(healthy=False, error=f"{type(e).__name__}: {e}")

    def _ping(self) -> None:
        """网络探测；子类可选覆写。"""
        raise NotImplementedError

    # ---- 行情拉取 ----
    @abstractmethod
    def fetch_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
        adjust: str = "none",
    ) -> pd.DataFrame:
        """拉取历史 OHLCV 并返回统一格式 DataFrame；无数据返回空 DataFrame。"""

    def fetch_realtime(self, symbol: str) -> dict | None:
        """最新行情（可选实现）：{price, bid, ask, ts} 或 None。"""
        return None

    # ---- 能力声明（DataManager 据此跳过不支持组合 / 裁剪起止区间） ----
    def supports(self, symbol: str, timeframe: str) -> bool:
        """该数据源能否提供 (symbol, timeframe)。默认 True，子类按需覆写。"""
        return True

    def max_lookback_days(self, symbol: str, timeframe: str) -> int | None:
        """该数据源对该周期最多能回溯多少天；None = 不额外限制。"""
        return None

    # ---- 工具 ----
    def _finalize(self, df: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
        """子类把原始数据整理成统一格式：小写列名 → UTC index → 补 adj_factor → 清洗。"""
        if df is None or df.empty:
            return empty_ohlcv()
        out = df.copy()
        out.columns = [str(c).strip().lower() for c in out.columns]
        out = ensure_utc_index(out, tz)
        if "adj_factor" not in out.columns:
            out["adj_factor"] = 1.0
        return validate_ohlcv(out)
