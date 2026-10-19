"""yfinance 数据源：美股 / ETF（Yahoo Finance）。

主路径使用 yfinance 客户端；若 Yahoo 对客户端限流（429），自动回退到
原生 Yahoo chart API（query2，带浏览器 UA），保证数据源韧性。

**Yahoo 各 interval 的硬限制（2026-10 实测，超出直接 HTTP 422）**：
- ``1m``  ：单次请求最多 8 天
- ``5m``/``15m``/``30m``/``90m``：只能取最近 60 天
- ``60m``（=1h）：只能取最近 730 天
- **``2h``/``4h`` Yahoo 根本不支持**（400 / 422）→ 已从 TF_MAP 移除
- ``1d``/``1wk``/``1mo``：无历史上限（可回溯到上市首日）
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import ClassVar

import pandas as pd

from quant.data.base import DataProvider, empty_ohlcv

try:  # 可选依赖，未安装时该数据源自动降级
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

# 系统周期 → yfinance interval（只保留 Yahoo 真实支持的粒度）
TF_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m",
    "1d": "1d", "1w": "1wk", "1M": "1mo",
}

# 各 interval 单次请求最多能取多少天（Yahoo 服务端硬限制，超过报 422）
_INTERVAL_MAX_DAYS = {
    "1m": 8,
    "5m": 60, "15m": 60, "30m": 60, "90m": 60,
    "60m": 730,
}
# Yahoo 分钟级单次返回上限（超过需分页）
_MAX_BARS = 500

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


class YFinanceProvider(DataProvider):
    name: ClassVar[str] = "yfinance"
    display_name: ClassVar[str] = "yFinance（美股/ETF）"
    asset_types: ClassVar[tuple[str, ...]] = ("stock", "etf")

    def __init__(self, cache_dir: str | None = None):
        """cache_dir: yfinance 1.6+ 的 SQLite 缓存目录（默认 %LOCALAPPDATA%，受限环境需指定）。"""
        self.cache_dir = cache_dir

    # ---- 能力声明 ----
    def supports(self, symbol: str, timeframe: str) -> bool:
        return timeframe in TF_MAP

    def max_lookback_days(self, symbol: str, timeframe: str) -> int | None:
        """Yahoo 分钟线有硬窗口；日/周/月不限。"""
        return _INTERVAL_MAX_DAYS.get(TF_MAP.get(timeframe, ""))

    def _ensure_cache(self) -> None:
        """把 yfinance 的 tz/cookie/isin 缓存指到可写目录，并调高 429 重试。"""
        if yf is None:
            return
        try:
            if self.cache_dir:
                from pathlib import Path
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                from yfinance.cache import set_cache_location
                set_cache_location(self.cache_dir)
            yf.config.network.retries = 5  # Yahoo 429 限流时自动重试
        except Exception:  # noqa: BLE001
            pass

    def _ping(self) -> None:
        """轻量健康探测：原生 chart API 请求 1 根日线 bar（真实网络，不走本地缓存）。"""
        now = datetime.now()
        self._fetch_raw_chart("AAPL", now - timedelta(days=2), now, "1d")

    def fetch_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
        adjust: str = "none",
    ) -> pd.DataFrame:
        # 按 Yahoo 的窗口限制裁剪起点，否则服务端直接 422
        start = self._clip_start(start, end, timeframe)
        if yf is not None:
            self._ensure_cache()
            interval = TF_MAP.get(timeframe)
            if interval is not None:
                try:
                    df = yf.Ticker(symbol, session=self._make_session()).history(
                        start=start.strftime("%Y-%m-%d"),
                        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        interval=interval,
                        auto_adjust=False,
                    )
                    if df is not None and not df.empty:
                        return self._from_yf_df(df)
                except Exception:  # noqa: BLE001
                    pass  # 限流/异常 → 走原生兜底
        return self._fetch_raw_chart(symbol, start, end, timeframe)

    def _make_session(self):
        """按 yfinance 渠道的代理模式构造 requests.Session（direct/proxy/auto）。"""
        from quant.data.proxy import get_mode, make_requests_session

        return make_requests_session(get_mode("yfinance"))

    def _make_opener(self):
        """按 yfinance 渠道的代理模式构造 urllib opener（None=跟随系统）。"""
        from quant.data.proxy import build_urllib_opener, get_mode

        return build_urllib_opener(get_mode("yfinance"))

    def _clip_start(self, start: datetime, end: datetime, timeframe: str) -> datetime:
        """把起点裁进 Yahoo 允许的窗口内（避免必然 422 的请求）。"""
        max_days = _INTERVAL_MAX_DAYS.get(TF_MAP.get(timeframe, ""))
        if not max_days:
            return start
        earliest = end - timedelta(days=max_days - 1)
        return start if start > earliest else earliest

    # ---- yfinance 客户端结果 → 统一格式 ----
    def _from_yf_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.rename(
            columns={"Open": "open", "High": "high", "Low": "low",
                     "Close": "close", "Volume": "volume"}
        )
        if "Adj Close" in out.columns:
            close = out["close"].replace(0, pd.NA)
            out["adj_factor"] = (out["Adj Close"] / close).fillna(1.0)
        return self._finalize(out)

    # ---- 原生 Yahoo chart API 兜底（浏览器 UA，绕过客户端限流；分页 >500 根） ----
    def _fetch_raw_chart(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        interval = TF_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"yfinance 不支持的周期: {timeframe}（支持 {sorted(TF_MAP)}）")
        frames: list[pd.DataFrame] = []
        period1 = int(start.timestamp())
        period2 = int(end.timestamp())
        while period1 < period2:
            url = (
                "https://query2.finance.yahoo.com/v8/finance/chart/"
                f"{urllib.parse.quote(symbol)}?period1={period1}&period2={period2}"
                f"&interval={interval}&events=history"
            )
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            opener = self._make_opener()
            if opener is not None:
                resp = opener.open(req, timeout=30)  # noqa: S310
            else:
                resp = urllib.request.urlopen(req, timeout=30)  # noqa: S310
            with resp:
                payload = json.loads(resp.read().decode("utf-8"))
            result = payload["chart"]["result"]
            if not result:
                break
            r = result[0]
            ts = r.get("timestamp") or []
            if not ts:
                break
            quote = r["indicators"]["quote"][0]
            df = pd.DataFrame(
                {"open": quote.get("open"), "high": quote.get("high"),
                 "low": quote.get("low"), "close": quote.get("close"),
                 "volume": quote.get("volume")},
                index=pd.to_datetime(ts, unit="s", utc=True),
            )
            adjclose = r["indicators"].get("adjclose", [{}])[0].get("adjclose")
            if adjclose:
                df["adj_factor"] = adjclose / df["close"].replace(0, pd.NA)
            frames.append(df)
            last_ts = ts[-1]
            if len(ts) < _MAX_BARS or last_ts >= period2 - 86400:
                break
            period1 = last_ts + 1  # 继续下一页
        if not frames:
            return empty_ohlcv()
        return self._finalize(pd.concat(frames))  # 去重/排序在 _finalize 内
