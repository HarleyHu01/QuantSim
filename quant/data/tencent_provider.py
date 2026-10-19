"""腾讯行情数据源：A股 / ETF / 指数 的**最后一级兜底**渠道。

为什么单独成一个 Provider（而不是像原来那样塞在 akshare 内部）：
- 它走的是**两个不同端点**，能力边界完全不同，必须分别处理：
  - 日/周/月线 → ``web.ifzq.gtimg.cn/appstock/app/fqkline/get``，**支持按 end 回退分页**
  - 分钟线   → ``ifzq.gtimg.cn/appstock/app/kline/mkline``，**没有历史参数，只能取最近 640 根**
- 实测（2026-10 校验）关键事实：
  - ``fqkline/get`` 传 ``m5``/``m15``/``m30``/``m60`` 一律返回 ``{"code":1,"msg":"bad params"}``
    —— 旧实现把分钟线也塞进 fqkline，是「下载不下来」的一个真实原因
  - ``mkline`` 单次上限 = 640 根（cnt=1000/2000/5000 仍只回 640）
  - 两个端点的行字段顺序都是 **open, close, high, low**（不是 OHLC！），按 OHLC 解析会把
    high/low 读反，回测的止损/波动率全错
  - 成交量单位是「手」，需 ×100 转「股」（与 baostock 的「股」对齐）

定位：**兜底**。默认只在 baostock / akshare（或 yfinance）都失败后才被尝试，
因此它的实现优先保证「能拿到就别报错」，而不是覆盖度。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import ClassVar

import pandas as pd
import requests

from quant.data.base import DataProvider

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
REFERER = "https://gu.qq.com/"

# 分钟线端点（mkline）
_MINUTE_TF = {"1m": "m1", "5m": "m5", "15m": "m15", "30m": "m30", "1h": "m60", "60m": "m60"}
# 日线端点（fqkline）
_DAILY_TF = {"1d": "day", "1w": "week", "1M": "month"}

MAX_BARS = 640          # 两个端点共同的单次上限（实测）
MAX_PAGES = 30          # 日/周/月分页上限，防止异常情况死循环
_MINUTE_MAX_BARS = 640  # 分钟线只能拿到最近这么多根


def _to_tx_code(symbol: str) -> str:
    """symbol → 腾讯代码。600519.SH → sh600519；000001.SZ → sz000001；AAPL → 抛错。"""
    s = symbol.strip().upper()
    if "." in s:
        code, suffix = s.split(".")[:2]
        if suffix == "SH":
            return f"sh{code}"
        if suffix == "SZ":
            return f"sz{code}"
        raise ValueError(f"腾讯行情不支持该市场后缀: {symbol}（仅支持 .SH/.SZ）")
    if s.isdigit() and len(s) == 6:
        if s.startswith(("51", "56", "58", "5", "6", "9")):
            return f"sh{s}"
        if s.startswith(("399", "15", "16", "0", "3")):
            return f"sz{s}"
        raise ValueError(f"腾讯行情无法推断市场: {symbol}（请加 .SH / .SZ 后缀）")
    raise ValueError(f"腾讯行情仅支持沪深 A股/ETF/指数: {symbol}")


def _get_json(url: str, params: dict, timeout: float = 15.0,
              proxy_mode: str = "tencent") -> dict:
    """GET 一个腾讯行情 JSON 端点。

    走 ``requests``（项目其余数据源同款客户端），这样代理环境变量、连接池与
    NO_PROXY 行为与 akshare 通道保持一致；超时/网络错误由调用方捕获后切换到下一条链路。

    代理：按 ``proxy_mode`` 指定的渠道读模式（direct=直连 / proxy=走代理 / auto=跟随系统），
    由 quant.data.proxy 统一决定。腾讯现在是 akshare 内部的一条通道，
    因此由 akshare 调用时传 ``proxy_mode="akshare"``，沿用「akshare」这一档代理设置，
    不再单独占一个代理开关。
    """
    from quant.data.proxy import build_requests_proxies, get_mode

    resp = requests.get(url, params=params, timeout=timeout,
                        headers={"User-Agent": UA, "Referer": REFERER},
                        proxies=build_requests_proxies(get_mode(proxy_mode)))
    resp.raise_for_status()
    return resp.json()


def _parse_dt(v) -> pd.Timestamp:
    """腾讯时间：日线 '2024-02-26'；分钟线 '202610161500'。"""
    s = str(v).strip()
    if len(s) >= 12 and s.isdigit():
        return pd.Timestamp(datetime.strptime(s[:12], "%Y%m%d%H%M"))
    return pd.Timestamp(s)

def _rows_to_df(rows: list[list]) -> pd.DataFrame:
    """腾讯行数组 → OHLCV（注意字段序：date/open/close/high/low/volume）。"""
    out = []
    for r in rows:
        if not isinstance(r, list) or len(r) < 6:
            continue
        try:
            out.append({
                "datetime": _parse_dt(r[0]),
                "open": float(r[1]),
                "close": float(r[2]),
                "high": float(r[3]),
                "low": float(r[4]),
                "volume": float(r[5]) * 100.0,  # 手 → 股
            })
        except (ValueError, TypeError):
            continue
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).set_index("datetime")
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


# ================= 可复用的取数原语 =================
# 这三个函数把「腾讯取数」从 Provider 类里剥离出来，因为腾讯现在同时被两处使用：
#   1. TencentProvider（独立 provider，保留给直接引用它的调用方 / 测试）
#   2. akshare 内部的「腾讯通道」（见 quant/data/akshare_provider.py::_channel_tencent）
# 两条路径必须走**完全同一份**解析逻辑，否则字段序/成交量单位这类坑会重复踩。

def _fetch_minute_rows(tx: str, timeframe: str, timeout: float = 15.0) -> list[list]:
    """分钟线原始行：mkline 端点，无历史区间参数，只返回最近 640 根。

    > 2026-10 修正：这里必须用 ``ifzq.gtimg.cn``（无 web 前缀）。
    > ``web.ifzq.gtimg.cn`` 对 mkline 会 **301 重定向到 web3.ifzq.gtimg.cn**，
    > 而 web3 主机不可达（ConnectionError），导致分钟线全部失败。
    > 直接请求 ``ifzq.gtimg.cn`` 返回 200 且各周期均能拿到 640 根。
    """
    tf = _MINUTE_TF[timeframe]
    payload = _get_json("https://ifzq.gtimg.cn/appstock/app/kline/mkline",
                        {"param": f"{tx},{tf},,{_MINUTE_MAX_BARS}"}, timeout=timeout,
                        proxy_mode="akshare")
    node = (payload.get("data") or {}).get(tx) or {}
    return node.get(tf) or []


def _fetch_daily_rows(tx: str, timeframe: str, start: datetime, end: datetime,
                      adjust: str, timeout: float = 15.0) -> list[list]:
    """日/周/月线原始行：fqkline 端点，640 根/次，按 end 回退分页拼全区间。"""
    tf = _DAILY_TF[timeframe]
    adj = adjust if adjust in ("qfq", "hfq") else "qfq"
    rows_all: list[list] = []
    cursor = end
    floor = pd.Timestamp(start)
    for _ in range(MAX_PAGES):
        payload = _get_json(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            {"param": f"{tx},{tf},,{cursor:%Y-%m-%d},{MAX_BARS},{adj}"},
            timeout=timeout, proxy_mode="akshare",
        )
        node = (payload.get("data") or {}).get(tx) or {}
        rows: list[list] = []
        for key in (f"{adj}{tf}", f"qfq{tf}", f"hfq{tf}", tf, "day"):
            val = node.get(key)
            if isinstance(val, list) and val:
                rows = val
                break
        if not rows:
            break
        rows_all.extend(rows)
        page = _rows_to_df(rows)
        if page.empty:
            break
        first = page.index.min()
        if first <= floor:
            break  # 已覆盖到请求起点，无需再往前翻
        cursor = first.to_pydatetime() - timedelta(days=1)  # 回退一页继续取更早
    return rows_all


class TencentProvider(DataProvider):
    """腾讯行情（A股最后兜底）：日/周/月线可分页，分钟线仅最近 640 根。"""

    name: ClassVar[str] = "tencent"
    display_name: ClassVar[str] = "腾讯行情（A股兜底）"
    asset_types: ClassVar[tuple[str, ...]] = ("stock", "etf", "index")

    def __init__(self):
        self.last_source: str | None = None

    # ---- 能力声明（DataManager 用它跳过不支持的组合，避免无谓请求） ----
    def supports(self, symbol: str, timeframe: str) -> bool:
        if timeframe not in _MINUTE_TF and timeframe not in _DAILY_TF:
            return False
        try:
            _to_tx_code(symbol)
        except ValueError:
            return False
        return True

    def max_lookback_days(self, symbol: str, timeframe: str) -> int | None:
        """分钟线只能覆盖最近 640 根；日/周/月线靠分页，足够深（返回 None = 不额外限制）。"""
        if timeframe in _MINUTE_TF:
            bars_per_day = {"1m": 240, "5m": 48, "15m": 16, "30m": 8, "1h": 4, "60m": 4}[timeframe]
            return max(1, int(_MINUTE_MAX_BARS / bars_per_day) + 2)
        return None

    # ---- 拉取 ----
    def _rows_to_df(self, rows: list[list]) -> pd.DataFrame:
        """腾讯行数组 → OHLCV（注意字段序：date/open/close/high/low/volume）。"""
        return _rows_to_df(rows)

    def fetch_history(self, symbol, start, end, timeframe="1d", adjust="none") -> pd.DataFrame:
        tx = _to_tx_code(symbol)
        if timeframe in _MINUTE_TF:
            df = self._fetch_minute(tx, timeframe)
        elif timeframe in _DAILY_TF:
            df = self._fetch_daily(tx, timeframe, start, end, adjust)
        else:
            raise ValueError(f"腾讯行情不支持的周期: {timeframe}")
        if df is None or df.empty:
            return df
        # 腾讯返回的是本地时间（tz-naive）→ 补上海时区后再裁剪，避免 tz-aware/naive 比较报错
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Shanghai")
        lo = pd.Timestamp(start)
        if lo.tz is None:
            lo = lo.tz_localize("Asia/Shanghai")
        hi = pd.Timestamp(end) + pd.Timedelta(days=1)
        if hi.tz is None:
            hi = hi.tz_localize("Asia/Shanghai")
        df = df[(df.index >= lo) & (df.index <= hi)]
        self.last_source = "tencent"
        return self._finalize(df, tz="Asia/Shanghai")

    def _fetch_minute(self, tx: str, timeframe: str) -> pd.DataFrame:
        """分钟线：mkline 端点，无历史区间参数，只返回最近 640 根。"""
        return _rows_to_df(_fetch_minute_rows(tx, timeframe))

    def _fetch_daily(self, tx: str, timeframe: str, start: datetime, end: datetime,
                     adjust: str) -> pd.DataFrame:
        """日/周/月线：fqkline 端点，640 根/次，按 end 回退分页拼全区间。"""
        rows = _fetch_daily_rows(tx, timeframe, start, end, adjust)
        if not rows:
            return pd.DataFrame()
        df = _rows_to_df(rows)
        # 分页边界可能重叠同一根 bar → 去重后按时间排序
        return df[lambda d: ~d.index.duplicated(keep="last")].sort_index()

    # ---- 健康检查 ----
    def _ping(self) -> None:
        now = datetime.now()
        df = self._fetch_daily("sh600519", "1d", now - timedelta(days=5), now, "qfq")
        if df is None or df.empty:
            raise RuntimeError("腾讯行情健康检查返回空数据")
        self.last_source = "tencent"

    def health_check(self):
        h = super().health_check()
        h.source = self.last_source
        return h
