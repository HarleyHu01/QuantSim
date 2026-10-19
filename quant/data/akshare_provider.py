"""akshare 数据源：A股 / ETF / 指数 —— **第二级渠道**（备选）。

双通道（**东财优先，新浪兜底**）：
1. 东方财富 ``push2his.eastmoney.com`` —— 主通道，日/周/月 + 5/15/30/60 分钟，
   带 Referer 与浏览器 UA；每轮最多 2 次尝试，指数退避 2s/4s/8s。
2. 新浪 —— 兜底通道，经 ``akshare.stock_zh_a_daily``（日/周/月，支持 qfq/hfq）
   与 ``akshare.stock_zh_a_minute``（5/15/30/60 分钟）。东财被运营商或防火墙
   阻断时（TCP 443 与 TLS 握手都通，但 HTTP 层被服务端主动
   断连），由这条通道接住，避免整条 akshare 渠道全废。

> 2026-10 修正（第一次，已推翻）：曾把新浪子通道移除，理由是「渠道名单只有
> **baostock → akshare → 腾讯** 三家，新浪不在其中」。实践发现该决定有问题：
> ``push2his.eastmoney.com`` 是**单点**，域名一旦被阻断 akshare 就彻底失效，
> 而腾讯分钟线又受 640 根上限约束 → 分钟级数据无人供给。现恢复新浪为
> **同一 provider 内的兜底通道**（仍算「akshare 渠道」，不新增第四家 provider，
> 不改变 provider 优先级链，也不改变「腾讯固定在链尾」的约束）。

腾讯已独立成 ``quant/data/tencent_provider.py``，作为**最后一级兜底**渠道，
由 DataManager 的 provider 优先级链统一调度；它不在本 Provider 内部了。

历史踩坑（2026-10 校验）：
- 旧实现把腾讯分钟线塞进 ``fqkline/get``，该端点只认 day/week/month，
  传 m5/m15/m30/m60 一律 ``{"code":1,"msg":"bad params"}`` —— 已移除。
- 旧实现默认把国内域名加入 NO_PROXY（``_disable_proxy``），实测在这里
  **反而导致东财 RemoteDisconnected / 腾讯返回非 JSON**，因为该环境必须走系统代理。
  现已改为「不主动改代理环境变量」，只在显式配置 ``akshare_use_proxy=False`` 时跳过。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import ClassVar
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from quant.data.base import DataProvider

_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# 周期 → 东财 klt（无 1m 与 2h/4h）
_KLT = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60,
        "1d": 101, "1w": 102, "1M": 103}  # 东财 klt（1m 只有腾讯腿支持，见 _channels）
_FQT = {"qfq": 1, "hfq": 2, "none": 0}  # 东财 fqt

# 腾讯腿的周期集合（与 tencent_provider 的 _MINUTE_TF/_DAILY_TF 保持一致）
_TENCENT_MINUTE = {"1m", "5m", "15m", "30m", "1h", "60m"}
_TENCENT_DAILY = {"1d", "1w", "1M"}

_REQUEST_TIMEOUT = 10.0       # 单次请求超时（秒）
_HEALTH_TIMEOUT = 3.0         # 健康检查单通道超时
_MIN_INTERVAL = 1.0           # 批量请求最小间隔（秒）
_PAUSE_AFTER = 3              # 连续失败 N 次后暂停
_PAUSE_SECONDS = 30.0
_BACKOFF = [2.0, 4.0, 8.0]    # 指数退避
_CHANNEL_COOLDOWN = 300.0     # 通道「网络级不可达」后的熔断时长（秒）
# A股本地时区：三个通道解析出的 index 都是 naive 北京时间，请求边界可能带 UTC 时区，
# 比较前必须换算到同一口径（见 _clip_to_window 的 docstring）。
_LOCAL_TZ = ZoneInfo("Asia/Shanghai")

_lock = threading.Lock()
_last_request_ts = 0.0
_consecutive_failures = 0
# 通道熔断表：{通道标签: 恢复时间(monotonic)}。
# 某条通道被网络层阻断时，若每次拉取都先试它两轮再退避，
# 单次下载会白白多花 6~40 秒（连续失败 3 次还要 sleep 30s），表现为「akshare 完全不可用」。
# 因此把**传输层**失败（连不上/被断连/超时/代理拒绝）视为通道不可用，
# 熔断期内直接跳过往下一通道；退避 sleep 只留给限流类错误（429/5xx）。
_channel_down_until: dict[str, float] = {}


def _mark_channel_down(label: str) -> None:
    with _lock:
        _channel_down_until[label] = time.monotonic() + _CHANNEL_COOLDOWN


def _is_channel_down(label: str) -> bool:
    with _lock:
        until = _channel_down_until.get(label)
    return until is not None and time.monotonic() < until


def _clear_channel_down(label: str) -> None:
    with _lock:
        _channel_down_until.pop(label, None)


_CHANNEL_LABELS = ("东方财富", "腾讯", "新浪")


def channel_status() -> dict[str, str]:
    """当前通道熔断状态（供健康检查/UI 展示）：{"东方财富": "cooling 274s"} 之类。"""
    out = {}
    with _lock:
        now = time.monotonic()
        for label in _CHANNEL_LABELS:
            until = _channel_down_until.get(label)
            if until is None:
                out[label] = "ok"
            else:
                left = int(until - now)
                out[label] = f"cooling {left}s" if left > 0 else "ok"
    return out


def _is_transport_error(exc: Exception) -> bool:
    """传输层失败（不是限流）：换通道比原地退避更有意义。"""
    transport = (requests.exceptions.ConnectionError,
                 requests.exceptions.Timeout,
                 requests.exceptions.ProxyError,
                 requests.exceptions.SSLError,
                 ConnectionError, TimeoutError)
    if isinstance(exc, transport):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(k in text for k in ("remotedisconnected", "connection aborted",
                                   "connection refused", "connection reset",
                                   "max retries exceeded", "timed out",
                                   "proxy", "getaddrinfo", "ssl"))


def _throttle() -> None:
    """批量拉取频控：请求间隔 ≥1s。"""
    global _last_request_ts
    with _lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.monotonic()


def _record_failure() -> float:
    """记录一次失败，返回应等待的秒数（指数退避 / 连续失败暂停）。"""
    global _consecutive_failures
    with _lock:
        _consecutive_failures += 1
        n = _consecutive_failures
    if n % _PAUSE_AFTER == 0:
        return _PAUSE_SECONDS
    return _BACKOFF[min(n - 1, len(_BACKOFF) - 1)]


def _record_success() -> None:
    global _consecutive_failures
    with _lock:
        _consecutive_failures = 0


def _normalize_a_symbol(symbol: str) -> tuple[str, str, str]:
    """(6位代码, 市场前缀, 东财secid)。000001.SZ → ("000001","sz","0.000001")。"""
    symbol = symbol.strip().upper()
    if "." in symbol:
        code, suffix = symbol.split(".")[:2]
        if suffix in ("SH", "BJ"):
            return code, "sh", f"1.{code}" if suffix == "SH" else f"0.{code}"
        return code, "sz", f"0.{code}"
    code = symbol
    if code.startswith(("5", "6", "9")):
        return code, "sh", f"1.{code}"
    return code, "sz", f"0.{code}"


def _http_get(url: str, params: dict, referer: str, timeout: float,
              throttle: bool = True) -> requests.Response:
    """带浏览器 UA + 对应站 Referer 的 GET；健康检查可跳过频控（throttle=False）。

    代理：按当前 ``akshare`` 渠道的代理模式传 ``proxies``（direct=直连 / proxy=走代理
    / auto=跟随系统），由 quant.data.proxy 统一决定，实现逐渠道独立代理。
    """
    if throttle:
        _throttle()
    return requests.get(
        url, params=params, timeout=timeout,
        headers={"User-Agent": _BROWSER_UA, "Referer": referer,
                 "Accept": "application/json, text/plain, */*",
                 "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        proxies=_proxy_for_akshare(),
    )


def _proxy_for_akshare() -> dict[str, str | None] | None:
    """akshare 渠道的代理参数（延迟导入避免循环依赖）。"""
    from quant.data.proxy import build_requests_proxies, get_mode

    return build_requests_proxies(get_mode("akshare"))


def _clip_to_window(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """把通道返回的数据裁到请求区间内。

    为什么必须裁：腾讯的 ``mkline``/``fqkline`` 端点**没有区间参数**，总是返回
    「最近 640 根」或「从 cursor 往前 640 根」，并不遵守请求的 start/end。
    实测请求 400 天日线时它回了 640 根（回溯到 2024-02），请求周线时回了
    2014 年起的 640 根。若不裁剪：
      - 增量更新会拿回一大段已有数据（合并层能去重，但白占带宽与内存）
      - 更糟的是 ``UpdateResult.start/end`` 会报出远超请求区间的时间范围
    因此统一在 provider 内裁剪，让所有通道对上层呈现一致的语义。

    **时区必须显式对齐**（踩过的坑，1m 下载全挂的真因）：
    三个通道解析出来的 index 都是 **tz-naive 的北京时间**，而 ``DataManager``
    传进来的 ``start``/``end`` 常常是 **tz-aware（UTC）**。tz-aware Timestamp
    与 tz-naive ``datetime64`` 直接比较会抛
    ``TypeError: Invalid comparison between dtype=datetime64[us] and Timestamp``
    ——因为该 ``TypeError`` 只发生在腾讯这条腿，会被误判成「通道故障」而反复重试。
    修正：naive 索引一律按 ``Asia/Shanghai`` 解释，把请求边界换算到同一时区。
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    idx = df.index
    idx_tz = getattr(idx, "tz", None)
    if idx_tz is None:
        # 索引 tz-naive（= 北京时间）：把边界也降为 naive 北京时间
        if lo.tz is not None:
            lo = lo.tz_convert(_LOCAL_TZ).tz_localize(None)
        if hi.tz is not None:
            hi = hi.tz_convert(_LOCAL_TZ).tz_localize(None)
    else:
        # 索引 tz-aware：把边界对齐到索引的时区
        lo = lo.tz_localize(idx_tz) if lo.tz is None else lo.tz_convert(idx_tz)
        hi = hi.tz_localize(idx_tz) if hi.tz is None else hi.tz_convert(idx_tz)
    return df[(idx >= lo) & (idx <= hi)]


def _df_from_rows(rows: list[list], vol_mult: int = 1) -> pd.DataFrame:
    """统一行结构 [datetime, open, close, high, low, volume, ...] → OHLCV DataFrame。"""
    out = []
    for r in rows:
        if len(r) < 6:
            continue
        try:
            out.append({"datetime": pd.Timestamp(r[0]), "open": float(r[1]),
                        "close": float(r[2]), "high": float(r[3]),
                        "low": float(r[4]), "volume": float(r[5]) * vol_mult})
        except (ValueError, TypeError):
            continue
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).set_index("datetime")
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


# ================= 通道 =================

def _channel_eastmoney(code: str, pref: str, secid: str, start: datetime, end: datetime,
                       timeframe: str, adjust: str, timeout: float = _REQUEST_TIMEOUT,
                       throttle: bool = True) -> pd.DataFrame:
    """东方财富 kline（主通道，带 Referer）。klt=101 日线 / 5/15/30/60 分钟 / 102 周 / 103 月。

    东财无 1m 周期，传 1m 会直接拒绝（由腾讯腿独家承担 1m）。
    """
    if timeframe not in _KLT or timeframe == "1m":
        raise ValueError(f"东财通道不支持的周期: {timeframe}")
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid, "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": _KLT[timeframe], "fqt": _FQT.get(adjust, 1),
        "beg": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
    }
    r = _http_get(url, params, "https://quote.eastmoney.com/", timeout, throttle=throttle)
    data = (r.json() or {}).get("data") or {}
    klines = data.get("klines") or []
    rows = [k.split(",") for k in klines]
    # 东财成交量单位「手」→ ×100 转「股」
    return _df_from_rows(rows, vol_mult=100)


# ---- 新浪兜底通道 ----
# 新浪只认 sh/sz 前缀代码；支持的周期与东财不完全一致（无 1m）
_SINA_MINUTE = {"5m": "5", "15m": "15", "30m": "30", "1h": "60", "60m": "60"}
_SINA_DAILY = {"1d": "daily", "1w": "weekly", "1M": "monthly"}


def _sina_classify(code: str, pref: str) -> str:
    """按代码段判断新浪该用哪个**行情接口**：stock / etf / index。

    新浪的 `stock_zh_a_daily` 只认个股，对 ETF(510300) 与指数(000300) 会抛
    `JSONDecodeError` / `SyntaxError`（实测），必须换用专用接口：
      - ETF   → ``fund_etf_hist_sina(symbol="sh510300")``
      - 指数  → ``stock_zh_index_daily(symbol="sh000300")``
    """
    if pref == "sh":
        if code.startswith(("51", "56", "58", "50", "52")):
            return "etf"
        if code.startswith("000"):
            return "index"
    else:  # sz
        if code.startswith(("15", "16", "18")):
            return "etf"
        if code.startswith("399"):
            return "index"
    return "stock"


def _sina_daily_frame(code: str, pref: str, adjust: str) -> pd.DataFrame:
    """按标的类型选新浪的日线接口，返回原始 DataFrame（含 date 列）。"""
    import akshare as ak

    sym = f"{pref}{code}"
    kind = _sina_classify(code, pref)
    if kind == "etf":
        # ETF 新浪接口无复权参数（新浪 ETF 本身多为不复权/份额调整后价格）
        return ak.fund_etf_hist_sina(symbol=sym)
    if kind == "index":
        # 指数无复权概念
        return ak.stock_zh_index_daily(symbol=sym)
    adj = adjust if adjust in ("qfq", "hfq") else "qfq"
    return ak.stock_zh_a_daily(symbol=sym, adjust=adj)


def _channel_sina(code: str, pref: str, secid: str, start: datetime, end: datetime,
                  timeframe: str, adjust: str, timeout: float = _REQUEST_TIMEOUT,
                  throttle: bool = True) -> pd.DataFrame:
    """新浪 kline（兜底通道）：走 akshare 封装，日/周/月 + 5/15/30/60 分钟。

    标的类型自动分流（个股 / ETF / 指数），因为新浪三类标的是**三个不同接口**。

    注意：
    - ``stock_zh_a_daily`` 的 ``adjust="none"`` 实测返回空表，只有 qfq/hfq 有效，
      因此不复权请求在这里静默退回 qfq。
    - ETF/指数接口无复权概念，忽略 adjust。
    - 返回的 DataFrame 是 RangeIndex + ``date``/``day`` 列，需要自己转成 DatetimeIndex。
    - 成交量单位已经是「股」，不需再 ×100。
    """
    sym = f"{pref}{code}"  # 600519 → sh600519
    if throttle:
        _throttle()

    if timeframe in _SINA_MINUTE:
        import akshare as ak

        # 分钟线只有个股接口支持（ETF/指数统一按个股代码尝试，能取到就用）
        adj = adjust if adjust in ("qfq", "hfq") else "qfq"
        df = ak.stock_zh_a_minute(symbol=sym, period=_SINA_MINUTE[timeframe], adjust=adj)
        return _sina_to_ohlcv(df, "day")

    if timeframe in _SINA_DAILY:
        period = _SINA_DAILY[timeframe]
        daily = _sina_daily_frame(code, pref, adjust)
        if daily is None or daily.empty:
            return pd.DataFrame()
        if period == "daily":
            return _sina_to_ohlcv(daily, "date")
        # 新浪无周/月封装，用日线重采样自算（与东财 klt=102/103 的 OHLC 聚合口径一致）
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.set_index("date")
        rule = "W-FRI" if period == "weekly" else "ME"
        agg = {"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}
        agg = {k: v for k, v in agg.items() if k in daily.columns}
        out = daily.resample(rule).agg(agg).dropna(subset=["open", "close"])
        return _sina_to_ohlcv(out.reset_index().rename(columns={"date": "day"}), "day")

    raise ValueError(f"新浪通道不支持的周期: {timeframe}")


def _channel_tencent(code: str, pref: str, secid: str, start: datetime, end: datetime,
                     timeframe: str, adjust: str, timeout: float = _REQUEST_TIMEOUT,
                     throttle: bool = True) -> pd.DataFrame:
    """腾讯 kline（第二通道）：复用 ``quant/data/tencent_provider.py`` 的取数与解析。

    为什么复用而不是自己写一份：
    - 腾讯两个端点的行字段序是 **date/open/close/high/low/volume**（不是 OHLC），
      成交量单位「手」需 ×100，minute 端点在 ``ifzq.gtimg.cn``（**不能带 web. 前缀**，
      否则 301 到不可达的 web3）。这些坑已经在该模块里踩过并记录，重复实现只会重蹈覆辙。

    能力边界（实测）：
    - 日/周/月 → ``fqkline/get``，640 根/次分页，可回溯到上市首日
    - 分钟线   → ``mkline``，**只能取最近 640 根**，且没有历史区间参数
    - **唯一支持 1m 的源**（akshare 另两条腿都不支持 1m）

    因此 1m 由本通道独家承担；5/15/30/60 分钟在三条腿里都能拿到，腾讯作为中间层，
    前面东财不通时先用它、它也不支持时才落到新浪。
    """
    from quant.data.tencent_provider import (_DAILY_TF, _MINUTE_TF, _fetch_daily_rows,
                                             _fetch_minute_rows, _rows_to_df, _to_tx_code)

    tx = _to_tx_code(f"{code}.{'SH' if pref == 'sh' else 'SZ'}")
    if timeframe in _MINUTE_TF:
        rows = _fetch_minute_rows(tx, timeframe, timeout=timeout)
        df = _rows_to_df(rows)
    elif timeframe in _DAILY_TF:
        rows = _fetch_daily_rows(tx, timeframe, start, end, adjust, timeout=timeout)
        df = _rows_to_df(rows)
    else:
        raise ValueError(f"腾讯通道不支持的周期: {timeframe}")
    if df is None or df.empty:
        return pd.DataFrame()
    # 腾讯返回本地时间（tz-naive）；裁剪后交给 _finalize 统一补时区
    return df


def _sina_to_ohlcv(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """新浪 akshare 返回表 → 统一 OHLCV（DatetimeIndex）。"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if time_col not in out.columns:
        raise ValueError(f"新浪返回缺少时间列 {time_col}: {list(out.columns)}")
    out["datetime"] = pd.to_datetime(out[time_col])
    keep = ["datetime", "open", "high", "low", "close", "volume"]
    out = out[[c for c in keep if c in out.columns]]
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out.set_index("datetime").sort_index()


class AkshareProvider(DataProvider):
    """A股/ETF/指数第二级渠道：仅东方财富（失败即交给链尾的腾讯兜底）。"""

    name: ClassVar[str] = "akshare"
    display_name: ClassVar[str] = "AKShare（A股备选）"
    asset_types: ClassVar[tuple[str, ...]] = ("stock", "etf", "index")

    def __init__(self, use_proxy: bool = True):
        """use_proxy 保留兼容；本类不再改写进程级代理环境变量（见模块 docstring）。"""
        self.use_proxy = use_proxy
        self._patch_requests_ua()
        self.last_source: str | None = None  # 最近一次成功通道（东方财富/腾讯/新浪）
        self.channels: list[dict] | None = None  # 最近一次健康检查的逐通道明细
        self._primary_latency_ms: float | None = None  # 主用通道自己的耗时

    # ---- 能力声明 ----
    def supports(self, symbol: str, timeframe: str) -> bool:
        if timeframe not in _KLT:
            return False
        try:
            code, _, _ = _normalize_a_symbol(symbol)
        except Exception:  # noqa: BLE001
            return False
        return code.isdigit() and len(code) == 6

    def max_lookback_days(self, symbol: str, timeframe: str) -> int | None:
        """1m 只能靠腾讯腿，覆盖约 4 个交易日；其余周期有新浪较深的历史，不额外限制。"""
        if timeframe == "1m":
            return 7
        return None

    _ua_patched = False

    @classmethod
    def _patch_requests_ua(cls) -> None:
        """给 requests 全局注入浏览器 UA（akshare 内部调用也生效）。"""
        if cls._ua_patched:
            return
        cls._ua_patched = True

        def _default_headers():
            return {"User-Agent": _BROWSER_UA,
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}

        try:
            import requests.utils
            requests.utils.default_headers = _default_headers
        except Exception:  # noqa: BLE001
            pass

    # ---- 通道链 ----
    def _channels(self, timeframe: str):
        """可用通道 [(标签, 函数)]，**顺序按周期能力动态决定**。

        三条腿都算「akshare 渠道」，不新增 provider；某条腿整体不可用时由下一条接住，
        全部失败才整体报错、交给别的 provider。

        能力边界（实测，2026-10，600519.SH）：
        - **东财**：日/周/月 + 5/15/30/60 分钟，无 1m。
        - **腾讯**：日/周/月 + 1/5/15/30/60 分钟。**唯一支持 1m 的腿**。
          分钟线只能取最近 640 根（端点无历史参数）；日/周/月可分页回溯到上市首日。
        - **新浪**：日/周/月 + 5/15/30/60 分钟，无 1m。分钟线固定 1970 根。

        **顺序为什么随周期变**——这是「数据广度」的直接体现，不能一刀切：
        - 分钟线(5m~1h)：新浪 1970 根 ≫ 腾讯 640 根（约 3 倍深度），
          所以 **新浪排在腾讯前面**，优先拿更深的历史；腾讯退为真正的兜底。
          若把腾讯放前面，用户会莫名其妙只能拿到 640 根（约 13 个交易日的 5m），
          比原来只有新浪时**反而缩水**。
        - 1m：只有腾讯支持，独占。
        - 日/周/月线：两家都能回溯到 2001 年（腾讯 6022 / 新浪 6023 根），
          深度相当，但腾讯约 0.55s、新浪约 1.6s，**腾讯更快**，故排在前面。
        """
        chans = []
        if timeframe == "1m":
            # 东财、新浪都没有 1m，腾讯独占（也就没有排序问题）
            if timeframe in _TENCENT_MINUTE:
                chans.append(("腾讯", _channel_tencent))
            return chans
        if timeframe != "1m":
            chans.append(("东方财富", _channel_eastmoney))
        if timeframe in _TENCENT_DAILY:
            # 日/周/月：腾讯更快，优先
            chans.append(("腾讯", _channel_tencent))
            if timeframe in _SINA_DAILY:
                chans.append(("新浪", _channel_sina))
        else:
            # 分钟线：新浪历史更深（1970 vs 640），优先新浪
            if timeframe in _SINA_MINUTE:
                chans.append(("新浪", _channel_sina))
            if timeframe in _TENCENT_MINUTE:
                chans.append(("腾讯", _channel_tencent))
        return chans

    def fetch_history(self, symbol, start, end, timeframe="1d", adjust="none") -> pd.DataFrame:
        code, pref, secid = _normalize_a_symbol(symbol)
        if not (code.isdigit() and len(code) == 6):
            # 非 A股/ETF 代码（如美股 AAPL）快速失败，避免无意义的多通道尝试拖慢 fallback 链
            raise ValueError(f"非 A股/ETF 代码（akshare 仅支持沪深 A股/ETF）: {symbol}")
        if timeframe not in _KLT:
            raise ValueError(f"不支持的周期: {timeframe}（akshare 支持 {'/'.join(_KLT)}）")
        errs: list[str] = []
        for label, fn in self._channels(timeframe):
            if _is_channel_down(label):
                # 熔断中：该通道此前已被判定网络级不可达，直接跳过，不再浪费一次超时
                errs.append(f"{label}: 网络不可达（熔断中，详见健康检查）")
                continue
            for attempt in range(2):  # 每通道最多 2 次
                try:
                    df = fn(code, pref, secid, start, end, timeframe, adjust)
                    if df is None or df.empty:
                        raise ValueError("返回空数据")
                    df = _clip_to_window(df, start, end)
                    if df.empty:
                        raise ValueError("该区间内无数据")
                    _record_success()
                    _clear_channel_down(label)
                    self.last_source = label
                    return self._finalize(df, tz="Asia/Shanghai")
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{label}[第{attempt + 1}次]: {type(e).__name__}: {e}")
                    if _is_transport_error(e):
                        # 连不上/被断连：换通道 ≈ 立即重试同一个死域名，没有意义。
                        # 标记熔断并跳出本通道，直接把机会让给下一条。
                        _mark_channel_down(label)
                        break
                    # 限流/服务端错误：退避后重试（指数退避 2/4/8s，连续 3 次失败暂停 30s）
                    time.sleep(_record_failure())
        self.last_source = None
        raise RuntimeError("akshare 拉取失败（东财/腾讯/新浪 均不可用）: " + " | ".join(errs))

    # ---- 健康检查（逐通道独立探测，每条腿记自己的延迟） ----
    def _ping(self) -> None:
        """逐条通道独立探测，每条腿记录**自己**的耗时与结果。

        为什么不能「一条链试到通为止」：
        健康检查若按 东财→腾讯→新浪 串联，东财永久不通时前 1.3s 全花在等它超时，
        最后成功的新浪会把这段等待一起算进自己的延迟 → UI 显示「新浪 2010ms 延迟异常」，
        而它实际只有 0.7s。**必须逐通道分别计时**，否则延迟数字是假的。

        探测用 1d 而非分钟线：1d 三条腿都支持，能公平比较；也避开 640 根窗口限制。
        超时用 ``_HEALTH_TIMEOUT=3s``，三条腿并发，总耗时不叠加。

        注意：这里**不受熔断表影响**。熔断只用来省掉批量下载里的无效等待，
        而健康检查是用户显式点「重新检测」，必须真的去探一次，否则通道恢复后
        UI 会一直显示「熔断中」而无法自愈。
        """
        from concurrent.futures import ThreadPoolExecutor

        code, pref, secid = _normalize_a_symbol("000001.SZ")
        now = datetime.now()
        start = now - timedelta(days=3)
        chans = self._channels("1d")

        def _probe(item):
            label, fn = item
            t0 = time.perf_counter()
            try:
                df = fn(code, pref, secid, start, now, "1d", "qfq",
                        timeout=_HEALTH_TIMEOUT, throttle=False)
                ms = (time.perf_counter() - t0) * 1000
                if df is None or df.empty:
                    return label, False, ms, "返回空数据"
                _clear_channel_down(label)
                return label, True, ms, None
            except Exception as e:  # noqa: BLE001
                ms = (time.perf_counter() - t0) * 1000
                if _is_transport_error(e):
                    _mark_channel_down(label)
                return label, False, ms, f"{type(e).__name__}: {e}"

        # 并发探测：三条腿互不等待，总耗时 ≈ 最慢的那条
        with ThreadPoolExecutor(max_workers=len(chans)) as ex:
            results = list(ex.map(_probe, chans))

        details: list[dict] = []
        first_ok: tuple[str, float] | None = None
        status = channel_status()  # 一次取齐，避免逐条加锁
        for label, ok, ms, err in results:
            is_preferred = bool(ok and first_ok is None)
            ms_r = round(ms, 1)
            if is_preferred:
                first_ok = (label, ms_r)
            details.append({
                "name": label,
                "healthy": ok,
                "latency_ms": ms_r,
                "error": err,
                # 熔断状态带剩余秒数（"cooling 274s"），UI 才能显示「N 秒后自动重试」
                "breaker": status.get(label, "ok"),
                "preferred": is_preferred,
            })
        self.channels = details
        self.last_source = first_ok[0] if first_ok else None
        self._primary_latency_ms = first_ok[1] if first_ok else None
        if first_ok is None:
            raise RuntimeError("akshare 不可用（东财 / 腾讯 / 新浪 全部失败）: "
                               + " | ".join(f"{d['name']}={d['error']}" for d in details))

    def health_check(self):
        """延迟只报**主用通道**自己的耗时，不是整条链的累计值。

        这是上一版最误导人的地方：东财不通时，串联探测把它的超时算进新浪，
        UI 上出现「新浪 2010ms 延迟异常」，实际新浪只要 0.7s。
        """
        self.channels = None
        self._primary_latency_ms = None
        h = super().health_check()
        h.source = self.last_source
        h.channels = self.channels
        if h.healthy and self._primary_latency_ms is not None:
            h.latency_ms = self._primary_latency_ms
        return h
