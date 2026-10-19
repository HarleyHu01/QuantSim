"""baostock 数据源：A股（股票 / 指数 / ETF）K线。

baostock（证券宝，https://www.baostock.com）是免费开源的 A股数据平台，
通过官方 Python 客户端（``pip install baostock``）以 TCP 直连官方服务器提供：

- 股票（1990-12-19 至今）与 ETF（2026-01-05 至今）：日线/周线/月线 + 5/15/30/60 分钟线
- 指数（2006-01-01 至今）：日线/周线/月线，**不提供分钟线**
- 复权：adjustflag 1=后复权 2=前复权 3=不复权；**分钟线不支持复权**（按不复权处理并告警）；
  指数无复权概念（按不复权处理并告警）
- 成交量单位：**股**（无需 ×100）；成交额单位：元
- 分钟线（股票/ETF）仅保留近 5 年 / 2026-01-05 起

与 akshare 的分工：
- A股**股票/指数/ETF** → baostock（主源，直连官方服务器）
- A股 **期货** → 仅 akshare（baostock 不覆盖期货）；指数分钟线 → akshare/yfinance

实现要点：
- baostock 是全局 TCP 会话，login/query/logout 全程加模块锁串行化，避免并发冲突
- 每次操作 login → query → logout，保证会话不泄漏（重复 login 的错误码按已登录处理）
- 查询字段按 frequency 区分（真实服务器严格校验，实测）：
  日线含 preclose 不含 time；周/月线不含 preclose；分钟线含 time 不含 preclose
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import ClassVar

import pandas as pd

from quant.data.base import DataProvider, ProviderHealth

try:  # 可选依赖（requirements-optional.txt）
    import baostock as bs
except ImportError:  # pragma: no cover
    bs = None

log = logging.getLogger(__name__)

# 周期 → baostock frequency（"1h" 与 "60m" 同为 60 分钟线）
_TF_MAP = {"5m": "5", "15m": "15", "30m": "30", "60m": "60", "1h": "60",
           "1d": "d", "1w": "w", "1M": "m"}
_MINUTE_TFS = frozenset({"5m", "15m", "30m", "60m", "1h"})
_MINUTE_FREQS = frozenset({"5", "15", "30", "60"})
# 复权 → adjustflag（1=后复权 2=前复权 3=不复权）
_ADJ_MAP = {"qfq": "2", "hfq": "1", "none": "3"}
# 查询字段按 frequency 区分（baostock 官方字段表 + 真实服务器实测）：
# - 日线：含 preclose，**不含 time**（传 time 报 10004012）
# - 周线/月线：**不含 preclose**（实测传 preclose 报 10004012），也不含 time
# - 分钟线（5/15/30/60）：date+time，**不含 preclose**（preclose 仅日线有效）
_FIELDS_DAILY = "date,code,open,high,low,close,preclose,volume,amount"
_FIELDS_WEEKLY_MONTHLY = "date,code,open,high,low,close,volume,amount"
_FIELDS_MINUTE = "date,time,code,open,high,low,close,volume,amount"

_LOGIN_ALREADY = "10001001"  # baostock 重复登录错误码（按已登录处理）

_bs_lock = threading.Lock()  # baostock 全局会话锁（login/query/logout 串行化）

# ================= 会话保持（性能关键） =================
# 实测（2026-10）：baostock 的 ``login()`` 单次耗时 **11~13 秒**，而一次
# ``query_history_k_data_plus`` 只要 2~3 秒。原实现「每条查询都 login→query→logout」，
# 于是拉一个周期的数据要 15 秒以上 —— 这正是「baostock 总超时」的真正原因，
# 与网络抖动无关（TCP 建连只要 49ms）。
#
# 现在改为**进程内保持一条登录会话**：首次用到时登录一次，之后所有查询复用它。
# 只有遇到明确的会话失效错误才重建。登出交给 ``close_session()``（进程退出/手动重置）。
_logged_in = False
_QUERY_RETRIES = 2  # 查询失败后重建会话并重试的次数
_RETRY_BACKOFF = 1.5  # 重试前的退避基数（秒），逐次递增
# 以 10002 开头的登录错误码是**传输层**问题（10002007「网络接收错误」等），
# 换一次会话通常就好了，值得重试。
# 其余非 0 码是服务器明确拒绝（参数错、账号问题……），重试只会白费时间并加大压力。
_NETWORK_CODE_PREFIX = "10002"


class LoginRejected(RuntimeError):
    """登录被服务器明确拒绝（非传输层问题）——重试没有意义，直接上抛。"""
# 线程局部状态：timeout / connect_timeout 只对当前线程本次调用生效。
# 曾经的严重 bug：健康检查 _ping() 传 timeout=6.0，而旧实现把它写进全局
# _SOCKET_TIMEOUT 且永不恢复 → 之后每次 login 都只有 6s 预算，
# 而 login 实测要 11~13s，于是「健康检查跑过一次之后就永远登录失败」。
_tls = threading.local()

# 服务器明确拒绝（不是网络问题）的错误码：重试只会让对方更烦，直接判死。
# 10001011 = 「黑名单用户，请与管理员联系」。这类拒绝一旦出现，
# login 往往要等几十秒才返回，把它当网络错误反复重试 / 预热 / 健康检查
# 等于持续加压，只会让恢复更慢。不同机器、不同网络下是否会碰到各不相同，
# 所以这里的文案只说明「服务器拒绝登录」这一事实，不对原因下判断。
_FATAL_LOGIN_CODES = {
    "10001011": "baostock 服务器拒绝登录（10001011）。可稍后重试，或换个网络环境",
}
# 两种「不要立刻再试」的状态，各带冷却时间：
#   fatal  —— 服务器明确拒绝（黑名单），冷却 30 分钟
#   down   —— 传输层连不上，冷却 2 分钟
# 目的都是让健康检查在 1 秒内给出可读结论，而不是干等 3 分钟再报「健康检查超时」。
_FATAL_COOLDOWN = 1800.0
_DOWN_COOLDOWN = 120.0
_last_fatal: str | None = None
_last_fatal_at: float = 0.0
_last_down: str | None = None
_last_down_at: float = 0.0


def _set_fatal(msg: str) -> None:
    """记录「服务器明确拒绝」状态（长冷却）。"""
    global _last_fatal, _last_fatal_at
    _last_fatal = msg
    _last_fatal_at = time.monotonic()


def _set_down(msg: str) -> None:
    """记录「网络暂时连不上」状态（短冷却）。"""
    global _last_down, _last_down_at
    _last_down = msg
    _last_down_at = time.monotonic()


def fatal_reason() -> str | None:
    """当前是否处于「不要立刻重试」状态；返回可读原因（冷却期内有效）。"""
    now = time.monotonic()
    if _last_fatal and now - _last_fatal_at < _FATAL_COOLDOWN:
        return _last_fatal
    if _last_down and now - _last_down_at < _DOWN_COOLDOWN:
        return _last_down
    return None


def clear_fatal() -> None:
    """清除「不要立刻重试」状态（用户换网络 / 点「重新检测」时调用）。"""
    global _last_fatal, _last_down
    _last_fatal = None
    _last_down = None


def _ensure_login() -> None:
    """确保已登录（幂等）。调用方必须已持有 ``_bs_lock``。

    服务器明确拒绝（黑名单）时直接抛错，不再重试——反复登录只会延长封禁。
    """
    global _logged_in
    if bs is None:
        raise RuntimeError("baostock 未安装")
    if _logged_in:
        return
    blocked = fatal_reason()
    if blocked:
        raise RuntimeError(blocked)
    lg = bs.login()
    code = str(getattr(lg, "error_code", "") or "")
    if code in _FATAL_LOGIN_CODES:
        _set_fatal(_FATAL_LOGIN_CODES[code])
        raise RuntimeError(_FATAL_LOGIN_CODES[code])
    if code not in ("0", _LOGIN_ALREADY):
        msg = f"baostock 登录失败: {code} {lg.error_msg}"
        if code.startswith(_NETWORK_CODE_PREFIX):
            # 传输层失败（如 10002007「网络接收错误」）——建连函数已把原因写进 _set_down，
            # 抛普通 RuntimeError 让上层换会话重试。
            raise RuntimeError(msg)
        # 服务器明确拒绝：重试无意义，直接上抛且不重试。
        raise LoginRejected(msg)
    _logged_in = True


def _force_relogin() -> None:
    """丢弃当前会话并重新登录（服务器中途掉线后恢复用）。调用方须持有 ``_bs_lock``。

    黑名单状态下不做任何网络尝试，直接置为未登录并返回。
    """
    global _logged_in
    if bs is None:
        return
    try:
        bs.logout()
    except Exception:  # noqa: BLE001
        pass
    _logged_in = False
    if fatal_reason():
        return
    try:
        lg = bs.login()
        code = str(getattr(lg, "error_code", "") or "")
        if code in _FATAL_LOGIN_CODES:
            _set_fatal(_FATAL_LOGIN_CODES[code])
        _logged_in = code in ("0", _LOGIN_ALREADY)
    except Exception:  # noqa: BLE001
        _logged_in = False


def close_session() -> None:
    """关闭登录会话（进程退出 / 需要强制重置时调用）。"""
    global _logged_in
    if bs is None:
        return
    with _bs_lock:
        if not _logged_in:
            return
        try:
            bs.logout()
        except Exception:  # noqa: BLE001
            pass
        _logged_in = False


def session_state() -> dict:
    """当前会话状态（供健康检查/UI 展示）。"""
    return {"logged_in": _logged_in, "fatal": fatal_reason()}


# baostock 官方客户端的 socket **不设超时**（socketutil.connect 里是裸 socket() + connect()），
# 服务器不响应时 login() 会无限阻塞。首次 login 曾观测到 92 秒才成功。
# 这里把 socketutil 里「建连」的两个函数包一层，注入 settimeout。
#
# 超时值分两档：
#   _CONNECT_TIMEOUT —— 单纯 TCP 建连。实测连 public-api.baostock.com:10030
#       要么 124ms 建上、要么直接 8 秒不通，没有任何中间状态，所以给 10 秒足够区分。
#   _SOCKET_TIMEOUT  —— 建连之后的读写（login 握手实测 11~13 秒，个别情况更久），给 30 秒。
_CONNECT_TIMEOUT = 10.0
_SOCKET_TIMEOUT = 30.0
# 健康检查用的建连超时：要比默认更短，否则 UI 上「检测中」要转十秒。
_HEALTH_CONNECT_TIMEOUT = 6.0
_socket_patched = False


def _current_connect_timeout() -> float:
    """当前线程生效的建连超时。"""
    return getattr(_tls, "connect_timeout", None) or _CONNECT_TIMEOUT


def _current_socket_timeout() -> float:
    """当前线程生效的读超时（线程局部覆盖优先，否则用默认值）。"""
    return getattr(_tls, "timeout", None) or _SOCKET_TIMEOUT


def _patch_socket_timeout(timeout: float | None = None) -> None:
    """给 baostock 内部创建的 socket 注入超时（幂等，只打一次补丁）。

    ``timeout`` 只对**当前线程本次调用**生效（线程局部），不会污染其它线程，
    也不会改变后续调用的默认值。
    """
    global _socket_patched
    if bs is None:  # pragma: no cover
        return
    _tls.timeout = None if timeout is None else max(1.0, float(timeout))
    if _socket_patched:
        return
    try:
        import socket as _socket

        import baostock.util.socketutil as bs_sock
    except ImportError:  # pragma: no cover
        return

    def _make_socket(connect: bool = False) -> "_socket.socket":
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.settimeout(_current_connect_timeout() if connect else _current_socket_timeout())
        return s

    def connect(self):  # noqa: ANN001
        """重写 SocketUtil.connect：带超时建连，失败时不把未初始化变量塞进 context。"""
        try:
            my_socket = _make_socket(connect=True)
            my_socket.connect((bs_sock.cons.BAOSTOCK_SERVER_IP,
                               bs_sock.cons.BAOSTOCK_SERVER_PORT))
            my_socket.settimeout(_current_socket_timeout())  # 建连后换成读写超时
        except Exception as e:
            reason = (f"baostock 服务器连接失败（TCP {bs_sock.cons.BAOSTOCK_SERVER_IP}:"
                      f"{bs_sock.cons.BAOSTOCK_SERVER_PORT} {_current_connect_timeout():.0f}s 超时）")
            log.warning("%s: %s", reason, e)
            _set_down(reason)
            setattr(bs_sock.context, "default_socket", None)
            return
        setattr(bs_sock.context, "default_socket", my_socket)

    def get_default_socket():
        """重写 get_default_socket：带超时建连。"""
        try:
            sock = _make_socket(connect=True)
            sock.connect((bs_sock.cons.BAOSTOCK_SERVER_IP,
                          bs_sock.cons.BAOSTOCK_SERVER_PORT))
            sock.settimeout(_current_socket_timeout())
        except Exception as e:
            reason = (f"baostock 服务器连接失败（TCP {bs_sock.cons.BAOSTOCK_SERVER_IP}:"
                      f"{bs_sock.cons.BAOSTOCK_SERVER_PORT} {_current_connect_timeout():.0f}s 超时）")
            log.warning("%s: %s", reason, e)
            _set_down(reason)
            return None
        return sock

    bs_sock.SocketUtil.connect = connect
    bs_sock.get_default_socket = get_default_socket
    _socket_patched = True
    log.debug("已为 baostock 注入 socket 超时（建连 %.0fs / 读写 %.0fs）",
              _CONNECT_TIMEOUT, _SOCKET_TIMEOUT)


def _to_baostock_code(symbol: str) -> str:
    """统一 A股代码 → baostock 代码。

    - 带后缀：.SH → sh.X；.SZ → sz.X（000001.SH → sh.000001 上证指数；000001.SZ → sz.000001 平安银行）
    - 6 位纯数字自动推断市场/类型：
      - 51/56/58 开头 → sh（沪 ETF）；15/16 开头 → sz（深 ETF）
      - 000 开头 → sh（上证指数系列）；399 开头 → sz（深证指数系列）
      - 5/6/9 开头 → sh（沪股票）；0/3 开头 → sz（深股票）
    - 北交所（4/8 开头、.BJ）baostock 不支持 → ValueError
    """
    s = symbol.strip().upper()
    if "." in s:
        code, suffix = s.split(".")[:2]
        if suffix == "SH":
            return f"sh.{code}"
        if suffix == "SZ":
            return f"sz.{code}"
        raise ValueError(f"baostock 不支持的市场后缀: {symbol}（仅支持 .SH/.SZ）")
    if s.isdigit() and len(s) == 6:
        if s.startswith(("51", "56", "58")):
            return f"sh.{s}"  # 沪 ETF
        if s.startswith(("15", "16")):
            return f"sz.{s}"  # 深 ETF
        if s.startswith("000"):
            return f"sh.{s}"  # 上证指数（sh.000001 等）
        if s.startswith("399"):
            return f"sz.{s}"  # 深证指数（sz.399001 等）
        if s.startswith(("5", "6", "9")):
            return f"sh.{s}"
        if s.startswith(("0", "3")):
            return f"sz.{s}"
        raise ValueError(f"baostock 不支持的代码段（北交所 4/8 开头等）: {symbol}")
    raise ValueError(f"非 A股代码（baostock 仅支持沪深股票/指数/ETF）: {symbol}")


def _is_index_bs(bs_code: str) -> bool:
    """baostock 代码是否为指数（sh.000xxx / sz.399xxx，指数无分钟线、无复权概念）。"""
    return bs_code.startswith("sh.000") or bs_code.startswith("sz.399")


def _parse_bs_dt(date_s: str, time_s: str) -> pd.Timestamp:
    """baostock 日期/时间 → Timestamp（naive，待 _finalize 转 Asia/Shanghai→UTC）。

    - 日线：time 为空串，date 形如 "2024-01-02"
    - 分钟线：time 形如 "YYYYMMDDHHMMSSsss"（17 位，含毫秒）或 "HH:MM:SS"
    """
    if not time_s:
        return pd.Timestamp(date_s)
    t = time_s.strip()
    if len(t) >= 14 and t[:4].isdigit():
        try:
            return pd.Timestamp(datetime.strptime(t[:14], "%Y%m%d%H%M%S"))
        except ValueError:
            pass
    try:
        return pd.Timestamp(f"{date_s} {t}")
    except ValueError:
        return pd.Timestamp(date_s)


def _rows_to_df(fields: list[str], rows: list[list]) -> pd.DataFrame:
    """baostock 行数据 → 统一 OHLCV DataFrame（naive index，Asia/Shanghai 本地时间）。

    按返回的字段名（rs.get_fields()）解析，兼容日线（无 time）与分钟线（含 time）两种布局。
    """
    out = []
    for r in rows:
        row = dict(zip(fields, r))
        try:
            out.append({
                "datetime": _parse_bs_dt(row.get("date", ""), row.get("time", "")),
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
                "volume": float(row["volume"]),  # 单位：股
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).set_index("datetime")
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def _query(code: str, start: str, end: str, freq: str, adjflag: str,
           timeout: float | None = None) -> tuple[list[str], list[list]]:
    """查询历史数据，返回 (字段名列表, 原始行数据)。

    会话由 ``_ensure_login()`` 保持（不再每次查询重登）；查询失败会重建会话重试。
    ``timeout`` 只对本次调用所在的线程生效（线程局部），不会污染全局 socket 超时
    ——旧实现会把它永久写进全局 ``_SOCKET_TIMEOUT``，健康检查传 6s 后
    所有后续 login 都只有 6s 预算，而 login 需要 11~13s，于是永久 10002007。
    """
    _patch_socket_timeout(timeout)
    if freq in ("d",):
        fields = _FIELDS_DAILY
    elif freq in _MINUTE_FREQS:
        fields = _FIELDS_MINUTE
    else:  # w / m
        fields = _FIELDS_WEEKLY_MONTHLY
    try:
        with _bs_lock:
            last_err: Exception | None = None
            # 登录本身也可能被服务器断开：实测 10002007「网络接收错误」既可能出现在
            # login 也可能出现在 query。因此 login 也要在重试循环内。
            for attempt in range(_QUERY_RETRIES):
                try:
                    _ensure_login()
                    rs = bs.query_history_k_data_plus(
                        code, fields, start_date=start, end_date=end,
                        frequency=freq, adjustflag=adjflag,
                    )
                    if rs.error_code != "0":
                        raise RuntimeError(f"baostock 查询失败: {rs.error_code} {rs.error_msg}")
                    # 真实 baostock 的 ResultData 提供 .fields 属性（无 get_fields() 方法）
                    field_names = list(rs.fields) if getattr(rs, "fields", None) else fields.split(",")
                    rows: list[list] = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    return field_names, rows
                except LoginRejected:
                    raise  # 服务器明确拒绝：不重试、不重建会话
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if fatal_reason():
                        # 服务器明确拒绝（黑名单）：立刻放弃，别再反复 login 加压
                        raise RuntimeError(fatal_reason()) from e
                    log.warning("baostock 查询失败（第 %d/%d 次），重建会话后重试: %s",
                                attempt + 1, _QUERY_RETRIES, e)
                    _force_relogin()
                    time.sleep(_RETRY_BACKOFF * (attempt + 1))  # 给服务器一点恢复时间
            raise RuntimeError(f"baostock 查询失败（已重试 {_QUERY_RETRIES} 次）: {last_err}")
    finally:
        _tls.timeout = None


class BaostockProvider(DataProvider):
    """baostock 数据源：A股股票/指数/ETF（日/周/月 + 5/15/30/60 分钟，qfq/hfq/none）。"""

    name: ClassVar[str] = "baostock"
    display_name: ClassVar[str] = "BaoStock（A股主源）"
    asset_types: ClassVar[tuple[str, ...]] = ("stock", "etf", "index")

    def __init__(self):
        # baostock 经官方客户端 TCP 直连服务器，不受 HTTP 系统代理影响，无需代理配置
        self.last_source: str | None = None  # 最近一次成功数据源（恒为 "baostock"）

    # ---- 能力声明（DataManager 据此直接跳到下一级渠道，不发无效请求） ----
    def supports(self, symbol: str, timeframe: str) -> bool:
        """baostock 无 1m；指数无分钟线；北交所代码不支持。"""
        if timeframe not in _TF_MAP:
            return False
        try:
            code = _to_baostock_code(symbol)
        except ValueError:
            return False
        if _is_index_bs(code) and timeframe in _MINUTE_TFS:
            return False
        return True

    def max_lookback_days(self, symbol: str, timeframe: str) -> int | None:
        """日/周/月线可回溯到上市首日（不设限）；分钟线实测最早约 2005 年起、
        2020 年仍稳定可取，这里收在 2500 天（约 6.8 年）。

        收窄的实际意义：避免「下载全部历史」把分钟线从 2000 年一路拉下来
        （实测 730 天就要 65 秒，跨度再放大到 20 年会等到超时）。
        """
        if timeframe in _MINUTE_TFS:
            return 2500
        return None


    def fetch_history(self, symbol, start, end, timeframe="1d", adjust="none") -> pd.DataFrame:
        if bs is None:
            raise RuntimeError("baostock 未安装（pip install baostock）")
        if timeframe not in _TF_MAP:
            raise ValueError(f"不支持的周期: {timeframe}（baostock 支持 5m/15m/30m/60m/1d/1w/1M）")
        code = _to_baostock_code(symbol)
        freq = _TF_MAP[timeframe]
        is_index = _is_index_bs(code)
        if is_index and timeframe in _MINUTE_TFS:
            # 指数无分钟线（官方文档：指数日/周/月线 2006-01-01 至今，无分钟线）
            raise ValueError(
                f"baostock 指数不支持分钟线（{symbol}），请使用日/周/月线，"
                f"或走 akshare/yfinance fallback")
        if is_index and adjust != "none":
            # 指数无复权概念：服务器接受复权参数但数据不变，按不复权处理并告警
            log.warning("baostock 指数无复权概念（%s），按不复权拉取 %s", adjust, symbol)
            adjflag = "3"
        elif timeframe in _MINUTE_TFS and adjust != "none":
            # baostock 分钟线不支持复权，官方仅返回不复权数据
            log.warning("baostock 分钟线不支持复权（%s），按不复权拉取 %s", adjust, symbol)
            adjflag = "3"
        else:
            adjflag = _ADJ_MAP.get(adjust, "3")
        fields, rows = _query(code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                              freq, adjflag)
        if not rows:
            raise RuntimeError(
                f"baostock 返回空数据（{symbol} {timeframe} "
                f"{start:%Y-%m-%d}~{end:%Y-%m-%d} 无行情或代码无效）")
        self.last_source = "baostock"
        return self._finalize(_rows_to_df(fields, rows), tz="Asia/Shanghai")

    def _ping(self) -> None:
        """轻量探测：拉取平安银行（sz.000001）最近 5 天日线。

        **不传 timeout**：socket 超时是线程局部覆盖，健康检查若传一个小的值（曾传 6 秒），
        在会话未预热时会连 login 都来不及完成，只会得到误导性的 `10002007 网络接收错误`。
        login 已由 `DataManager.warm_up()` 在启动时预热，正常路径下这里只跑一次
        2~3 秒的查询。
        """
        if bs is None:
            raise RuntimeError("baostock 未安装")
        blocked = fatal_reason()
        if blocked:
            # 服务器端明确拒绝（黑名单）：不做任何网络尝试，直接给出可读原因
            raise RuntimeError(blocked)
        if _bs_lock.locked() and not _logged_in:
            # 另一个线程正在首次登录（多半是启动预热，login 可达 1 分钟）。
            # 直接等锁会一路撞上 8 秒总预算，UI 只会显示误导性的「健康检查超时」。
            # 这里快速失败并说明真实原因。
            raise RuntimeError("baostock 正在建立连接，请稍后重新检测")
        now = datetime.now()
        # 健康检查用**短建连超时**：TCP 30030 只有「通」和「不通」两种结果，
        # 10 秒足够分辨。读写超时沿用默认 30 秒（login 握手本身慢）。
        _tls.connect_timeout = _HEALTH_CONNECT_TIMEOUT
        try:
            fields, rows = _query("sz.000001", (now - timedelta(days=5)).strftime("%Y-%m-%d"),
                                  now.strftime("%Y-%m-%d"), "d", "3")
        finally:
            _tls.connect_timeout = None
        if not rows:
            raise RuntimeError("baostock 健康检查返回空数据")
        self.last_source = "baostock"

    def health_check(self) -> ProviderHealth:
        h = super().health_check()
        h.source = self.last_source  # 实际数据源（baostock）
        return h
