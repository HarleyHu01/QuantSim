"""交易时段判定：给模拟盘做门控，非交易时段不拉数、不撮合。

设计取舍：
- **不引入交易日历依赖**（baostock/akshare 都带日历，但每次查都要联网且慢）。
  这里用「工作日 + 时段窗口」近似：周一至周五的开盘时段视为可交易。
  法定节假日会被误判为可交易 —— 但此时数据源本来就没有新 bar，
  `_tick` 只会空手而归，不会产生错误成交。代价只是一天多几次空拉。
  这个取舍是有意的：宁可多拉几次，也不要因为日历不准而漏掉真实成交。
- 时段窗口按标的所属市场的本地时间判断，返回时区感知的边界供 UI 展示。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

# 各市场的连续竞价时段（本地时间）。A股另有 09:15-09:25 集合竞价、14:57 收盘集合竞价，
# 这里只覆盖连续竞价 —— 模拟盘按 K 线收盘撮合，集合竞价不产生独立 K 线。
# A股是两个时段（午休分开），美股一个时段。
_SESSION_STARTS: dict[str, tuple[time, ...]] = {
    "cn": (time(9, 30), time(13, 0)),
    "us": (time(9, 30),),
}
_SESSIONS: dict[str, tuple[time, time]] = {
    "cn": (time(9, 30), time(15, 0)),
    "us": (time(9, 30), time(16, 0)),
}

MARKET_TZ: dict[str, str] = {
    "cn": "Asia/Shanghai",
    "us": "America/New_York",
}

# A股午休（11:30-13:00）：这段时间没有新 K 线，轮询可以直接跳过。
_CN_LUNCH: tuple[time, time] = (time(11, 30), time(13, 0))


@dataclass(frozen=True)
class SessionState:
    """某一时刻某市场的交易状态。"""

    market: str
    is_open: bool
    reason: str  # 人类可读：交易中 / 午休 / 已收盘 / 周末
    local_now: datetime
    next_open: datetime | None  # 下一次开盘（UTC）；无法确定时为 None
    seconds_to_next_open: float | None

    def to_dict(self) -> dict:
        return {
            "market": self.market,
            "is_open": self.is_open,
            "reason": self.reason,
            "local_now": self.local_now.isoformat(),
            "next_open": self.next_open.isoformat() if self.next_open else None,
            "seconds_to_next_open": (
                round(self.seconds_to_next_open, 1)
                if self.seconds_to_next_open is not None
                else None
            ),
        }


def market_of(source: str | None, symbol: str = "") -> str:
    """按数据源或标的推断市场：cn / us。"""
    if source in ("baostock", "akshare", "tencent"):
        return "cn"
    if source == "yfinance":
        return "us"
    # source 未知时按代码形态兜底：6 位数字 = A股
    code = symbol.split(".")[0]
    if symbol and code.isdigit() and len(code) == 6:
        return "cn"
    return "us"


def _in_window(t: time, lo: time, hi: time) -> bool:
    return lo <= t < hi


class _Clock:
    """可注入的时钟，便于测试固定时间。"""

    def now(self, tz: str) -> datetime:
        return datetime.now(ZoneInfo(tz))


_default_clock = _Clock()


def session_state(
    market: str,
    *,
    now: datetime | None = None,
    clock: _Clock | None = None,
) -> SessionState:
    """判断市场当前是否处于连续竞价时段。

    `now` 传入 UTC 感知时间（便于测试）；不传则用系统时钟。
    """
    tz_name = MARKET_TZ.get(market, "UTC")
    tz = ZoneInfo(tz_name)
    if now is None:
        local = (clock or _default_clock).now(tz_name)
    else:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        local = now.astimezone(tz)

    lo, hi = _SESSIONS.get(market, (time(0, 0), time(23, 59)))
    t = local.time()
    weekday = local.weekday()  # 0=周一

    # 工作日判断（周末一定是闭市）
    if weekday >= 5:
        nxt = _next_open(market, local, include_today=False)
        return SessionState(
            market=market, is_open=False, reason="周末休市", local_now=local,
            next_open=nxt,
            seconds_to_next_open=(nxt - local.astimezone(timezone.utc)).total_seconds()
            if nxt else None,
        )

    if market == "cn" and _in_window(t, *_CN_LUNCH):
        nxt = _next_open(market, local, include_today=True)
        return SessionState(
            market=market, is_open=False, reason="午间休市", local_now=local,
            next_open=nxt,
            seconds_to_next_open=(nxt - local.astimezone(timezone.utc)).total_seconds()
            if nxt else None,
        )

    if _in_window(t, lo, hi):
        return SessionState(
            market=market, is_open=True, reason="交易中", local_now=local,
            next_open=None, seconds_to_next_open=None,
        )

    nxt = _next_open(market, local, include_today=(t < lo))
    return SessionState(
        market=market, is_open=False,
        reason="未开盘" if t < lo else "已收盘", local_now=local,
        next_open=nxt,
        seconds_to_next_open=(nxt - local.astimezone(timezone.utc)).total_seconds()
        if nxt else None,
    )


def _next_open(market: str, local: datetime, *, include_today: bool) -> datetime | None:
    """计算下一次开盘时刻（UTC）。最多向后找 10 天，避开周末。"""
    tz = ZoneInfo(MARKET_TZ.get(market, "UTC"))
    starts = _SESSION_STARTS.get(market, (time(0, 0),))
    day = local.date()
    for offset in range(0, 11):
        cand_day = day + timedelta(days=offset)
        if cand_day.weekday() >= 5:
            continue
        for start in starts:
            cand = datetime.combine(cand_day, start, tzinfo=tz)
            if offset == 0 and not include_today:
                continue
            if cand > local:
                return cand.astimezone(timezone.utc)
    return None
