"""数据源代理控制：每条渠道独立选择「直连 / 走代理 / 跟随系统」。

背景（用户真实痛点）：
- 走系统代理时，国内站点（baostock / akshare 东方财富 / 腾讯）会延迟甚至超时；
- 关掉代理时，国外站点（yfinance / Yahoo）又连不上。

因此需要**逐渠道**决定网络请求是否走代理，而不是一刀切。

三种模式（PROXY_MODES）：
- ``auto``   —— 跟随系统环境变量（requests / urllib 默认行为，尊重 HTTP(S)_PROXY 与 NO_PROXY）
- ``direct`` —— 强制直连，完全绕过代理（即使系统开了代理）
- ``proxy``  —— 强制走系统代理（忽略 NO_PROXY，显式使用 HTTP(S)_PROXY）

实现要点：
- 用模块级可变注册表 ``_modes`` 记录「provider name → 模式」，各 Provider 在**每次请求时**
  调用 ``get_mode(name)`` 读取最新值，因此改配置后无需重建 Provider 进程内立即生效。
- 默认值 DEFAULT_MODES：国内三源（baostock/akshare/tencent）直连，yfinance 跟随系统。
- baostock 经官方客户端 TCP 直连服务器，不经 HTTP 代理，其模式仅作记录、不影响行为。
"""
from __future__ import annotations

import os
from typing import Any

PROXY_MODES = ("auto", "direct", "proxy")

# 各数据源默认代理模式（国内直连更快，国外跟随系统）
DEFAULT_MODES: dict[str, str] = {
    "baostock": "direct",   # TCP 直连，HTTP 代理本就不生效；统一标记为直连
    "akshare": "direct",    # 东方财富（国内），直连
    "tencent": "direct",    # 腾讯行情（国内），直连
    "yfinance": "auto",     # Yahoo（国外），跟随系统（通常需代理，用户可显式改 proxy）
}

# 运行时注册表（provider name → mode），由 settings 路由 / DataManager 写入
_modes: dict[str, str] = {}


def normalize_mode(mode: str | None) -> str:
    """非法/空模式回退为 'auto'。"""
    return mode if mode in PROXY_MODES else "auto"


def get_mode(provider: str) -> str:
    """读取某数据源当前代理模式（未设置时用默认值）。"""
    if provider in _modes:
        return _modes[provider]
    return DEFAULT_MODES.get(provider, "auto")


def get_all_modes() -> dict[str, str]:
    """返回全部已知数据源的当前模式（默认值 + 已覆盖）。"""
    out = dict(DEFAULT_MODES)
    out.update(_modes)
    return out


def set_modes(modes: dict[str, str]) -> None:
    """整体覆盖运行时注册表（过滤非法模式）。"""
    _modes.clear()
    for name, mode in (modes or {}).items():
        _modes[name] = normalize_mode(mode)


def _system_proxy() -> str | None:
    """从环境变量解析系统代理（HTTPS_PROXY 优先，其次 HTTP_PROXY / ALL_PROXY）。"""
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                "ALL_PROXY", "all_proxy"):
        v = os.environ.get(key)
        if v and v.strip():
            return v.strip()
    return None


def build_requests_proxies(mode: str | None) -> dict[str, str | None] | None:
    """返回传给 ``requests.get(proxies=...)`` 的值。

    - ``direct`` → ``{"http": None, "https": None}``（None = 该协议不走代理）
    - ``proxy``  → 显式传入系统代理（忽略 NO_PROXY）
    - ``auto``   → None（requests 默认，跟随环境变量 + NO_PROXY）
    """
    mode = normalize_mode(mode)
    if mode == "direct":
        return {"http": None, "https": None}
    if mode == "proxy":
        p = _system_proxy()
        if p:
            return {"http": p, "https": p}
        return None  # 无系统代理可用 → 退化为跟随系统
    return None


def make_requests_session(mode: str | None) -> Any:
    """构造带代理策略的 requests.Session（给 yfinance 内部使用）。

    - ``direct`` → ``trust_env=False``（彻底忽略环境代理）
    - ``proxy``  → 显式 ``session.proxies``（忽略 NO_PROXY）
    - ``auto``   → 默认 session（trust_env=True，跟随系统）
    """
    import requests

    mode = normalize_mode(mode)
    session = requests.Session()
    if mode == "direct":
        session.trust_env = False
    elif mode == "proxy":
        p = _system_proxy()
        if p:
            session.proxies = {"http": p, "https": p}
    return session


def build_urllib_opener(mode: str | None) -> Any | None:
    """返回给 ``urllib.request`` 用的 opener；``None`` = 用默认 urlopen（跟随系统）。

    - ``direct`` → ``ProxyHandler({})``（空字典 = 全部直连）
    - ``proxy``  → 显式 ProxyHandler（忽略 NO_PROXY）
    - ``auto``   → None
    """
    import urllib.request

    mode = normalize_mode(mode)
    if mode == "direct":
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if mode == "proxy":
        p = _system_proxy()
        if p:
            return urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": p, "https": p}))
    return None
