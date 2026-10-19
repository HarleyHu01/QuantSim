"""数据源逐渠道代理控制测试：模式注册表 + requests/urllib 代理参数构造。"""
from __future__ import annotations

import pytest

from quant.data import proxy


def test_default_modes_domestic_direct_yfinance_auto():
    """默认：国内三源直连，yfinance 跟随系统。"""
    assert proxy.DEFAULT_MODES["baostock"] == "direct"
    assert proxy.DEFAULT_MODES["akshare"] == "direct"
    assert proxy.DEFAULT_MODES["tencent"] == "direct"
    assert proxy.DEFAULT_MODES["yfinance"] == "auto"


def test_get_mode_defaults_and_override():
    proxy.set_modes({})
    assert proxy.get_mode("akshare") == "direct"
    assert proxy.get_mode("unknown") == "auto"
    # 覆盖后立即生效
    proxy.set_modes({"akshare": "proxy"})
    assert proxy.get_mode("akshare") == "proxy"
    proxy.set_modes({})  # 还原


def test_normalize_mode_rejects_bad_values():
    assert proxy.normalize_mode("direct") == "direct"
    assert proxy.normalize_mode("proxy") == "proxy"
    assert proxy.normalize_mode("auto") == "auto"
    assert proxy.normalize_mode("bogus") == "auto"
    assert proxy.normalize_mode(None) == "auto"


def test_get_all_modes_includes_defaults_plus_override():
    proxy.set_modes({"yfinance": "proxy"})
    modes = proxy.get_all_modes()
    assert modes["baostock"] == "direct"
    assert modes["yfinance"] == "proxy"
    proxy.set_modes({})


def test_build_requests_proxies_direct():
    p = proxy.build_requests_proxies("direct")
    assert p == {"http": None, "https": None}


def test_build_requests_proxies_auto_is_none():
    assert proxy.build_requests_proxies("auto") is None


def test_build_requests_proxies_proxy_resolves_env(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    p = proxy.build_requests_proxies("proxy")
    assert p == {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}


def test_build_requests_proxies_proxy_no_env_falls_back_none(monkeypatch):
    for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
              "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    assert proxy.build_requests_proxies("proxy") is None


def test_make_requests_session_direct_ignores_env():
    s = proxy.make_requests_session("direct")
    assert s.trust_env is False


def test_make_requests_session_auto_trusts_env():
    s = proxy.make_requests_session("auto")
    assert s.trust_env is True


def test_make_requests_session_proxy_sets_proxies(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    s = proxy.make_requests_session("proxy")
    assert s.proxies.get("http") == "http://127.0.0.1:7890"


def test_build_urllib_opener_direct_returns_opener():
    op = proxy.build_urllib_opener("direct")
    assert op is not None


def test_build_urllib_opener_auto_returns_none():
    assert proxy.build_urllib_opener("auto") is None
