"""数据模块测试：归一化、存储合并、fallback、增量、健康检查、API。"""
from __future__ import annotations

import importlib.util
import logging
from datetime import datetime, timezone

import pandas as pd
import pytest

from quant.core.config import AppConfig
from quant.data.base import empty_ohlcv
from quant.data.manager import DataManager
from quant.data.storage import MarketStore
from quant.storage.db import init_db
from quant.storage.repositories import MarketDataRepo, SymbolsRepo
from tests.conftest import BrokenProvider, FakeProvider


@pytest.fixture(autouse=True)
def _reset_akshare_breaker():
    """每个用例前清空 akshare 通道熔断表。

    熔断表是模块级全局状态（``_channel_down_until``）：一个用例把东财标记为
    「网络不可达」后，后续用例会直接跳过东财，造成与执行顺序相关的假失败。
    """
    from quant.data import akshare_provider as _ak

    _ak._channel_down_until.clear()
    _ak._consecutive_failures = 0
    yield
    _ak._channel_down_until.clear()
    _ak._consecutive_failures = 0


@pytest.fixture(autouse=True)
def _reset_baostock_session():
    """每个用例前后把 baostock 登录态归零。

    登录态是模块级全局变量（``_logged_in``，会话保持的设计使然）：用例里用
    monkeypatch 换成假的 ``bs`` 模块后，若沿用上一用例留下的 True，假模块的
    ``login()`` 根本不会被调用，于是「登录失败」等用例会静默通过而不报错。
    """
    from quant.data import baostock_provider as _bp

    _bp._logged_in = False
    yield
    _bp._logged_in = False



# ---------- 归一化 ----------

def test_empty_ohlcv_shape():
    df = empty_ohlcv()
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "adj_factor"]
    assert df.empty


def test_finalize_normalizes_columns_and_tz():
    p = FakeProvider()
    raw = pd.DataFrame(
        {"Open": [1.0], "High": [1.1], "Low": [0.9], "Close": [1.05], "Volume": [100]},
        index=pd.to_datetime(["2024-01-02"]),  # naive → 按本地解释后转 UTC
    )
    out = p._finalize(raw)
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "adj_factor"]
    assert out.index.tz is not None
    assert out.index.tz.utcoffset(None) is not None  # UTC
    assert (out["adj_factor"] == 1.0).all()


# ---------- 存储合并 ----------

def test_store_merge_dedupes(tmp_config):
    store = MarketStore(tmp_config.effective_market_dir)
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df1 = pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0,
                        "adj_factor": 1.0}, index=idx)
    store.save("T1", "1d", df1)
    # 重叠 3 天 + 新增 2 天
    idx2 = pd.date_range("2024-01-03", periods=5, freq="D", tz="UTC")
    df2 = pd.DataFrame({"open": 2.0, "high": 2.1, "low": 1.9, "close": 2.0, "volume": 2.0,
                        "adj_factor": 1.0}, index=idx2)
    merged = store.save("T1", "1d", df2)
    assert len(merged) == 7  # 5 + 5 - 3 重叠
    assert merged.index.is_monotonic_increasing
    loaded = store.load("T1", "1d")
    assert len(loaded) == 7


def test_store_roundtrip_keeps_utc(tmp_config):
    store = MarketStore(tmp_config.effective_market_dir)
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0,
                       "adj_factor": 1.0}, index=idx)
    store.save("T2", "1d", df)
    loaded = store.load("T2", "1d")
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.tz is not None


# ---------- DataManager 更新与 fallback ----------

def test_update_ok_writes_meta(manager):
    results = manager.update(symbols=["TEST"], timeframes=["1d"])
    assert len(results) == 1
    r = results[0]
    assert r.ok and r.source == "fake"
    assert r.rows == 10
    assert manager.store.exists("TEST", "1d")
    meta = manager.market_repo.get_meta("TEST", "1d")
    assert meta["rows"] == 10
    assert meta["last_bar_time"]


def test_update_fallback_chain(tmp_config):
    """broken 不健康 → 自动切到 fake。"""
    init_db(tmp_config.effective_db_path)
    store = MarketStore(tmp_config.effective_market_dir)
    # priority 配成 [broken, fake]
    tmp_config.data.providers.stock = ["broken", "fake"]
    dm = DataManager(
        tmp_config, store,
        MarketDataRepo(tmp_config.effective_db_path),
        SymbolsRepo(tmp_config.effective_db_path),
        providers={"fake": FakeProvider(), "broken": BrokenProvider()},
    )
    dm.symbols_repo.upsert(symbol="TEST", name="Test", asset_type="stock", source="broken")
    results = dm.update(symbols=["TEST"], timeframes=["1d"])
    assert results[0].ok
    assert results[0].source == "fake"


def test_update_all_providers_fail(tmp_config):
    init_db(tmp_config.effective_db_path)
    store = MarketStore(tmp_config.effective_market_dir)
    tmp_config.data.providers.stock = ["broken"]
    dm = DataManager(
        tmp_config, store,
        MarketDataRepo(tmp_config.effective_db_path),
        SymbolsRepo(tmp_config.effective_db_path),
        providers={"broken": BrokenProvider()},
    )
    dm.symbols_repo.upsert(symbol="TEST", name="Test", asset_type="stock", source="broken")
    results = dm.update(symbols=["TEST"], timeframes=["1d"])
    assert not results[0].ok
    assert results[0].error


def test_incremental_no_duplication(manager):
    """两次更新（第二次增量）总行数不变。"""
    r1 = manager.update(symbols=["TEST"], timeframes=["1d"])[0]
    r2 = manager.update(symbols=["TEST"], timeframes=["1d"])[0]
    assert r1.ok and r2.ok
    assert r2.rows == r1.rows  # 增量合并后无重复


def test_health_check(manager):
    health = manager.health()
    assert "fake" in health
    assert health["fake"].healthy is True


def test_config_provider_defaults():
    """默认配置：两级渠道 baostock → akshare；腾讯已并入 akshare 内部（不再是独立 provider）。"""
    from quant.core.config import AppConfig

    cfg = AppConfig()
    assert cfg.data.akshare_use_proxy is False
    # 国内两级：主 → 备选（akshare 内部再走 东财→腾讯→新浪 三条腿）
    for at in ("stock", "etf", "index"):
        chain = getattr(cfg.data.providers, at)
        assert chain[0] == "baostock", f"{at} 主源应为 baostock"
        assert chain[-1] == "akshare", f"{at} 备选源应为 akshare"
        assert "tencent" not in chain, f"{at} 不应再把 tencent 当成独立 provider"
    # 美股只走 yfinance
    assert cfg.data.providers.us_stock == ["yfinance"]
    # ccxt / 期货已彻底移除
    dumped = cfg.data.providers.model_dump()
    assert "crypto" not in dumped and "future" not in dumped
    assert not hasattr(cfg.data, "ccxt_exchanges")
    assert not hasattr(cfg.data, "ccxt_timeout_ms")


def test_config_no_crypto_symbols():
    """默认标的池不再包含加密货币。"""
    from quant.core.config import AppConfig

    cfg = AppConfig()
    syms = [s["symbol"] for s in cfg.symbols]
    assert not any("/" in s for s in syms)
    assert all(s["asset_type"] != "crypto" for s in cfg.symbols)


def test_akshare_requests_browser_ua():
    """requests 全局默认 UA 被替换为浏览器 UA（东方财富拒绝 python-requests UA）。"""
    import requests

    from quant.data.akshare_provider import AkshareProvider

    AkshareProvider(use_proxy=False)
    headers = requests.utils.default_headers()
    assert "Mozilla" in headers.get("User-Agent", "")
    assert "python-requests" not in headers.get("User-Agent", "")


# ---------- A股多通道 fallback ----------

def test_akshare_normalize_symbol():
    from quant.data.akshare_provider import _normalize_a_symbol

    assert _normalize_a_symbol("000001") == ("000001", "sz", "0.000001")
    assert _normalize_a_symbol("600519") == ("600519", "sh", "1.600519")
    assert _normalize_a_symbol("000001.SZ") == ("000001", "sz", "0.000001")
    assert _normalize_a_symbol("600519.SH") == ("600519", "sh", "1.600519")
    assert _normalize_a_symbol("510300.SH") == ("510300", "sh", "1.510300")  # 沪ETF


def test_akshare_channel_parsing(monkeypatch):
    """东方财富通道原始响应解析成统一 OHLCV（离线）。"""
    from datetime import datetime

    from quant.data.akshare_provider import _channel_eastmoney

    now = datetime.now()

    def fake_get(url, params, referer, timeout, throttle=True):
        class R:
            def json(self):
                if "push2his" in url:
                    return {"data": {"klines": [
                        "2024-01-02,7.89,7.74,7.91,7.74,1158366,1075742,5.31,1.2,0.09,1.32"]}}
                raise AssertionError(url)
        return R()

    monkeypatch.setattr("quant.data.akshare_provider._http_get", fake_get)
    df = _channel_eastmoney("000001", "sz", "0.000001", now, now, "1d", "qfq")
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert df.iloc[0]["volume"] == 1158366 * 100  # 手→股
    assert df.iloc[0]["close"] == 7.74


def test_akshare_channels_include_sina_fallback(monkeypatch):
    """akshare 内部三条腿：东方财富 / 腾讯 / 新浪，顺序随周期能力动态决定。

    历史背景：曾有人以「渠道只该有三家」为由删掉新浪，结果东财域名被网络层
    阻断后 akshare 彻底失效（错误信息见 README 故障排查）。东财是单点，
    必须保留新浪兜底——它不是独立 provider，只是 akshare 内部的一条腿。
    腾讯同理，原本是独立第三级 provider，现并入本 provider。
    """
    from quant.data.akshare_provider import AkshareProvider

    provider = AkshareProvider(use_proxy=False)
    # 日/周/月：腾讯比新浪快（0.55s vs 1.6s），深度相当 → 腾讯优先
    assert [label for label, _ in provider._channels("1d")] == ["东方财富", "腾讯", "新浪"]
    assert [label for label, _ in provider._channels("1w")] == ["东方财富", "腾讯", "新浪"]
    # 分钟线：新浪 1970 根 ≫ 腾讯 640 根（约 3 倍深度）→ 新浪优先，腾讯退为兜底
    assert [label for label, _ in provider._channels("5m")] == ["东方财富", "新浪", "腾讯"]
    assert [label for label, _ in provider._channels("1h")] == ["东方财富", "新浪", "腾讯"]
    # 1m：只有腾讯支持，独占（东财/新浪都没有）
    assert [label for label, _ in provider._channels("1m")] == ["腾讯"]


def test_akshare_falls_back_to_sina_when_eastmoney_fails(monkeypatch):
    """东财失败 → 自动降级到新浪，整体成功（不再整体失败）。"""
    from datetime import datetime

    import pandas as pd

    from quant.data.akshare_provider import AkshareProvider, _channel_sina

    provider = AkshareProvider(use_proxy=False)

    def fake_em(*a, **k):
        raise ConnectionError("RemoteDisconnected")

    def fake_sina(code, pref, secid, start, end, timeframe, adjust,
                  timeout=10.0, throttle=True):
        return pd.DataFrame({"open": [1.0], "close": [1.0], "high": [1.0],
                             "low": [1.0], "volume": [100.0]},
                            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC")]))

    monkeypatch.setattr("quant.data.akshare_provider._channel_eastmoney", fake_em)
    monkeypatch.setattr("quant.data.akshare_provider._channel_tencent", fake_em)
    monkeypatch.setattr("quant.data.akshare_provider._channel_sina", fake_sina)
    monkeypatch.setattr("quant.data.akshare_provider.time.sleep", lambda s: None)
    monkeypatch.setattr("quant.data.akshare_provider._consecutive_failures", 0, raising=False)

    df = provider.fetch_history("000001", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "qfq")
    assert not df.empty
    assert provider.last_source == "新浪"
    assert _channel_sina is not None


def test_akshare_all_channels_fail_reports_both(monkeypatch):
    """三条腿都挂时，错误信息要提到三条腿，便于定位。"""
    from datetime import datetime

    import pytest as _pytest

    from quant.data.akshare_provider import AkshareProvider

    provider = AkshareProvider(use_proxy=False)
    monkeypatch.setattr("quant.data.akshare_provider._channel_eastmoney",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("em down")))
    monkeypatch.setattr("quant.data.akshare_provider._channel_tencent",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("tx down")))
    monkeypatch.setattr("quant.data.akshare_provider._channel_sina",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("sina down")))
    monkeypatch.setattr("quant.data.akshare_provider.time.sleep", lambda s: None)
    monkeypatch.setattr("quant.data.akshare_provider._consecutive_failures", 0, raising=False)

    with _pytest.raises(RuntimeError, match="东方财富.*腾讯.*新浪|新浪.*腾讯.*东方财富"):
        provider.fetch_history("000001", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "qfq")
    assert provider.last_source is None


def test_akshare_channel_success_marks_eastmoney(monkeypatch):
    """东财成功时 last_source 固定为「东方财富」。"""
    from datetime import datetime

    import pandas as pd

    from quant.data.akshare_provider import AkshareProvider

    provider = AkshareProvider(use_proxy=False)

    def fake_em(*a, **k):
        return pd.DataFrame({"open": [1.0], "close": [1.0], "high": [1.0],
                             "low": [1.0], "volume": [100.0]},
                            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC")]))

    monkeypatch.setattr("quant.data.akshare_provider._channel_eastmoney", fake_em)
    df = provider.fetch_history("000001", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "qfq")
    assert not df.empty
    assert provider.last_source == "东方财富"


def _ok_frame():
    import pandas as pd
    return pd.DataFrame({"open": [1.0], "close": [1.0], "high": [1.0],
                         "low": [1.0], "volume": [100.0]},
                        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC")]))


def _dead(*a, **k):
    raise ConnectionError("down")


def test_akshare_health_source(monkeypatch):
    """健康检查逐通道探测，``source`` 报**通道链里第一个可用**的那条腿。

    三条腿并发探测，因此每条都要打桩——否则会打到真实网络。
    """
    from quant.data import akshare_provider as ak

    provider = ak.AkshareProvider(use_proxy=False)
    monkeypatch.setattr(ak.time, "sleep", lambda s: None)

    # 全部健康 → source 取链首（东方财富），且三条腿明细都在
    for name in ("_channel_eastmoney", "_channel_tencent", "_channel_sina"):
        monkeypatch.setattr(ak, name, lambda *a, **k: _ok_frame())
    h = provider.health_check()
    assert h.healthy is True
    assert h.source == "东方财富"
    assert [c["name"] for c in h.channels] == ["东方财富", "腾讯", "新浪"]
    assert all(c["healthy"] for c in h.channels)

    # 三条腿全挂 → 不健康
    for name in ("_channel_eastmoney", "_channel_tencent", "_channel_sina"):
        monkeypatch.setattr(ak, name, _dead)
    h2 = provider.health_check()
    assert h2.healthy is False
    assert h2.source is None

    # 东财挂、腾讯挂、只有新浪通 → 健康，source 标注为新浪（真实可用通道）
    monkeypatch.setattr(ak, "_channel_eastmoney", _dead)
    monkeypatch.setattr(ak, "_channel_tencent", _dead)
    monkeypatch.setattr(ak, "_channel_sina", lambda *a, **k: _ok_frame())
    h3 = provider.health_check()
    assert h3.healthy is True
    assert h3.source == "新浪"
    assert h3.channels[2]["preferred"] is True

    # 延迟只报主用通道自己的耗时，不是整条链累计值（本次回归点）
    assert h3.latency_ms == h3.channels[2]["latency_ms"]


def test_akshare_breaker_skips_dead_channel_fast(monkeypatch):
    """东财连接不上后应被熔断，后续拉取直接走新浪，不再重复等东财超时。

    背景：东财 kline 接口在本机被网络层阻断（详见 README 故障排查），
    若每次都先试东财两轮再退避，单次下载会白白多花数秒到数十秒。
    """
    from datetime import datetime

    import pandas as pd

    from quant.data import akshare_provider as ak
    from quant.data.akshare_provider import AkshareProvider, channel_status

    provider = AkshareProvider(use_proxy=False)
    # 三条腿各自计数：东财与腾讯都用 dead 打桩，但必须分开统计，
    # 否则 calls["em"] 会把腾讯那条腿的调用也算进去。
    calls = {"em": 0, "tx": 0, "sina": 0}

    def dead_em(*a, **k):
        calls["em"] += 1
        raise ConnectionError("Connection aborted / RemoteDisconnected")

    def dead_tx(*a, **k):
        calls["tx"] += 1
        raise ConnectionError("Connection aborted / RemoteDisconnected")

    def ok_sina(code, pref, secid, start, end, timeframe, adjust,
                timeout=10.0, throttle=True):
        calls["sina"] += 1
        return pd.DataFrame({"open": [1.0], "close": [1.0], "high": [1.0],
                             "low": [1.0], "volume": [100.0]},
                            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC")]))

    monkeypatch.setattr(ak, "_channel_eastmoney", dead_em)
    monkeypatch.setattr(ak, "_channel_tencent", dead_tx)
    monkeypatch.setattr(ak, "_channel_sina", ok_sina)
    monkeypatch.setattr(ak.time, "sleep", lambda s: None)

    # 第一次：东财打一次即被熔断（不做第二次尝试），新浪接住
    df1 = provider.fetch_history("000001", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "qfq")
    assert not df1.empty
    assert provider.last_source == "新浪"
    assert calls["em"] == 1, "传输层失败应立即熔断，不该重试第二次"
    assert "cooling" in channel_status()["东方财富"]

    # 第二次：东财、腾讯均已熔断 → 两条腿一次都不再调用，直达新浪
    df2 = provider.fetch_history("000001", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "qfq")
    assert not df2.empty
    assert calls["em"] == 1, "熔断期内不应再碰东财"
    assert calls["tx"] == 1, "熔断期内不应再碰腾讯"


def test_akshare_breaker_expires_after_cooldown(monkeypatch):
    """熔断到期后应自动恢复尝试，而不是永久跳过该通道。"""
    import time as _time
    from datetime import datetime

    import pandas as pd

    from quant.data import akshare_provider as ak
    from quant.data.akshare_provider import AkshareProvider, channel_status

    provider = AkshareProvider(use_proxy=False)
    ak._mark_channel_down("东方财富")
    assert "cooling" in channel_status()["东方财富"]

    # 把恢复时间改到过去，模拟冷却结束
    ak._channel_down_until["东方财富"] = _time.monotonic() - 1
    assert channel_status()["东方财富"] == "ok"

    def ok_em(*a, **k):
        return pd.DataFrame({"open": [1.0], "close": [1.0], "high": [1.0],
                             "low": [1.0], "volume": [100.0]},
                            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC")]))

    monkeypatch.setattr(ak, "_channel_eastmoney", ok_em)
    df = provider.fetch_history("000001", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "qfq")
    assert not df.empty
    assert provider.last_source == "东方财富"


def test_akshare_health_ignores_breaker(monkeypatch):
    """健康检查不受熔断影响：用户点「重新检测」时必须真的去探一次，才能自愈。"""
    import pandas as pd

    from quant.data import akshare_provider as ak
    from quant.data.akshare_provider import AkshareProvider

    provider = AkshareProvider(use_proxy=False)
    ak._mark_channel_down("东方财富")

    def ok_em(*a, **k):
        return pd.DataFrame({"open": [1.0], "close": [1.0], "high": [1.0],
                             "low": [1.0], "volume": [100.0]},
                            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC")]))

    monkeypatch.setattr(ak, "_channel_eastmoney", ok_em)
    h = provider.health_check()
    assert h.healthy is True
    assert h.source == "东方财富"
    assert ak.channel_status()["东方财富"] == "ok"


def test_akshare_clip_to_window_handles_tz_aware_bounds():
    """回归：请求边界是 tz-aware(UTC) 而通道索引是 tz-naive 时不得抛 TypeError。

    这是 1m 下载全挂的真因——`_clip_to_window` 只在**索引**带时区时才对齐边界，
    索引 tz-naive 而 start/end 带 UTC 时区时直接比较，pandas 抛
    ``Invalid comparison between dtype=datetime64[us] and Timestamp``；
    该异常又被 fallback 链捕获成「腾讯通道失败」而重试，掩盖了真正的问题。
    """
    from datetime import datetime, timedelta, timezone

    import pandas as pd

    from quant.data.akshare_provider import _clip_to_window

    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 09:30"), pd.Timestamp("2024-01-02 15:00"),
         pd.Timestamp("2024-06-03 10:00")],
        dtype="datetime64[us]",
    )
    df = pd.DataFrame({"open": [1.0, 2.0, 3.0]}, index=idx)

    # tz-aware(UTC) 边界 → naive(北京时间) 索引：换算后应正常裁剪
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    out = _clip_to_window(df, start, end)
    assert len(out) == 2
    assert out.index[0] == pd.Timestamp("2024-01-02 09:30")

    # tz-naive 边界同样可用（此路径原本就 OK，防回归）
    out2 = _clip_to_window(df, datetime(2024, 6, 1), datetime(2024, 6, 30))
    assert len(out2) == 1
    assert out2.index[0] == pd.Timestamp("2024-06-03 10:00")

    # 索引带时区时，naive 边界要被 localize 到索引时区
    idx_tz = idx.tz_localize("Asia/Shanghai")
    df_tz = pd.DataFrame({"open": [1.0, 2.0, 3.0]}, index=idx_tz)
    out3 = _clip_to_window(df_tz, datetime(2024, 1, 1), datetime(2024, 1, 31))
    assert len(out3) == 2
    assert out3.index.tz is not None

    # 区间外无数据 → 空表（调用方据此报「该区间内无数据」）
    empty = _clip_to_window(df, datetime(2020, 1, 1), datetime(2020, 2, 1))
    assert empty.empty


def test_akshare_transport_error_classification():
    """传输层错误（换通道）与限流错误（原地退避）要能区分开。"""
    import requests

    from quant.data.akshare_provider import _is_transport_error

    assert _is_transport_error(ConnectionError("x")) is True
    assert _is_transport_error(requests.exceptions.ProxyError("p")) is True
    assert _is_transport_error(requests.exceptions.Timeout("t")) is True
    assert _is_transport_error(requests.exceptions.SSLError("s")) is True
    assert _is_transport_error(RuntimeError("Connection aborted.")) is True
    assert _is_transport_error(RuntimeError("RemoteDisconnected")) is True
    # 限流 / 服务端 5xx 不算传输失败 → 应原地退避重试
    assert _is_transport_error(RuntimeError("HTTP 429 Too Many Requests")) is False
    assert _is_transport_error(RuntimeError("服务端 500")) is False


def test_akshare_rejects_non_a_share():
    """非 A股/ETF 代码快速失败（不拖慢 fallback 链）。"""
    import pytest as _pytest

    from quant.data.akshare_provider import AkshareProvider

    p = AkshareProvider(use_proxy=False)
    with _pytest.raises(ValueError, match="非 A股"):
        p.fetch_history("AAPL", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "none")


def test_akshare_supports_matrix():
    """akshare 能力声明：含 1m（腾讯腿独家提供）、无 2h/4h；日/周/月/分钟（1-60）支持。"""
    from quant.data.akshare_provider import AkshareProvider

    p = AkshareProvider(use_proxy=False)
    for tf in ("1m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"):
        assert p.supports("600519.SH", tf) is True, tf
    for tf in ("2h", "4h"):
        assert p.supports("600519.SH", tf) is False, tf
    # 1m 只能靠腾讯腿，回溯窗口被裁到很短（约 4 个交易日）
    assert p.max_lookback_days("600519.SH", "1m") == 7
    assert p.max_lookback_days("600519.SH", "1d") is None
    # 非 A股代码不支持
    assert p.supports("AAPL", "1d") is False


# ---------- 腾讯（A股最后兜底） ----------

def _tx_payload(rows: list[list], key: str = "qfqday"):
    return {"code": 0, "data": {"sh600519": {key: rows}}}


class _FakeResp:
    def __init__(self, payload):
        self._p = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


def test_tencent_code_and_parsing(monkeypatch):
    """腾讯字段序是 date/open/close/high/low/volume —— 按 OHLC 读会把 high/low 读反。"""
    from quant.data.tencent_provider import TencentProvider, _to_tx_code

    assert _to_tx_code("600519.SH") == "sh600519"
    assert _to_tx_code("000001.SZ") == "sz000001"
    assert _to_tx_code("600519") == "sh600519"

    p = TencentProvider()
    df = p._rows_to_df([
        ["2024-01-02", "7.89", "7.74", "7.91", "7.74", "1158366"],
        ["2024-01-03", "8.00", "8.10", "8.20", "7.95", "2000"],
    ])
    assert df.iloc[0]["open"] == 7.89
    assert df.iloc[0]["close"] == 7.74
    assert df.iloc[0]["high"] == 7.91   # 第 4 列才是 high
    assert df.iloc[0]["low"] == 7.74    # 第 5 列才是 low
    assert df.iloc[0]["volume"] == 1158366 * 100  # 手→股


def test_tencent_daily_fetch(monkeypatch):
    """日线走 fqkline/get，按 end 回退分页，去重合并。"""
    from datetime import datetime

    import requests

    from quant.data.tencent_provider import TencentProvider

    pages = [
        _tx_payload([["2024-01-04", "10", "11", "12", "9", "100"],
                     ["2024-01-05", "11", "12", "13", "10", "200"]]),
        _tx_payload([["2024-01-02", "8", "9", "10", "7", "300"],
                     ["2024-01-03", "9", "10", "11", "8", "400"]]),
        _tx_payload([], "day"),
    ]
    calls = {"n": 0}

    def fake_get(url, params, timeout=None, headers=None, proxies=None):
        if "fqkline" in url:
            payload = pages[min(calls["n"], len(pages) - 1)]
            calls["n"] += 1
            return _FakeResp(payload)
        raise AssertionError(url)

    monkeypatch.setattr(requests, "get", fake_get)
    p = TencentProvider()
    df = p.fetch_history("600519.SH", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "qfq")
    assert len(df) == 4  # 两页去重合并
    # 索引统一存 UTC，按上海时区还原才是 A股本地日期
    assert str(df.index.min().tz_convert("Asia/Shanghai").date()) == "2024-01-02"
    assert calls["n"] >= 1


def test_tencent_minute_uses_mkline(monkeypatch):
    """分钟线必须走 mkline 端点（fqkline 只认 day/week/month）。"""
    from datetime import datetime

    import requests

    from quant.data.tencent_provider import TencentProvider

    seen: list[str] = []

    def fake_get(url, params, timeout=None, headers=None, proxies=None):
        seen.append(url)
        assert "mkline" in url, f"分钟线不应走 {url}"
        return _FakeResp(_tx_payload(
            [["202401021000", "10", "10.5", "10.6", "9.9", "100"],
             ["202401021005", "10.5", "10.6", "10.7", "10.4", "120"]],
            "m5"))

    monkeypatch.setattr(requests, "get", fake_get)
    p = TencentProvider()
    df = p.fetch_history("600519.SH", datetime(2024, 1, 1), datetime(2024, 1, 10), "5m", "qfq")
    assert len(df) == 2
    # 202401021000 是北京时间 10:00 → 存成 UTC 后是 02:00
    local = df.index[0].tz_convert("Asia/Shanghai")
    assert local.hour == 10 and local.minute == 0
    assert all("mkline" in u for u in seen)


def test_tencent_supports_and_lookback():
    """腾讯能力声明：无 2h/4h；分钟线回溯很浅（约 640 根）。"""
    from quant.data.tencent_provider import TencentProvider

    p = TencentProvider()
    for tf in ("1m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"):
        assert p.supports("600519.SH", tf) is True, tf
    for tf in ("2h", "4h"):
        assert p.supports("600519.SH", tf) is False, tf
    # 分钟线有明确回溯上限，日/周/月不限
    assert p.max_lookback_days("600519.SH", "5m") is not None
    assert p.max_lookback_days("600519.SH", "1d") is None


def test_tencent_rejects_non_a_share():
    import pytest as _pytest

    from quant.data.tencent_provider import TencentProvider, _to_tx_code

    p = TencentProvider()
    with _pytest.raises(ValueError, match="仅支持沪深"):
        _to_tx_code("AAPL")
    assert p.supports("AAPL", "1d") is False


# ---------- baostock（A股主源） ----------

class _FakeLogin:
    """模拟 bs.login() 返回。"""

    def __init__(self, error_code="0", error_msg="success"):
        self.error_code = error_code
        self.error_msg = error_msg
        self.login_id = "fake-id"


class _FakeResult:
    """模拟 baostock 查询结果集（真实对象以 .fields 属性返回字段名，无 get_fields()）。"""

    def __init__(self, rows=None, error_code="0", error_msg="", fields=None):
        self.error_code = error_code
        self.error_msg = error_msg
        self._rows = rows or []
        self.fields = fields.split(",") if fields else []  # 与真实 baostock 一致
        self._i = 0

    def next(self):
        if self._i < len(self._rows):
            return True
        return False

    def get_row_data(self):
        row = self._rows[self._i]
        self._i += 1
        return row


class _FakeBs:
    """模拟 baostock 模块（记录调用参数，返回可控行数据）。"""

    def __init__(self, rows=None, login_error="0"):
        # 默认日线行（fields: date,code,open,high,low,close,preclose,volume,amount）
        self.rows = rows if rows is not None else [
            ["2024-01-02", "sz.000001", "7.89", "7.91", "7.74", "7.86",
             "7.88", "1158366", "91234567"],
            ["2024-01-03", "sz.000001", "7.86", "8.00", "7.80", "7.95",
             "7.86", "1300000", "103000000"],
        ]
        self.login_error = login_error
        self.login_calls = 0
        self.logout_calls = 0
        self.last_call = None

    def login(self):
        self.login_calls += 1
        return _FakeLogin(error_code=self.login_error)

    def logout(self):
        self.logout_calls += 1

    def query_history_k_data_plus(self, code, fields, start_date, end_date,
                                  frequency, adjustflag):
        self.last_call = {"code": code, "fields": fields, "start_date": start_date,
                          "end_date": end_date, "frequency": frequency,
                          "adjustflag": adjustflag}
        return _FakeResult(self.rows, fields=fields)


def test_baostock_symbol_normalize():
    from quant.data.baostock_provider import _to_baostock_code

    # 股票
    assert _to_baostock_code("000001") == "sh.000001"  # 无后缀 000 → 上证指数
    assert _to_baostock_code("600519") == "sh.600519"
    assert _to_baostock_code("000001.SZ") == "sz.000001"  # 带 .SZ → 深市（平安银行股票）
    assert _to_baostock_code("600519.SH") == "sh.600519"
    assert _to_baostock_code("300750.SZ") == "sz.300750"  # 创业板
    assert _to_baostock_code("688981.SH") == "sh.688981"  # 科创板
    # ETF（51/56/58 沪、15/16 深）
    assert _to_baostock_code("510300") == "sh.510300"
    assert _to_baostock_code("510300.SH") == "sh.510300"
    assert _to_baostock_code("588000.SH") == "sh.588000"
    assert _to_baostock_code("159915") == "sz.159915"
    assert _to_baostock_code("159915.SZ") == "sz.159915"
    # 指数
    assert _to_baostock_code("000001.SH") == "sh.000001"  # 上证指数
    assert _to_baostock_code("000300.SH") == "sh.000300"  # 沪深300
    assert _to_baostock_code("399001.SZ") == "sz.399001"  # 深证成指
    assert _to_baostock_code("399006.SZ") == "sz.399006"
    with pytest.raises(ValueError):
        _to_baostock_code("AAPL")  # 非 A股
    with pytest.raises(ValueError):
        _to_baostock_code("830799.BJ")  # 北交所不支持


def test_baostock_rows_to_df_offline():
    """原始行按字段名解析 → 统一 OHLCV（成交量单位：股，无需 ×100）。

    日线布局（date 起，无 time）与分钟线布局（date+time 起，无 preclose）各自正确。
    """
    from quant.data.baostock_provider import _rows_to_df

    # 日线：date,code,open,high,low,close,preclose,volume,amount
    daily_fields = ["date", "code", "open", "high", "low", "close",
                    "preclose", "volume", "amount"]
    daily_rows = [
        ["2024-01-02", "sz.000001", "7.89", "7.91", "7.74", "7.86",
         "7.88", "1158366", "91234567"],
        ["2024-01-03", "sz.000001", "7.86", "8.00", "7.80", "7.95",
         "7.86", "1300000", "103000000"],
    ]
    df = _rows_to_df(daily_fields, daily_rows)
    assert len(df) == 2
    assert df.iloc[0]["close"] == 7.86
    assert df.iloc[0]["volume"] == 1158366  # 股
    assert str(df.index[0]) == "2024-01-02 00:00:00"  # 日线 date

    # 分钟线：date,time,code,open,high,low,close,volume,amount
    minute_fields = ["date", "time", "code", "open", "high", "low", "close",
                     "volume", "amount"]
    minute_rows = [
        ["2024-01-03", "20240103093500000", "sz.000001", "7.86", "8.00", "7.80",
         "7.95", "1300000", "103000000"],
    ]
    df2 = _rows_to_df(minute_fields, minute_rows)
    assert len(df2) == 1
    assert str(df2.index[0]) == "2024-01-03 09:35:00"  # 分钟线 date+time（毫秒截断）
    assert df2.iloc[0]["volume"] == 1300000


def test_baostock_fetch_history_offline(monkeypatch):
    """离线拉取：登录一次 → 查询 → 不登出（会话保持）+ qfq → adjustflag=2 + UTC 归一化。

    日线查询字段**不含 time**（baostock 日线传 time 会报 10004012）；
    无后缀 000001 归为上证指数（sh.000001），故用 000001.SZ（平安银行股票）验证股票路径。

    会话保持是性能必需：实测 ``login()`` 单次 11~13 秒、而一次 query 只要 2~3 秒，
    「每条查询都 login→query→logout」会让每次拉数据白付十几秒（这正是历史上
    「baostock 总超时」的真正原因）。因此这里断言 logout 次数为 0。
    """
    from quant.data import baostock_provider as bp

    fake = _FakeBs()
    monkeypatch.setattr(bp, "bs", fake)
    p = bp.BaostockProvider()
    df = p.fetch_history("000001.SZ", datetime(2024, 1, 1), datetime(2024, 1, 10),
                         "1d", "qfq")
    assert fake.login_calls == 1  # 首次用到才登录
    assert fake.logout_calls == 0  # 会话保持，查询结束不登出
    assert bp.session_state()["logged_in"] is True
    assert fake.last_call["code"] == "sz.000001"
    assert fake.last_call["frequency"] == "d"
    assert fake.last_call["adjustflag"] == "2"  # qfq
    assert "time" not in fake.last_call["fields"]  # 日线禁止 time 字段
    assert "preclose" in fake.last_call["fields"]
    assert fake.last_call["start_date"] == "2024-01-01"  # YYYY-MM-DD 字符串
    assert len(df) == 2
    assert df.index.tz is not None  # 已转 UTC
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "adj_factor"]
    assert p.last_source == "baostock"

    # 第二次拉取复用已有会话：不再登录
    p.fetch_history("600519.SH", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "qfq")
    assert fake.login_calls == 1


def test_baostock_session_reused_across_providers(monkeypatch):
    """会话是模块级、跨 provider 实例共享的（进程内只登录一次）。"""
    from quant.data import baostock_provider as bp

    fake = _FakeBs()
    monkeypatch.setattr(bp, "bs", fake)
    bp.BaostockProvider().fetch_history(
        "600519.SH", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "qfq")
    assert fake.login_calls == 1
    bp.BaostockProvider().fetch_history(
        "000001.SZ", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "qfq")
    assert fake.login_calls == 1  # 新实例也不重新登录

    bp.close_session()
    assert fake.logout_calls == 1
    assert bp.session_state()["logged_in"] is False


def test_baostock_query_retries_after_relogin(monkeypatch):
    """查询返回错误码 → 重建会话并重试一次，而不是立刻把 provider 判死。

    baostock 服务器会中途掉线（实测长查询跑到一半返回 10002007 网络接收错误，
    之后所有查询都失败直到重新 login）。
    """
    from quant.data import baostock_provider as bp

    fake = _FakeBs()
    calls = {"q": 0}
    orig = fake.query_history_k_data_plus

    def flaky(code, fields, start_date, end_date, frequency, adjustflag):
        calls["q"] += 1
        if calls["q"] == 1:
            return _FakeResult([], error_code="10002007", error_msg="网络接收错误。")
        return orig(code, fields, start_date, end_date, frequency, adjustflag)

    monkeypatch.setattr(fake, "query_history_k_data_plus", flaky)
    monkeypatch.setattr(bp, "bs", fake)
    p = bp.BaostockProvider()
    df = p.fetch_history("000001.SZ", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "qfq")
    assert len(df) == 2  # 重试后成功
    assert calls["q"] == 2
    assert fake.login_calls == 2  # 首次登录 + 掉线后重登


def test_baostock_minute_forced_no_adjust(monkeypatch, caplog):
    """分钟线不支持复权：请求 qfq 时按不复权拉取并告警。"""
    from quant.data import baostock_provider as bp

    fake = _FakeBs()
    monkeypatch.setattr(bp, "bs", fake)
    p = bp.BaostockProvider()
    with caplog.at_level(logging.WARNING, logger="quant.data.baostock_provider"):
        p.fetch_history("600519", datetime(2024, 1, 1), datetime(2024, 1, 3),
                        "5m", "qfq")
    assert fake.last_call["frequency"] == "5"
    assert fake.last_call["adjustflag"] == "3"  # 分钟线强制不复权
    assert "time" in fake.last_call["fields"]  # 分钟线含 time 字段
    assert "preclose" not in fake.last_call["fields"]  # preclose 仅日线级有效
    assert "不支持复权" in caplog.text


def test_baostock_weekly_monthly_fields(monkeypatch):
    """周线/月线字段不含 preclose 与 time（实测传 preclose 报 10004012）。"""
    from quant.data import baostock_provider as bp

    fake = _FakeBs()
    monkeypatch.setattr(bp, "bs", fake)
    p = bp.BaostockProvider()
    p.fetch_history("600519", datetime(2024, 1, 1), datetime(2024, 1, 31), "1w", "hfq")
    assert fake.last_call["frequency"] == "w"
    assert fake.last_call["adjustflag"] == "1"  # hfq
    assert "preclose" not in fake.last_call["fields"]
    assert "time" not in fake.last_call["fields"]
    p.fetch_history("600519", datetime(2024, 1, 1), datetime(2024, 1, 31), "1M", "none")
    assert fake.last_call["frequency"] == "m"
    assert "preclose" not in fake.last_call["fields"]


def test_baostock_etf_fetch_offline(monkeypatch):
    """ETF 代码 → sh.51xxxx / sz.15xxxx，查询字段与股票同周期（日线含 preclose 不含 time）。"""
    from quant.data import baostock_provider as bp

    assert bp.BaostockProvider.asset_types == ("stock", "etf", "index")
    fake = _FakeBs()
    monkeypatch.setattr(bp, "bs", fake)
    p = bp.BaostockProvider()
    df = p.fetch_history("510300.SH", datetime(2026, 1, 5), datetime(2026, 1, 10),
                         "1d", "qfq")
    assert fake.last_call["code"] == "sh.510300"
    assert fake.last_call["frequency"] == "d"
    assert fake.last_call["adjustflag"] == "2"
    assert "preclose" in fake.last_call["fields"] and "time" not in fake.last_call["fields"]
    assert len(df) == 2 and p.last_source == "baostock"

    p.fetch_history("159915.SZ", datetime(2026, 1, 5), datetime(2026, 1, 10),
                    "5m", "none")
    assert fake.last_call["code"] == "sz.159915"
    assert fake.last_call["frequency"] == "5"
    assert "time" in fake.last_call["fields"] and "preclose" not in fake.last_call["fields"]


def test_baostock_index_fetch_offline(monkeypatch, caplog):
    """指数代码 → sh.000xxx / sz.399xxx；请求复权时按不复权处理并告警。"""
    from quant.data import baostock_provider as bp

    fake = _FakeBs()
    monkeypatch.setattr(bp, "bs", fake)
    p = bp.BaostockProvider()
    with caplog.at_level(logging.WARNING, logger="quant.data.baostock_provider"):
        df = p.fetch_history("000001.SH", datetime(2024, 1, 1), datetime(2024, 1, 10),
                             "1d", "qfq")
    assert fake.last_call["code"] == "sh.000001"
    assert fake.last_call["adjustflag"] == "3"  # 指数无复权 → 强制不复权
    assert "指数无复权概念" in caplog.text
    assert len(df) == 2

    p.fetch_history("399001.SZ", datetime(2024, 1, 1), datetime(2024, 1, 10),
                    "1w", "none")
    assert fake.last_call["code"] == "sz.399001"
    assert fake.last_call["frequency"] == "w"


def test_baostock_index_minute_rejected(monkeypatch):
    """指数不支持分钟线：快速失败并提示走 akshare/yfinance fallback。"""
    from quant.data import baostock_provider as bp

    monkeypatch.setattr(bp, "bs", _FakeBs())
    p = bp.BaostockProvider()
    with pytest.raises(ValueError, match="指数不支持分钟线"):
        p.fetch_history("000001.SH", datetime(2024, 1, 1), datetime(2024, 1, 10),
                        "5m", "none")


def test_baostock_health_source(monkeypatch):
    """健康检查：探测 sz.000001 并记录 source=baostock。"""
    from quant.data import baostock_provider as bp

    fake = _FakeBs()
    monkeypatch.setattr(bp, "bs", fake)
    p = bp.BaostockProvider()
    h = p.health_check()
    assert h.healthy is True
    assert h.source == "baostock"
    assert fake.last_call["code"] == "sz.000001"


def test_baostock_errors(monkeypatch):
    """非 A股代码 / 不支持周期 / 空数据 / 登录失败 → 明确报错。"""
    from quant.data import baostock_provider as bp

    monkeypatch.setattr(bp, "bs", _FakeBs())
    p = bp.BaostockProvider()
    # 非 A股代码（无需网络，校验即失败）
    with pytest.raises(ValueError, match="非 A股"):
        p.fetch_history("AAPL", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "none")
    # 不支持周期（2h 已从周期表移除，仍应明确报错）
    with pytest.raises(ValueError, match="不支持的周期"):
        p.fetch_history("000001", datetime(2024, 1, 1), datetime(2024, 1, 10), "1m", "none")
    with pytest.raises(ValueError, match="不支持的周期"):
        p.fetch_history("000001", datetime(2024, 1, 1), datetime(2024, 1, 10), "2h", "none")
    # 空数据
    fake_empty = _FakeBs(rows=[])
    monkeypatch.setattr(bp, "bs", fake_empty)
    with pytest.raises(RuntimeError, match="空数据"):
        p.fetch_history("000001", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "none")
    # 登录失败
    # 注意：会话是模块级全局状态，上面几次成功查询已把 _logged_in 置为 True；
    # 这里换了新的假模块，必须先丢弃旧会话，否则 _ensure_login 会直接复用而不登录。
    bp.close_session()
    fake_fail = _FakeBs(login_error="10001002")
    monkeypatch.setattr(bp, "bs", fake_fail)
    with pytest.raises(RuntimeError, match="登录失败"):
        p.fetch_history("000001", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "none")
    assert fake_fail.login_calls == 1


@pytest.mark.skipif(importlib.util.find_spec("baostock") is None,
                    reason="baostock 未安装，跳过真实拉取测试")
def test_baostock_real_fetch_000001():
    """真实拉取 000001（平安银行）日线：验证 baostock 连通性与复权数据。

    baostock 服务器偶尔抽风（``10002007 网络接收错误``）或拒绝登录
    （``10001011``），这是服务器侧问题、与本项目代码无关；集成测试不该因此
    把 CI 判红，故遇到网络/服务端拒绝类错误时 skip。
    """
    from datetime import datetime, timedelta

    from quant.data.baostock_provider import BaostockProvider

    provider = BaostockProvider()
    end = datetime.now()
    start = end - timedelta(days=45)
    try:
        df = provider.fetch_history("000001", start, end, "1d", "qfq")
    except RuntimeError as e:
        msg = str(e)
        if ("网络接收错误" in msg or "超时" in msg or "登录失败" in msg
                or "空数据" in msg or "拒绝登录" in msg or "换个网络环境" in msg
                or "连接失败" in msg):
            pytest.skip(f"baostock 服务器暂不可用，跳过真实拉取测试: {msg}")
        raise
    assert df is not None and len(df) >= 10
    assert provider.last_source == "baostock"
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert df["close"].iloc[-1] > 0


def test_baostock_capabilities():
    """baostock 能力声明：无 1m、无 2h/4h；指数无分钟线；北交所不支持。"""
    from quant.data.baostock_provider import BaostockProvider

    p = BaostockProvider()
    for tf in ("5m", "15m", "30m", "1h", "1d", "1w", "1M"):
        assert p.supports("600519.SH", tf) is True, tf
    for tf in ("1m", "2h", "4h"):
        assert p.supports("600519.SH", tf) is False, tf
    # 指数无分钟线（日/周/月可以）
    assert p.supports("000300.SH", "5m") is False
    assert p.supports("000300.SH", "1d") is True
    # 北交所 / 非 A股
    assert p.supports("830799.BJ", "1d") is False
    assert p.supports("AAPL", "1d") is False


def test_tencent_merged_into_akshare_not_in_chain():
    """腾讯已并入 akshare 内部（东财→腾讯→新浪），不再作为独立 provider 出现在优先级链上。

    否则健康表会把同一家腾讯显示两遍，代理开关也要设两处。
    TencentProvider 类本身仍保留，供直接引用与测试。
    """
    from quant.core.config import AppConfig
    from quant.data.tencent_provider import TencentProvider

    prio = AppConfig().data.providers
    for at in ("stock", "etf", "index"):
        chain = getattr(prio, at)
        assert chain == ["baostock", "akshare"], f"{at} 链应只有两级，实际 {chain}"
    # 类仍在，可独立实例化
    assert TencentProvider().name == "tencent"

    # akshare 内部确实存在腾讯这条腿
    from quant.data.akshare_provider import AkshareProvider
    labels = [label for label, _ in AkshareProvider()._channels("1d")]
    assert "腾讯" in labels


def test_health_always_includes_all_providers(tmp_config):
    """健康检查必须始终包含所有注册数据源：慢/失败的也显示为超时/不可用，不能丢弃。"""
    import time

    from quant.data.manager import DataManager
    from quant.data.storage import MarketStore
    from quant.storage.db import init_db
    from quant.storage.repositories import MarketDataRepo, SymbolsRepo

    class SlowProvider(FakeProvider):
        name = "slow"

        def _ping(self) -> None:
            time.sleep(5)  # 模拟慢数据源（超过健康检查预算）

    init_db(tmp_config.effective_db_path)
    store = MarketStore(tmp_config.effective_market_dir)
    dm = DataManager(tmp_config, store, MarketDataRepo(tmp_config.effective_db_path),
                     SymbolsRepo(tmp_config.effective_db_path),
                     providers={"fast": FakeProvider(), "slow": SlowProvider()})
    h = dm.health(timeout=1)  # 1s 预算，slow 必然超时
    assert set(h.keys()) == {"fast", "slow"}  # 两个都展示
    assert h["fast"].healthy is True
    assert h["slow"].healthy is False
    assert "超时" in (h["slow"].error or "")


def test_load_bars_slice(manager):
    manager.update(symbols=["TEST"], timeframes=["1d"])
    df = manager.load_bars("TEST", "1d")
    assert len(df) == 10
    df2 = manager.load_bars("TEST", "1d", start=df.index[5].strftime("%Y-%m-%d"))
    assert len(df2) <= 5


# ---------- 多周期 / 分钟级增量边界 ----------

def test_multi_timeframe_incremental_no_dupe(manager):
    """多个周期各自增量更新，均无重复。"""
    for tf in ["1m", "1h", "1w", "1M"]:
        r1 = manager.update(symbols=["TEST"], timeframes=[tf])[0]
        r2 = manager.update(symbols=["TEST"], timeframes=[tf])[0]
        assert r1.ok and r2.ok, tf
        assert r2.rows == r1.rows, tf
        df = manager.store.load("TEST", tf)
        assert df.index.is_unique, tf
        assert df.index.is_monotonic_increasing, tf


def test_minute_incremental_overlap_boundary(manager):
    """分钟级增量边界：第二次更新窗口与第一次重叠 61 根，合并后去重正确。

    第一次窗口 [S0, S0+99m]（100 根）；增量起点 = 最后 bar − 60 分钟重叠 →
    第二次窗口 [S0+39m, S0+138m]（100 根）。重叠 61 根，合并后应为
    100 + 100 − 61 = 139 根，且无重复。
    """
    dm = manager
    dm.providers["fake"] = FakeProvider(periods=100, freq="1min", use_start=True)
    r1 = dm.update(symbols=["TEST"], timeframes=["1m"])[0]
    assert r1.ok and r1.rows == 100
    df1 = dm.store.load("TEST", "1m")
    s0 = df1.index.min()

    r2 = dm.update(symbols=["TEST"], timeframes=["1m"])[0]
    assert r2.ok
    df2 = dm.store.load("TEST", "1m")
    assert len(df2) == 139  # 100 + 100 - 61 重叠
    assert df2.index.is_unique
    assert df2.index.min() == s0
    assert df2.index.max() > df1.index.max()  # 有新 bar 追加


def test_full_reload_replaces(tmp_config):
    """--full：清空本地数据后重拉，旧数据窗口不残留（复权切换场景）。"""
    from quant.data.manager import DataManager
    from quant.data.storage import MarketStore
    from quant.storage.db import init_db
    from quant.storage.repositories import MarketDataRepo, SymbolsRepo

    init_db(tmp_config.effective_db_path)
    tmp_config.data.providers.stock = ["fake"]
    store = MarketStore(tmp_config.effective_market_dir)
    dm = DataManager(tmp_config, store, MarketDataRepo(tmp_config.effective_db_path),
                     SymbolsRepo(tmp_config.effective_db_path),
                     providers={"fake": FakeProvider(window_start="2024-01-01", periods=10)})
    dm.symbols_repo.upsert(symbol="TEST", name="Test", asset_type="stock", source="fake")

    r1 = dm.update(symbols=["TEST"], timeframes=["1d"])[0]
    assert r1.ok and r1.rows == 10

    # 复权方式变化 → 全量重刷，数据窗口变为另一段
    dm.providers["fake"] = FakeProvider(window_start="2024-02-01", periods=5)
    r2 = dm.update(symbols=["TEST"], timeframes=["1d"], full_reload=True)[0]
    assert r2.ok and r2.rows == 5
    df = dm.store.load("TEST", "1d")
    assert len(df) == 5  # 旧窗口数据已被清除
    assert str(df.index.min().date()) == "2024-02-01"


def test_start_date_backfill(manager):
    """--start：从指定日期开始补拉历史。"""
    manager.providers["fake"] = FakeProvider(periods=10, freq="D", use_start=True)
    r = manager.update(symbols=["TEST"], timeframes=["1d"], start_date="2024-03-01")[0]
    assert r.ok and r.rows == 10
    df = manager.store.load("TEST", "1d")
    assert str(df.index.min().date()) == "2024-03-01"


def test_health_check(manager):
    health = manager.health()
    assert "fake" in health
    assert health["fake"].healthy is True


# ---------- provider 离线兼容（不联网） ----------

def test_yfinance_tf_map_and_finalize():
    from quant.data.yfinance_provider import TF_MAP, YFinanceProvider

    # 2h/4h 已移除：Yahoo 对这两个粒度返回 400/422，根本拿不到数据
    assert set(TF_MAP) == {"1m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"}
    assert "2h" not in TF_MAP and "4h" not in TF_MAP
    p = YFinanceProvider()
    df = pd.DataFrame(
        {"Open": [1.0], "High": [1.1], "Low": [0.9], "Close": [1.0], "Volume": [100],
         "Adj Close": [0.95]},
        index=pd.DatetimeIndex(["2024-01-02 14:30:00"], tz="America/New_York"),
    )
    out = p._from_yf_df(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "adj_factor"]
    assert out.index.tz is not None  # 已转 UTC
    assert out.iloc[0]["adj_factor"] == pytest.approx(0.95)


def test_yfinance_capabilities_and_window_clip():
    """Yahoo 分钟线有硬窗口：1m 8 天、5/15/30m 60 天、60m 730 天；超窗必须自动裁剪。"""
    from datetime import datetime, timedelta

    from quant.data.yfinance_provider import YFinanceProvider

    p = YFinanceProvider()
    for tf in ("1m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"):
        assert p.supports("AAPL", tf) is True, tf
    assert p.supports("AAPL", "2h") is False and p.supports("AAPL", "4h") is False

    assert p.max_lookback_days("AAPL", "1m") == 8
    assert p.max_lookback_days("AAPL", "5m") == 60
    assert p.max_lookback_days("AAPL", "1h") == 730
    assert p.max_lookback_days("AAPL", "1d") is None  # 日线不限

    end = datetime(2024, 6, 1)
    far = end - timedelta(days=3650)
    clipped = p._clip_start(far, end, "5m")
    assert clipped == end - timedelta(days=59)   # 裁进 60 天窗口
    assert p._clip_start(far, end, "1d") == far  # 日线不裁


def test_akshare_col_map_finalize():
    from quant.data.akshare_provider import AkshareProvider

    p = AkshareProvider()
    # 模拟 akshare 中文列 → 统一格式（成交量 手 → 股 ×100）
    raw = pd.DataFrame({
        "日期": ["2024-01-02"], "开盘": [10.0], "收盘": [10.5],
        "最高": [10.8], "最低": [9.9], "成交量": [10000],
    })
    out = p._finalize(
        pd.DataFrame({
            "datetime": pd.to_datetime(raw["日期"]),
            "open": raw["开盘"], "close": raw["收盘"],
            "high": raw["最高"], "low": raw["最低"],
            "volume": raw["成交量"] * 100,
        }).set_index("datetime"),
        tz="Asia/Shanghai",
    )
    assert len(out) == 1
    assert out.index.tz is not None  # 已从 Asia/Shanghai 转 UTC
    assert out.iloc[0]["volume"] == 1_000_000


def test_ccxt_module_removed():
    """ccxt 数据源已删除（用户要求只保留 yfinance + baostock + akshare + 腾讯）。"""
    import importlib.util

    assert importlib.util.find_spec("quant.data.ccxt_provider") is None


def test_timeframe_filename_case_safety(tmp_config):
    """Windows 文件系统大小写不敏感：1m 与 1M 必须落到不同文件。"""
    from quant.data.storage import MarketStore

    store = MarketStore(tmp_config.effective_market_dir)
    p1 = store.path_for("X", "1m")
    p2 = store.path_for("X", "1M")
    assert p1.name != p2.name  # 1m → "1m"，1M → "1mo"
    store.save("X", "1m", _fake_minute_frame())
    store.save("X", "1M", _fake_monthly_frame())
    m1 = store.load("X", "1m")
    m2 = store.load("X", "1M")
    assert len(m1) == 2 and len(m2) == 2  # 各自独立，未互相污染


def _fake_minute_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-08-14 09:30", periods=2, freq="1min", tz="UTC")
    return pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0,
                         "volume": 1.0, "adj_factor": 1.0}, index=idx)


def _fake_monthly_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-08-01", periods=2, freq="1MS", tz="UTC")
    return pd.DataFrame({"open": 2.0, "high": 2.1, "low": 1.9, "close": 2.0,
                         "volume": 2.0, "adj_factor": 1.0}, index=idx)


def test_api_health_environment():
    """/api/health 返回运行环境标识（local/sandbox）。"""
    from fastapi.testclient import TestClient

    from quant.api.app import app

    with TestClient(app) as client:
        h = client.get("/api/health").json()
        assert "environment" in h and h["environment"] in ("local", "sandbox")


# ---------- 标的注册 / 删除 ----------

def test_timeframe_table_has_no_gaps():
    """TIMEFRAMES 只保留真实可下载的粒度：没有 2h/4h，且各 Provider 表与之对齐。"""
    from quant.core.models import TF_COVERAGE_HINT, TF_LABELS, TIMEFRAMES

    assert TIMEFRAMES == ["1m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"]
    assert "2h" not in TIMEFRAMES and "4h" not in TIMEFRAMES
    assert set(TF_LABELS) == set(TIMEFRAMES)
    assert set(TF_COVERAGE_HINT) == set(TIMEFRAMES)


def test_infer_symbol_type():
    from quant.data.base import infer_symbol_type
    # 加密货币已被移除：含 "/" 不再识别为 crypto/ccxt
    assert infer_symbol_type("BTC/USDT") == ("stock", "yfinance")
    assert infer_symbol_type("ETH-USDT") == ("stock", "yfinance")  # 无斜杠按股票
    # A股股票
    assert infer_symbol_type("600519") == ("stock", "baostock")  # 6 位纯数字 → A股
    assert infer_symbol_type("000001.SZ") == ("stock", "baostock")  # 深主板（平安银行）
    assert infer_symbol_type("300750.SZ") == ("stock", "baostock")  # 创业板
    assert infer_symbol_type("688981.SH") == ("stock", "baostock")  # 科创板
    # A股 ETF（51/56/58 沪、15/16 深）
    assert infer_symbol_type("510300.SH") == ("etf", "baostock")  # 沪深300ETF
    assert infer_symbol_type("510300") == ("etf", "baostock")
    assert infer_symbol_type("159915.SZ") == ("etf", "baostock")  # 创业板ETF
    assert infer_symbol_type("159915") == ("etf", "baostock")
    assert infer_symbol_type("588000.SH") == ("etf", "baostock")  # 科创50ETF
    assert infer_symbol_type("560010.SH") == ("etf", "baostock")
    assert infer_symbol_type("163417.SZ") == ("etf", "baostock")
    # A股指数（000xxx.SH / 无后缀 000xxx → 沪指；399xxx → 深指）
    assert infer_symbol_type("000001.SH") == ("index", "baostock")  # 上证指数
    assert infer_symbol_type("000300.SH") == ("index", "baostock")  # 沪深300
    assert infer_symbol_type("000905.SH") == ("index", "baostock")  # 中证500
    assert infer_symbol_type("000001") == ("index", "baostock")  # 无后缀 000 → 沪指
    assert infer_symbol_type("399001.SZ") == ("index", "baostock")  # 深证成指
    assert infer_symbol_type("399006.SZ") == ("index", "baostock")  # 创业板指
    assert infer_symbol_type("399001") == ("index", "baostock")
    # 美股
    assert infer_symbol_type("AAPL") == ("stock", "yfinance")
    assert infer_symbol_type("tsla") == ("stock", "yfinance")  # 大小写归一


def test_update_auto_registers_unknown_symbol(manager):
    """update 遇到未注册标的自动注册（按格式推断类型）。"""
    results = manager.update(symbols=["NEWSYM"], timeframes=["1d"])
    assert results[0].ok
    meta = manager.symbols_repo.get("NEWSYM")
    assert meta is not None and meta["asset_type"] == "stock"


def test_update_auto_registers_crypto_symbol(tmp_config):
    """加密货币代码不再被识别为 crypto：注册为 yfinance 股票并因无数据优雅失败。"""
    from quant.data.manager import DataManager
    from quant.data.storage import MarketStore
    from quant.storage.db import init_db
    from quant.storage.repositories import MarketDataRepo, SymbolsRepo

    init_db(tmp_config.effective_db_path)
    store = MarketStore(tmp_config.effective_market_dir)
    dm = DataManager(tmp_config, store, MarketDataRepo(tmp_config.effective_db_path),
                     SymbolsRepo(tmp_config.effective_db_path), providers={})
    dm.update(symbols=["BTC/USDT"], timeframes=["1d"])
    meta = dm.symbols_repo.get("BTC/USDT")
    assert meta is not None
    assert meta["asset_type"] == "stock" and meta["source"] == "yfinance"
    assert meta["asset_type"] != "crypto"


def test_delete_symbol_removes_data(manager):
    """删除标的：符号池 + parquet + 元信息一起清理。"""
    manager.update(symbols=["TEST"], timeframes=["1d"])
    assert manager.store.exists("TEST", "1d")
    assert manager.market_repo.get_meta("TEST", "1d") is not None
    removed = manager.delete_symbol("TEST")
    assert removed == 1
    assert manager.symbols_repo.get("TEST") is None
    assert not manager.store.exists("TEST", "1d")
    assert manager.market_repo.get_meta("TEST", "1d") is None


def test_full_reload_failure_keeps_old_data(tmp_config):
    """--full 拉取失败时不得删除已有数据（先拉取成功后再替换）。"""
    from quant.data.manager import DataManager
    from quant.data.storage import MarketStore
    from quant.storage.db import init_db
    from quant.storage.repositories import MarketDataRepo, SymbolsRepo

    init_db(tmp_config.effective_db_path)
    tmp_config.data.providers.stock = ["fake"]
    store = MarketStore(tmp_config.effective_market_dir)
    dm = DataManager(tmp_config, store, MarketDataRepo(tmp_config.effective_db_path),
                     SymbolsRepo(tmp_config.effective_db_path),
                     providers={"fake": FakeProvider(window_start="2024-01-01", periods=10)})
    dm.symbols_repo.upsert(symbol="TEST", name="T", asset_type="stock", source="fake")

    r1 = dm.update(symbols=["TEST"], timeframes=["1d"])[0]
    assert r1.ok and dm.store.exists("TEST", "1d")

    # 第二次 --full：数据源拉取失败 → 旧数据必须保留
    dm.providers["fake"] = FakeProvider(fail_fetch=True)
    r2 = dm.update(symbols=["TEST"], timeframes=["1d"], full_reload=True)[0]
    assert not r2.ok
    assert dm.store.exists("TEST", "1d")  # 旧数据未丢
    assert dm.market_repo.get_meta("TEST", "1d") is not None
    assert dm.store.load("TEST", "1d").shape[0] == 10


# ---------- API ----------

def _make_test_app(tmp_config, monkeypatch):
    """构建用临时配置 + FakeProvider 的 FastAPI 应用（幂等、不联网）。"""
    from fastapi.testclient import TestClient

    from quant.api import app as app_module
    from quant.data.manager import DataManager
    from quant.data.storage import MarketStore
    from quant.storage.db import init_db
    from quant.storage.repositories import MarketDataRepo, SymbolsRepo

    init_db(tmp_config.effective_db_path)
    tmp_config.data.providers.stock = ["fake"]
    store = MarketStore(tmp_config.effective_market_dir)
    dm = DataManager(tmp_config, store, MarketDataRepo(tmp_config.effective_db_path),
                     SymbolsRepo(tmp_config.effective_db_path),
                     providers={"fake": FakeProvider()})
    # lifespan 里会调用 DataManager.create(config) → 替换 create 方法返回我们的假管理器
    monkeypatch.setattr(app_module.DataManager, "create",
                        classmethod(lambda cls, config=None: dm))
    return app_module.app, dm


def test_api_health_and_symbols(tmp_config, monkeypatch):
    from fastapi.testclient import TestClient

    app, _ = _make_test_app(tmp_config, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        r2 = client.get("/api/data/symbols")
        assert r2.status_code == 200
        assert isinstance(r2.json()["symbols"], list)


def test_api_status_shape(tmp_config, monkeypatch):
    """status 返回：覆盖区间/最新 bar/文件时间/过期标记 + 数据源健康。"""
    from fastapi.testclient import TestClient

    app, _ = _make_test_app(tmp_config, monkeypatch)
    with TestClient(app) as client:
        # 先更新一个标的（seed_symbols 已把默认配置里的 AAPL 写入临时库）
        r = client.post("/api/data/update", json={"symbols": ["AAPL"], "timeframes": ["1d"]})
        assert r.status_code == 200
        body = client.get("/api/data/status").json()
        assert "health" in body and "symbols" in body
        aapl = next(s for s in body["symbols"] if s["symbol"] == "AAPL")
        assert aapl["rows"] == 10
        assert aapl["last_bar_time"] and aapl["updated_at"]
        assert "file_mtime" in aapl and "stale" in aapl and "stale_seconds" in aapl
        assert aapl["source"] == "fake"
