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


def test_config_proxy_and_exchange_defaults():
    """默认配置：akshare 禁用代理、baostock 为 A股主源、ccxt 交易所链按易访问性排序。"""
    from quant.core.config import AppConfig

    cfg = AppConfig()
    assert cfg.data.akshare_use_proxy is False
    assert cfg.data.providers.stock[0] == "baostock"  # A股主源
    assert "akshare" in cfg.data.providers.stock  # akshare 备用
    assert cfg.data.providers.etf[0] == "baostock"  # ETF 也 baostock 优先
    assert "index" in cfg.data.providers.model_dump()  # 指数资产类型
    assert cfg.data.providers.future[0] == "akshare"  # 期货仅 akshare
    assert cfg.data.ccxt_exchanges[0] in ("okx", "bitget", "gateio", "binance")
    assert len(cfg.data.ccxt_exchanges) >= 4
    assert cfg.data.ccxt_timeout_ms > 0


def test_akshare_disable_proxy_env(monkeypatch):
    """use_proxy=False 时把国内域名写入 NO_PROXY（直连不走系统代理）。"""
    import os

    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    from quant.data.akshare_provider import AkshareProvider

    AkshareProvider(use_proxy=False)
    merged = os.environ.get("no_proxy", "") + os.environ.get("NO_PROXY", "")
    assert "eastmoney.com" in merged
    assert "sina.com.cn" in merged


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
    """三通道的原始响应解析成统一 OHLCV（离线）。"""
    import requests as _requests

    from quant.data.akshare_provider import (AkshareProvider, _channel_eastmoney,
                                             _channel_tencent)
    from datetime import datetime, timedelta

    now = datetime.now()

    def fake_get(url, params, referer, timeout, throttle=True):
        class R:
            def json(self):
                if "push2his" in url:
                    return {"data": {"klines": [
                        "2024-01-02,7.89,7.74,7.91,7.74,1158366,1075742,5.31,1.2,0.09,1.32"]}}
                if "gtimg" in url:
                    return {"data": {"sz000001": {"qfqday": [
                        ["2024-01-02", "7.89", "7.74", "7.91", "7.74", "1158366.0"]]}}}
                raise AssertionError(url)
        return R()

    monkeypatch.setattr("quant.data.akshare_provider._http_get", fake_get)
    df = _channel_eastmoney("000001", "sz", "0.000001", now, now, "1d", "qfq")
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert df.iloc[0]["volume"] == 1158366 * 100  # 手→股
    assert df.iloc[0]["close"] == 7.74

    df2 = _channel_tencent("000001", "sz", "0.000001", now, now, "1d", "qfq")
    assert df2.iloc[0]["volume"] == 1158366 * 100
    assert df2.iloc[0]["high"] == 7.91


def test_akshare_fallback_switches_channel(monkeypatch):
    """东财失败自动切腾讯/雪球；重试与退避生效；记录实际通道。"""
    from quant.data.akshare_provider import AkshareProvider
    from datetime import datetime

    provider = AkshareProvider(use_proxy=False)
    calls = {"em": 0, "tx": 0, "xq": 0}
    slept = []

    def fake_em(*a, **k):
        calls["em"] += 1
        raise ConnectionError("RemoteDisconnected")

    def fake_tx(*a, **k):
        calls["tx"] += 1
        import pandas as pd
        return pd.DataFrame({"open": [1.0], "close": [1.0], "high": [1.0],
                             "low": [1.0], "volume": [100.0]},
                            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC")]))

    def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr("quant.data.akshare_provider._channel_eastmoney", fake_em)
    monkeypatch.setattr("quant.data.akshare_provider._channel_tencent", fake_tx)
    monkeypatch.setattr("quant.data.akshare_provider.time.sleep", fake_sleep)
    monkeypatch.setattr("quant.data.akshare_provider._consecutive_failures", 0, raising=False)

    df = provider.fetch_history("000001", datetime(2024, 1, 1), datetime(2024, 1, 10),
                                "1d", "qfq")
    assert calls["em"] == 2 and calls["tx"] == 1  # 东财 2 次后切腾讯
    assert provider.last_source == "腾讯"
    assert df is not None and not df.empty
    assert slept  # 退避等待发生过


def test_akshare_health_source(monkeypatch):
    """健康检查记录实际可用通道并标注在 ProviderHealth.source。"""
    from quant.data.akshare_provider import AkshareProvider

    provider = AkshareProvider(use_proxy=False)

    def fake_tencent(*a, **k):
        import pandas as pd
        return pd.DataFrame({"open": [1.0], "close": [1.0], "high": [1.0],
                             "low": [1.0], "volume": [100.0]},
                            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC")]))

    monkeypatch.setattr("quant.data.akshare_provider._channel_eastmoney",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("x")))
    monkeypatch.setattr("quant.data.akshare_provider._channel_tencent", fake_tencent)
    h = provider.health_check()
    assert h.healthy is True
    assert h.source == "腾讯"


def test_akshare_real_fetch_000001():
    """真实拉取 000001（平安银行）日线：至少一个通道成功（东财/腾讯/新浪）。"""
    from datetime import datetime, timedelta

    from quant.data.akshare_provider import AkshareProvider

    provider = AkshareProvider(use_proxy=False)
    end = datetime.now()
    start = end - timedelta(days=45)
    df = provider.fetch_history("000001", start, end, "1d", "qfq")
    assert df is not None and len(df) >= 10
    assert provider.last_source in ("东方财富", "腾讯", "新浪/雪球")
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert df["close"].iloc[-1] > 0


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
    """离线拉取：login→query→logout 全链路 + qfq → adjustflag=2 + UTC 归一化。

    日线查询字段**不含 time**（baostock 日线传 time 会报 10004012）；
    无后缀 000001 归为上证指数（sh.000001），故用 000001.SZ（平安银行股票）验证股票路径。
    """
    from quant.data import baostock_provider as bp

    fake = _FakeBs()
    monkeypatch.setattr(bp, "bs", fake)
    p = bp.BaostockProvider()
    df = p.fetch_history("000001.SZ", datetime(2024, 1, 1), datetime(2024, 1, 10),
                         "1d", "qfq")
    assert fake.login_calls == 1 and fake.logout_calls == 1
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
    # 不支持周期
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
    fake_fail = _FakeBs(login_error="10001002")
    monkeypatch.setattr(bp, "bs", fake_fail)
    with pytest.raises(RuntimeError, match="登录失败"):
        p.fetch_history("000001", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d", "none")


@pytest.mark.skipif(importlib.util.find_spec("baostock") is None,
                    reason="baostock 未安装，跳过真实拉取测试")
def test_baostock_real_fetch_000001():
    """真实拉取 000001（平安银行）日线：验证 baostock 连通性与复权数据。"""
    from datetime import datetime, timedelta

    from quant.data.baostock_provider import BaostockProvider

    provider = BaostockProvider()
    end = datetime.now()
    start = end - timedelta(days=45)
    df = provider.fetch_history("000001", start, end, "1d", "qfq")
    assert df is not None and len(df) >= 10
    assert provider.last_source == "baostock"
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert df["close"].iloc[-1] > 0


def test_ccxt_default_exchanges():
    from quant.data.ccxt_provider import CcxtProvider

    p = CcxtProvider()
    assert len(p.exchange_ids) >= 4
    assert p.exchange_ids[0] in ("okx", "bitget", "gateio", "binance")


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

    assert set(TF_MAP) >= {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M"}
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


def test_ccxt_tf_map_and_construct():
    from quant.data.ccxt_provider import CcxtProvider, TF_MAP

    assert TF_MAP["1M"] == "1M" and TF_MAP["1h"] == "1h" and TF_MAP["1w"] == "1w"
    # 构造不访问网络
    p = CcxtProvider(exchange_ids=["kraken"])
    assert p.exchange_ids == ["kraken"]


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

def test_infer_symbol_type():
    from quant.data.base import infer_symbol_type
    assert infer_symbol_type("BTC/USDT") == ("crypto", "ccxt")
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
    """加密标的自动注册为 crypto/ccxt（数据源不可用时优雅失败但已注册）。"""
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
    assert meta is not None and meta["asset_type"] == "crypto" and meta["source"] == "ccxt"


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
