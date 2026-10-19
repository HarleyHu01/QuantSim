"""交易时段与分钟级模拟盘的实时语义测试。

覆盖三件事：
1. `market_hours.session_state` 对盘中/午休/收盘/周末/美股判定正确
2. 启动水位线不重放历史（这是之前 5m 模拟盘「一启动就冒出几个月前成交」的根因）
3. 非交易时段的门控，以及日线账户必须放行（cron 在收盘后触发）
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from quant.core.config import AppConfig
from quant.core.market_hours import market_of, session_state
from quant.data.manager import DataManager
from quant.data.storage import MarketStore
from quant.engine.paper import BAR_CRON, PaperTradingEngine
from quant.storage.db import init_db
from quant.storage.repositories import MarketDataRepo, PaperRepo, SymbolsRepo
from tests.conftest import FakeProvider
from tests.strategies import StepStrategy  # noqa: F401  确保注册

ZERO = dict(commission_rate=0.0, commission_min=0.0, commission_fixed=0.0,
            stamp_tax_rate=0.0, transfer_fee_rate=0.0, slippage_bps=0.0,
            t_plus_enabled=False, max_position_pct=1.0)


# ---------------- market_hours ----------------
@pytest.mark.parametrize("market,now,expect_open,reason", [
    # 2026-10-19 是周一
    ("cn", datetime(2026, 10, 19, 2, 0, tzinfo=timezone.utc), True, "交易中"),    # 10:00 CST
    ("cn", datetime(2026, 10, 19, 4, 0, tzinfo=timezone.utc), False, "午间休市"),  # 12:00 CST
    ("cn", datetime(2026, 10, 19, 5, 30, tzinfo=timezone.utc), True, "交易中"),    # 13:30 CST
    ("cn", datetime(2026, 10, 19, 8, 0, tzinfo=timezone.utc), False, "已收盘"),    # 16:00 CST
    ("cn", datetime(2026, 10, 19, 1, 0, tzinfo=timezone.utc), False, "未开盘"),    # 09:00 CST
    ("cn", datetime(2026, 10, 18, 2, 0, tzinfo=timezone.utc), False, "周末休市"),  # 周日
    # 美股（2026-10-19 周一，EDT=UTC-4）
    ("us", datetime(2026, 10, 19, 14, 0, tzinfo=timezone.utc), True, "交易中"),    # 10:00 EDT
    ("us", datetime(2026, 10, 19, 21, 0, tzinfo=timezone.utc), False, "已收盘"),   # 17:00 EDT
])
def test_session_state(market, now, expect_open, reason):
    ss = session_state(market, now=now)
    assert ss.is_open is expect_open
    assert ss.reason == reason
    if not expect_open:
        assert ss.next_open is not None
        assert ss.seconds_to_next_open is not None and ss.seconds_to_next_open > 0


def test_session_state_lunch_break_next_open_is_1300_cst():
    """午休时下一次开盘必须是当日 13:00（回归：曾错误返回已过去的 09:30）。"""
    ss = session_state("cn", now=datetime(2026, 10, 19, 4, 0, tzinfo=timezone.utc))
    assert ss.reason == "午间休市"
    assert ss.seconds_to_next_open == pytest.approx(3600.0, abs=1.0)


def test_market_of():
    assert market_of("baostock", "600519.SH") == "cn"
    assert market_of("akshare", "000001.SZ") == "cn"
    assert market_of("tencent", "600519.SH") == "cn"
    assert market_of("yfinance", "AAPL") == "us"
    assert market_of(None, "600519.SH") == "cn"
    assert market_of(None, "AAPL") == "us"


# ---------------- 水位线 / 门控 ----------------
@pytest.fixture
def paper_minute_env(tmp_path):
    """5m 环境：fake provider 提供 60 根 5m bar（覆盖多根历史）。"""
    cfg = AppConfig(
        data_dir=tmp_path, db_path=tmp_path / "quant.db", market_dir=tmp_path / "market",
        symbols=[{"symbol": "TEST", "name": "T", "asset_type": "stock", "source": "fake"}],
    )
    init_db(cfg.effective_db_path)
    cfg.data.providers.stock = ["fake"]
    store = MarketStore(cfg.effective_market_dir)
    dm = DataManager(cfg, store, MarketDataRepo(cfg.effective_db_path),
                     SymbolsRepo(cfg.effective_db_path),
                     providers={"fake": FakeProvider(periods=60, freq="5min", use_start=True,
                                              flat=True, anchor_end=True)})
    dm.symbols_repo.upsert(symbol="TEST", name="T", asset_type="stock", source="fake")
    dm.update(["TEST"], ["5m"])
    repo = PaperRepo(cfg.effective_db_path)
    eng = PaperTradingEngine(cfg, dm, AsyncIOScheduler(), repo, dm.symbols_repo)
    return cfg, dm, repo, eng


def test_start_watermark_does_not_replay_history(paper_minute_env):
    """启动水位线回退 1 根，首次 tick 只补算最后 1 根，不再重放全部 60 根历史。"""
    cfg, dm, repo, eng = paper_minute_env
    bars = dm.load_bars("TEST", "5m")
    assert len(bars) == 60

    aid = repo.create_account("分钟账户", 1_000_000, {})
    eng.start(aid, ["TEST"], "5m", ["test_step"],
              settings={**ZERO, "execution": "same_close"}, warmup_bars=1)
    st = eng._state[aid]
    # 水位线 = 倒数第 2 根 → 启动时只剩最后 1 根待处理
    assert st.processed_ts["TEST"] == bars.index[-2]

    processed: list = []
    orig = eng._process_bar

    def spy(state, symbol, row, history):
        processed.append(row.name)
        return orig(state, symbol, row, history)

    eng._process_bar = spy  # type: ignore[method-assign]
    eng._tick(aid, "TEST")
    # 不该重放 60 根历史：首次 tick 最多处理
    #   水位线回退的 1 根 + 增量拉取的重叠窗口（OVERLAP_BARS=2）+ 期间新产生的 bar。
    # 断言「数量级正确」而不是锁死具体值，否则会随运行时刻漂移。
    assert len(processed) <= 5, f"重放了历史: {len(processed)} 根"
    assert all(ts >= bars.index[-2] for ts in processed)


def test_start_watermark_zero_waits_for_new_bar(paper_minute_env):
    """warmup_bars=0：水位线 = 最后一根，首次 tick 不补算任何历史 bar。"""
    cfg, dm, repo, eng = paper_minute_env
    bars = dm.load_bars("TEST", "5m")
    aid = repo.create_account("分钟账户2", 1_000_000, {})
    eng.start(aid, ["TEST"], "5m", ["test_step"], warmup_bars=0)
    st = eng._state[aid]
    assert st.processed_ts["TEST"] == bars.index[-1]

    processed: list = []
    orig = eng._process_bar

    def spy(state, symbol, row, history):
        processed.append(row.name)
        return orig(state, symbol, row, history)

    eng._process_bar = spy  # type: ignore[method-assign]
    eng._tick(aid, "TEST")
    # 唯一可能被处理的是 tick 期间新拉到的 bar，绝不会是历史那 60 根
    assert all(ts >= bars.index[-1] for ts in processed)


def test_recover_prefers_saved_watermark(paper_minute_env):
    """recover 时用持久化水位线续跑：不重放历史，也不丢停机期间的新 bar。"""
    cfg, dm, repo, eng = paper_minute_env
    bars = dm.load_bars("TEST", "5m")
    aid = repo.create_account("续跑账户", 1_000_000, {})
    eng.start(aid, ["TEST"], "5m", ["test_step"], warmup_bars=0)
    eng._tick(aid, "TEST")
    saved = eng._state[aid].processed_ts["TEST"]
    assert saved >= bars.index[-1]

    # 停掉旧引擎但保持账户 running，模拟进程重启
    eng._unregister_jobs(aid)
    eng._state.pop(aid, None)
    repo.set_status(aid, "running")

    eng2 = PaperTradingEngine(cfg, dm, AsyncIOScheduler(), repo, dm.symbols_repo)
    eng2.recover()
    restored = eng2._state[aid]
    assert restored.processed_ts["TEST"] == saved
    eng2._tick(aid, "TEST")
    # 只在其基础上前进，绝不回退到历史
    assert eng2._state[aid].processed_ts["TEST"] >= saved


def test_minute_account_gated_when_market_closed(paper_minute_env, monkeypatch):
    """非交易时段：分钟级 tick 必须被跳过，不拉数据。"""
    cfg, dm, repo, eng = paper_minute_env
    aid = repo.create_account("门控账户", 1_000_000, {})
    eng.start(aid, ["TEST"], "5m", ["test_step"])
    st = eng._state[aid]

    called = {"n": 0}

    def fake_update(*a, **kw):
        called["n"] += 1
        return []

    monkeypatch.setattr(dm, "update", fake_update)
    # 2026-10-18 是周日
    import quant.engine.paper as paper_mod
    monkeypatch.setattr(paper_mod, "session_state",
                        lambda m, now=None: session_state(m, now=datetime(
                            2026, 10, 18, 2, 0, tzinfo=timezone.utc)))
    eng._tick(aid, "TEST")
    assert called["n"] == 0
    assert "周末休市" in st.last_tick_skip["TEST"]


def test_daily_account_not_gated_after_close(paper_minute_env, monkeypatch):
    """日线账户在 15:30 触发时处于「已收盘」，必须放行，否则永远拿不到当日日线。"""
    cfg, dm, repo, eng = paper_minute_env
    aid = repo.create_account("日线账户", 1_000_000, {})
    eng.start(aid, ["TEST"], "1d", ["test_step"])
    st = eng._state[aid]

    import quant.engine.paper as paper_mod
    monkeypatch.setattr(paper_mod, "session_state",
                        lambda m, now=None: session_state(m, now=datetime(
                            2026, 10, 19, 8, 0, tzinfo=timezone.utc)))  # 16:00 CST 已收盘
    assert eng._gate(st, "TEST", datetime(2026, 10, 19, 8, 0, tzinfo=timezone.utc)) is None


def test_minute_jobs_use_bar_aligned_cron(paper_minute_env):
    """分钟级任务必须是 cron 对齐 K 线边界，而不是会漂移的 interval。"""
    cfg, dm, repo, eng = paper_minute_env
    aid = repo.create_account("调度账户", 1_000_000, {})
    eng.start(aid, ["TEST"], "5m", ["test_step"])
    jobs = {j.id: j for j in eng.scheduler.get_jobs()}
    job = jobs[f"paper_{aid}_TEST_5m"]
    assert "cron" in str(job.trigger)
    assert f"minute='{BAR_CRON['5m'][0]}'" in str(job.trigger)
    assert f"second='{BAR_CRON['5m'][1]}'" in str(job.trigger)
    eng.stop(aid)


def test_realtime_status_shape(paper_minute_env):
    """status.realtime 暴露市场时段与数据新鲜度，供 UI 展示延迟。"""
    cfg, dm, repo, eng = paper_minute_env
    aid = repo.create_account("观测账户", 1_000_000, {})
    eng.start(aid, ["TEST"], "5m", ["test_step"])
    eng._tick(aid, "TEST")
    rt = eng.status(aid)["realtime"]
    assert rt["is_minute"] is True
    assert rt["timeframe"] == "5m"
    assert "TEST" in rt["symbols"]
    sym = rt["symbols"]["TEST"]
    for key in ("market", "market_open", "session", "last_tick_at", "bar_lag_seconds"):
        assert key in sym
    eng.stop(aid)
