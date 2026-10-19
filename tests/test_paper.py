"""模拟盘测试：fake feed 3 bar → 挂单→成交→落库→recover 全链路。"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from quant.core.config import AppConfig
from quant.core.models import Order
from quant.data.manager import DataManager
from quant.data.storage import MarketStore
from quant.engine.paper import PaperTradingEngine
from quant.storage.db import init_db
from quant.storage.repositories import MarketDataRepo, PaperRepo, SymbolsRepo
from tests.conftest import FakeProvider
from tests.strategies import HalfStepStrategy, StepStrategy  # noqa: F401  确保注册


@pytest.fixture
def paper_env(tmp_path):
    cfg = AppConfig(
        data_dir=tmp_path, db_path=tmp_path / "quant.db", market_dir=tmp_path / "market",
        symbols=[{"symbol": "TEST", "name": "T", "asset_type": "stock", "source": "fake"}],
    )
    init_db(cfg.effective_db_path)
    cfg.data.providers.stock = ["fake"]
    store = MarketStore(cfg.effective_market_dir)
    dm = DataManager(cfg, store, MarketDataRepo(cfg.effective_db_path),
                     SymbolsRepo(cfg.effective_db_path),
                     providers={"fake": FakeProvider(periods=3, freq="D", use_start=True, flat=True)})
    dm.symbols_repo.upsert(symbol="TEST", name="T", asset_type="stock", source="fake")
    dm.update(["TEST"], ["1d"])  # 预载 3 根 bar
    repo = PaperRepo(cfg.effective_db_path)
    scheduler = AsyncIOScheduler()
    eng = PaperTradingEngine(cfg, dm, scheduler, repo, dm.symbols_repo)
    return cfg, dm, repo, scheduler, eng


ZERO = dict(commission_rate=0.0, commission_min=0.0, commission_fixed=0.0,
            stamp_tax_rate=0.0, transfer_fee_rate=0.0, slippage_bps=0.0,
            t_plus_enabled=False, max_position_pct=1.0)


def test_paper_full_chain(paper_env):
    """fake feed 3 bar：挂单→成交→持仓/成交/净值落库。"""
    cfg, dm, repo, scheduler, eng = paper_env
    aid = repo.create_account("测试账户", 1_000_000, {})
    st = eng.start(aid, ["TEST"], "1d", ["test_step"], warmup_bars=None,
                   settings={**ZERO, "execution": "next_open"})
    assert st["status"] == "running"

    eng._tick(aid, "TEST")  # fake 3 根 bar

    trades = repo.list_trades(aid)
    assert len(trades) >= 1
    assert trades[0]["side"] == "buy"
    assert trades[0]["price"] == pytest.approx(10.5)  # bar1 开盘价（平价），无滑点

    positions = repo.list_positions(aid)
    assert positions and positions[0]["qty"] > 0

    assert len(repo.list_equity(aid)) >= 1
    assert repo.list_pending(aid) == []  # 挂单已成交清空

    st2 = eng.status(aid)
    assert st2["equity"] > 0 and st2["pending_orders"] == 0
    eng.stop(aid)
    assert repo.get_account(aid)["status"] == "stopped"


def test_paper_pending_persisted_until_next_bar(paper_env):
    """bar0 产生挂单并落库；bar1 成交后清空。"""
    cfg, dm, repo, scheduler, eng = paper_env
    aid = repo.create_account("测试2", 1_000_000, {})
    eng.start(aid, ["TEST"], "1d", ["test_step"], settings={**ZERO, "execution": "next_open"},
              warmup_bars=None)

    bars = dm.load_bars("TEST", "1d")
    # 只处理 bar0 → 挂单应已持久化
    eng._process_bar(eng._state[aid], "TEST", bars.iloc[0], bars.loc[:bars.index[0]])
    assert len(repo.list_pending(aid)) == 1

    # bar1 → 挂单成交并清空
    eng._process_bar(eng._state[aid], "TEST", bars.iloc[1], bars.loc[:bars.index[1]])
    assert repo.list_pending(aid) == []
    assert len(repo.list_trades(aid)) == 1
    eng.stop(aid)


def test_paper_recover_restores_and_discards_pending(paper_env):
    """recover：重建 Broker（持仓恢复）、丢弃 PENDING、重注册任务。"""
    cfg, dm, repo, scheduler, eng = paper_env
    aid = repo.create_account("测试3", 1_000_000, {})
    eng.start(aid, ["TEST"], "1d", ["test_step"], settings={**ZERO, "execution": "next_open"},
              warmup_bars=None)
    eng._tick(aid, "TEST")

    # 模拟进程重启前残留一条挂单（recover 应丢弃）
    repo.add_pending(aid, Order(datetime(2024, 1, 2, tzinfo=timezone.utc),
                                "TEST", "buy", 100, 10.0, reason="残留"))

    eng2 = PaperTradingEngine(cfg, dm, AsyncIOScheduler(), repo, dm.symbols_repo)
    eng2.recover()
    assert aid in eng2._state
    restored = eng2._state[aid]
    assert restored.brokers[0].position("TEST") is not None
    assert restored.brokers[0].position("TEST").qty > 0  # 从 paper_positions 恢复
    assert repo.list_pending(aid) == []  # PENDING 已丢弃，不重新撮合
    assert restored.brokers[0].cash > 0  # 现金从净值快照恢复
    jobs = [j.id for j in eng2.scheduler.get_jobs()]
    assert any(j.startswith(f"paper_{aid}_") for j in jobs)
    eng2.stop(aid)


def test_paper_signal_merge_two_strategies(paper_env):
    """signal_merge：多策略合成一次下单（不重复交易）。"""
    cfg, dm, repo, scheduler, eng = paper_env
    aid = repo.create_account("测试4", 1_000_000, {})
    eng.start(aid, ["TEST"], "1d", ["test_step", "test_step"],
              settings={**ZERO, "execution": "same_close"})
    eng._tick(aid, "TEST")
    trades = repo.list_trades(aid)
    # 两策略权重各 100 → 合成目标仍为 1 → 每次 bar 最多 1 笔订单
    buys = [t for t in trades if t["side"] == "buy"]
    assert len(buys) <= 3
    eng.stop(aid)


def test_paper_delete_account(paper_env):
    """stopped 账户可删除；cleanup 清理持仓/成交/净值记录。"""
    from fastapi import HTTPException

    from quant.api.routers import paper as paper_router
    from quant.api.state import app_state

    cfg, dm, repo, scheduler, eng = paper_env
    app_state["paper_engine"] = eng
    app_state["paper_repo"] = repo
    try:
        aid = repo.create_account("待删除", 1_000_000, {})
        r = paper_router.paper_delete_account(aid, cleanup=True)
        assert r["ok"] and repo.get_account(aid) is None

        # 运行中 → 409
        aid2 = repo.create_account("运行中", 1_000_000, {})
        repo.set_status(aid2, "running")
        eng._state[aid2] = object()  # 模拟内存态运行中
        try:
            paper_router.paper_delete_account(aid2, cleanup=True)
            assert False, "运行中账户删除应抛 409"
        except HTTPException as e:
            assert e.status_code == 409
        assert repo.get_account(aid2) is not None  # 未被删除
    finally:
        app_state.clear()


def test_paper_delete_cleanup_flag(paper_env):
    """cleanup=False：仅删账户行，保留历史记录（供审计）。"""
    from quant.api.routers import paper as paper_router
    from quant.api.state import app_state

    cfg, dm, repo, scheduler, eng = paper_env
    app_state["paper_engine"] = eng
    app_state["paper_repo"] = repo
    try:
        aid = repo.create_account("审计保留", 1_000_000, {})
        paper_router.paper_delete_account(aid, cleanup=False)
        assert repo.get_account(aid) is None
    finally:
        app_state.clear()


def test_system_shutdown_stops_paper_accounts(paper_env):
    """POST /api/system/shutdown：停止所有运行中模拟盘账户并落库。"""
    from quant.api.routers import system as system_router
    from quant.api.state import app_state

    cfg, dm, repo, scheduler, eng = paper_env
    app_state["paper_engine"] = eng
    app_state["shutdown_real"] = False  # 测试模式：不真正退出进程
    try:
        a1 = repo.create_account("关1", 1_000_000, {})
        a2 = repo.create_account("关2", 1_000_000, {})
        eng.start(a1, ["TEST"], "1d", ["test_step"])
        eng.start(a2, ["TEST"], "1d", ["test_step"])
        assert a1 in eng._state and a2 in eng._state
        r = system_router.shutdown()
        assert r["ok"] is True
        assert set(r["stopped_accounts"]) == {a1, a2}
        assert repo.get_account(a1)["status"] == "stopped"
        assert repo.get_account(a2)["status"] == "stopped"
        assert eng._state == {}
    finally:
        app_state.clear()


def test_system_shutdown_without_engine():
    """无运行引擎时 shutdown 仍正常返回。"""
    from quant.api.routers import system as system_router
    from quant.api.state import app_state

    app_state.clear()
    app_state["shutdown_real"] = False
    try:
        r = system_router.shutdown()
        assert r["ok"] is True and r["stopped_accounts"] == []
    finally:
        app_state.clear()


def test_paper_capital_split(paper_env):
    """capital_split：两策略独立子账户（同标的允许），独立持仓/成交落库，净值合并。"""
    cfg, dm, repo, scheduler, eng = paper_env
    aid = repo.create_account("split", 1_000_000, {})
    eng.start(aid, ["TEST"], "1d", ["test_step", "test_half"],
              settings={**ZERO, "execution": "same_close"},
              portfolio={"mode": "capital_split",
                         "weights": {"test_step": 60, "test_half": 40}})
    st = eng._state[aid]
    assert st.capital_split
    assert len(st.brokers) == 2
    assert st.brokers[0].cash == pytest.approx(600_000)  # 60% 资金池
    assert st.brokers[1].cash == pytest.approx(400_000)  # 40% 资金池

    eng._tick(aid, "TEST")  # fake 3 根 bar

    positions = repo.list_positions(aid)
    sub_keys = {p["sub_key"] for p in positions}
    assert sub_keys == {"test_step", "test_half"}  # 两个子账户各自持仓（同标的）
    trades = repo.list_trades(aid)
    assert {t["sub_key"] for t in trades} == {"test_step", "test_half"}
    assert {t["strategy_key"] for t in trades} == {"test_step", "test_half"}

    st2 = eng.status(aid)
    assert st2["mode"] == "capital_split"
    assert st2["equity"] == pytest.approx(1_000_000, rel=1e-3)  # 平价数据 → 无损益
    assert len(st2["sub_brokers"]) == 2
    eng.stop(aid)
    assert repo.get_account(aid)["status"] == "stopped"
