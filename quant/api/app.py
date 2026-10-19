"""FastAPI 应用入口（API 优先，独立于前端 UI）。"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quant.api.routers import backtest as backtest_router
from quant.api.routers import data as data_router
from quant.api.routers import fusion as fusion_router
from quant.api.routers import llm as llm_router
from quant.api.routers import ml as ml_router
from quant.api.routers import paper as paper_router
from quant.api.routers import portfolio as portfolio_router
from quant.api.routers import settings as settings_router
from quant.api.routers import strategies as strategies_router
from quant.api.routers import system as system_router
from quant.api.routers import workbuddy as workbuddy_router
from quant.api.state import app_state
from quant.core.config import AppConfig
from quant.core.utils import detect_environment
from quant.data.manager import DataManager
from quant.engine.paper import PaperTradingEngine
from quant.storage.db import init_db
from quant.storage.repositories import (BacktestRepo, LLMRepo, PaperRepo, PortfolioRepo,
                                       SettingsRepo, StrategiesRepo, SymbolsRepo)
from quant.strategies.registry import ensure_loaded, list_all

log = logging.getLogger(__name__)


def seed_symbols(config: AppConfig, dm: DataManager) -> None:
    """把配置里的默认标的写入符号池（幂等 upsert）。"""
    repo = dm.symbols_repo
    for s in config.symbols:
        repo.upsert(
            symbol=s["symbol"],
            name=s.get("name", ""),
            asset_type=s.get("asset_type", "stock"),
            source=s.get("source", "yfinance"),
            exchange=s.get("exchange", ""),
            lot_size=s.get("lot_size", 1),
            price_limit_pct=s.get("price_limit_pct"),
            t_plus=s.get("t_plus", "T+1"),
        )


def seed_strategies(repo: StrategiesRepo) -> None:
    """把注册表里的内置策略写入 strategies 表（幂等，默认权重 100、启用）。

    跳过 test_ 前缀（测试专用策略，避免污染生产库）。
    """
    ensure_loaded()
    for meta in list_all():
        if meta["key"].startswith("test_"):
            continue
        repo.upsert(key=meta["key"], name=meta["name"], description=meta["description"],
                    strategy_type="builtin", class_name=meta["key"],
                    config_json={}, enabled=1, weight=100.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = AppConfig.load()
    init_db(config.effective_db_path)
    dm = DataManager.create(config)
    seed_symbols(config, dm)
    dm.warm_up()  # 后台预热 baostock 登录会话（login 需 11~13s，不预热会让健康检查误报超时）

    scheduler = AsyncIOScheduler()
    scheduler.start()

    paper_repo = PaperRepo(config.effective_db_path)
    symbols_repo = SymbolsRepo(config.effective_db_path)
    strategies_repo = StrategiesRepo(config.effective_db_path)
    portfolio_repo = PortfolioRepo(config.effective_db_path)
    paper_engine = PaperTradingEngine(config, dm, scheduler, paper_repo, symbols_repo)
    paper_engine.recover()  # 恢复上次 running 的模拟盘账户
    seed_strategies(strategies_repo)

    app_state["config"] = config
    app_state["data_manager"] = dm
    app_state["scheduler"] = scheduler
    app_state["paper_engine"] = paper_engine
    app_state["paper_repo"] = paper_repo
    app_state["strategies_repo"] = strategies_repo
    app_state["portfolio_repo"] = portfolio_repo
    app_state["settings_repo"] = SettingsRepo(config.effective_db_path)
    app_state["backtest_repo"] = BacktestRepo(config.effective_db_path)
    app_state["llm_repo"] = LLMRepo(config.effective_db_path)
    log.info("QuantSim API 就绪: db=%s market=%s", config.effective_db_path, config.effective_market_dir)
    yield
    scheduler.shutdown(wait=False)
    app_state.clear()


app = FastAPI(title="QuantSim", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(data_router.router, prefix="/api")
app.include_router(backtest_router.router, prefix="/api")
app.include_router(paper_router.router, prefix="/api")
app.include_router(strategies_router.router, prefix="/api")
app.include_router(portfolio_router.router, prefix="/api")
app.include_router(settings_router.router, prefix="/api")
app.include_router(system_router.router, prefix="/api")
app.include_router(llm_router.router, prefix="/api")
app.include_router(ml_router.router, prefix="/api")
app.include_router(fusion_router.router, prefix="/api")
app.include_router(workbuddy_router.router, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "app": "QuantSim", "version": "1.0",
            "environment": detect_environment()}
