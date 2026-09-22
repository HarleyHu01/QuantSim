"""回测 API：单策略回测 / 组合回测 / 结果查询。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import pandas as pd

from quant.api.state import app_state
from quant.core.models import PortfolioConfig, TradingSettings
from quant.engine.backtest import EventDrivenBacktestEngine, VectorizedBacktestEngine
from quant.portfolio.portfolio import StrategyPortfolio
from quant.strategies.registry import create, ensure_loaded

router = APIRouter(prefix="/backtest", tags=["backtest"])

# 分钟级周期：基准缺失时不用分钟级基准（改用日线近似，避免免费分钟数据拉取卡住回测）
_MINUTE_TFS = {"5m", "15m", "30m", "60m", "1h", "2h", "4h"}


class BacktestSingleIn(BaseModel):
    symbol: str
    timeframe: str = "1d"
    strategy_id: str
    params: dict | None = None
    settings: dict | None = None
    engine: str = "event_driven"  # event_driven | vectorized
    start: str | None = None
    end: str | None = None


class BacktestComboIn(BaseModel):
    symbols: list[str]
    timeframe: str = "1d"
    strategy_ids: list[str]
    params: dict | None = None
    settings: dict | None = None
    portfolio: dict | None = None
    engine: str = "event_driven"
    start: str | None = None
    end: str | None = None


def _dm():
    dm = app_state.get("data_manager")
    if dm is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    return dm


def _build_engine(engine: str, settings: TradingSettings, port_cfg: PortfolioConfig):
    st = StrategyPortfolio(port_cfg)
    if engine == "vectorized":
        return VectorizedBacktestEngine(settings, st)
    return EventDrivenBacktestEngine(settings, st)


def _symbol_meta(dm, symbols: list[str]) -> dict[str, dict]:
    out = {}
    for s in symbols:
        m = dm.symbols_repo.get(s)
        out[s] = {k: m[k] for k in ("price_limit_pct", "t_plus", "lot_size") if m and m.get(k) is not None}
    return out


def _load_benchmark(dm, settings: TradingSettings) -> pd.DataFrame | None:
    """加载基准标的行情；本地缺失时自动拉取（仅用于对比，不作为交易标的）。

    分钟级回测（5m/15m/...）基准缺失时**不拉分钟级基准**（分钟数据量大、免费源易失败/限流，
    可能拖垮回测请求），改用**日线基准近似**（引擎内 reindex+ffill 对齐）；日线也缺失则
    触发日线自动拉取；再失败返回 None → 引擎回退用交易标的本体（买入持有）作为基准。
    """
    sym = settings.benchmark_symbol
    if not sym or not sym.strip():
        return None
    tf = settings.timeframe
    df = dm.load_bars(sym, tf)
    if not df.empty:
        return df
    if tf in _MINUTE_TFS:
        daily = dm.load_bars(sym, "1d")
        if not daily.empty:
            return daily
        try:
            dm.update([sym], ["1d"], incremental=True)
            daily = dm.load_bars(sym, "1d")
        except Exception:  # noqa: BLE001
            daily = pd.DataFrame()
        return daily if not daily.empty else None
    try:
        dm.update([sym], [tf], incremental=True)
        df = dm.load_bars(sym, tf)
    except Exception:  # noqa: BLE001
        df = pd.DataFrame()
    return df if not df.empty else None


def _saved_settings() -> TradingSettings:
    """使用已保存的 /api/settings/trading 配置（UI 交易设置页）。"""
    repo = app_state.get("settings_repo")
    if repo:
        saved = repo.get_trading()
        if saved:
            return TradingSettings(**saved)
    return TradingSettings()


def _persist(repo, engine, symbols, timeframe, strategy_ids, settings, port_cfg, result):
    run_id = repo.create_run(
        engine=result.engine, symbols=symbols, timeframe=timeframe,
        strategy_ids=strategy_ids,
        portfolio_cfg=port_cfg.model_dump(),
        settings=settings.model_dump(),
    )
    try:
        repo.add_equity(run_id, result.equity)
        repo.add_trades(run_id, result.trades)
        repo.finish_run(run_id, result.metrics)
        result.run_id = run_id
    except Exception as e:  # noqa: BLE001
        repo.fail_run(run_id, str(e))
        raise HTTPException(status_code=500, detail=f"结果落库失败: {e}")
    return run_id


def _summary(result) -> dict:
    combined_tail = {}
    for sym, df in (result.combined_signal or {}).items():
        tail = df.tail(30).reset_index()
        tail["datetime"] = tail["ts"].dt.strftime("%Y-%m-%d %H:%M")
        combined_tail[sym] = tail[["datetime", "target_weight", "reason", "confidence"]].to_dict("records")
    return {
        "run_id": result.run_id,
        "engine": result.engine,
        "symbols": result.symbols,
        "timeframe": result.timeframe,
        "start": str(result.start),
        "end": str(result.end),
        "metrics": result.metrics,
        "per_strategy_metrics": result.per_strategy_metrics,
        "trade_count": len(result.trades),
        "trades_head": [t.to_dict() for t in result.trades[:5]],
        "combined_signal_tail": combined_tail,
        "warnings": result.warnings[:20],
    }


@router.post("/single")
def backtest_single(body: BacktestSingleIn) -> dict:
    ensure_loaded()
    dm = _dm()
    bars = dm.load_bars(body.symbol, body.timeframe, body.start, body.end)
    if bars.empty:
        raise HTTPException(status_code=404, detail=f"无数据: {body.symbol} {body.timeframe}")
    settings = TradingSettings(**body.settings) if body.settings else _saved_settings()
    settings.timeframe = body.timeframe
    port_cfg = PortfolioConfig(mode="signal_merge", weights={body.strategy_id: 100.0})
    engine = _build_engine(body.engine, settings, port_cfg)
    strategy = create(body.strategy_id, body.params)
    benchmark = _load_benchmark(dm, settings)
    result = engine.run({body.symbol: bars}, [strategy], _symbol_meta(dm, [body.symbol]),
                        benchmark_bars=benchmark)
    repo = app_state["backtest_repo"]
    _persist(repo, body.engine, [body.symbol], body.timeframe, [body.strategy_id],
             settings, port_cfg, result)
    return _summary(result)


@router.post("/combo")
def backtest_combo(body: BacktestComboIn) -> dict:
    ensure_loaded()
    dm = _dm()
    symbol_bars = {}
    for s in body.symbols:
        bars = dm.load_bars(s, body.timeframe, body.start, body.end)
        if bars.empty:
            raise HTTPException(status_code=404, detail=f"无数据: {s} {body.timeframe}")
        symbol_bars[s] = bars
    settings = TradingSettings(**body.settings) if body.settings else _saved_settings()
    settings.timeframe = body.timeframe
    # 组合配置：请求体优先，否则用已保存的 /api/portfolio
    port_cfg_data = body.portfolio
    if port_cfg_data is None:
        saved = (app_state.get("portfolio_repo").get() if app_state.get("portfolio_repo") else None)
        if saved:
            import json
            port_cfg_data = {
                "mode": saved["mode"],
                "weights": json.loads(saved.get("weights_json") or "{}"),
                "rebalance_freq": saved["rebalance_freq"],
                "signal_threshold": saved["signal_threshold"],
            }
    port_cfg = PortfolioConfig(**(port_cfg_data or {}))
    engine = _build_engine(body.engine, settings, port_cfg)
    strategies = [create(k, (body.params or {}).get(k)) for k in body.strategy_ids]
    benchmark = _load_benchmark(dm, settings)
    result = engine.run(symbol_bars, strategies, _symbol_meta(dm, body.symbols),
                        benchmark_bars=benchmark)
    repo = app_state["backtest_repo"]
    _persist(repo, body.engine, body.symbols, body.timeframe, body.strategy_ids,
             settings, port_cfg, result)
    return _summary(result)


@router.get("/runs")
def list_runs(limit: int = 50) -> dict:
    return {"runs": app_state["backtest_repo"].list_runs(limit)}


@router.get("/runs/{run_id}")
def get_run(run_id: int) -> dict:
    run = app_state["backtest_repo"].get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="回测不存在")
    import json
    run["metrics_json"] = json.loads(run.get("metrics_json") or "{}")
    return run


@router.get("/runs/{run_id}/equity")
def get_run_equity(run_id: int, limit: int = 10000) -> dict:
    return {"equity": app_state["backtest_repo"].get_equity(run_id, limit)}


@router.get("/runs/{run_id}/trades")
def get_run_trades(run_id: int, limit: int = 1000) -> dict:
    return {"trades": app_state["backtest_repo"].get_trades(run_id, limit)}
