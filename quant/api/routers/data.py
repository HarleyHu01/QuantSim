"""数据管理相关 API：标的池 / 一键更新 / 状态 / 行情。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from quant.api.state import app_state
from quant.core.utils import parse_iso

router = APIRouter(prefix="/data", tags=["data"])

# 各周期"数据是否最新"的容忍滞后（秒）。1d 给 5 天容忍周末；1w/1M 相应放宽。
BAR_TOLERANCE_SECONDS = {
    "1m": 3 * 60, "5m": 3 * 300, "15m": 3 * 900, "30m": 3 * 1800,
    "1h": 3 * 3600, "2h": 3 * 7200, "4h": 3 * 14400,
    "1d": 5 * 86400, "1w": 14 * 86400, "1M": 45 * 86400,
}


class SymbolIn(BaseModel):
    symbol: str
    name: str = ""
    asset_type: str | None = None  # None → 按 symbol 格式推断
    source: str | None = None  # None → 按推断
    exchange: str = ""
    lot_size: int = 1
    price_limit_pct: float | None = None
    t_plus: str = "T+1"
    enabled: bool = True


class UpdateIn(BaseModel):
    symbols: list[str] | None = None  # None = 全部启用标的
    timeframes: list[str] | None = None  # None = 配置默认周期
    incremental: bool | None = None  # None = 配置默认
    adjust: str | None = None  # qfq | hfq | none
    start_date: str | None = None  # "YYYY-MM-DD"，缺省 = 自动（增量起点/配置 lookback）
    end_date: str | None = None  # "YYYY-MM-DD"，缺省 = 今天


def _dm():
    dm = app_state.get("data_manager")
    if dm is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    return dm


@router.get("/symbols")
def list_symbols() -> dict:
    return {"symbols": _dm().symbols_repo.list_all()}


@router.post("/symbols")
def add_symbol(body: SymbolIn) -> dict:
    """添加标的到符号池（asset_type/source 缺省时按格式推断）。"""
    from quant.data.base import infer_symbol_type
    symbol = body.symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol 不能为空")
    asset_type = body.asset_type or infer_symbol_type(symbol)[0]
    source = body.source or infer_symbol_type(symbol)[1]
    price_limit_pct = body.price_limit_pct
    if price_limit_pct is None and asset_type == "stock":
        price_limit_pct = 0.10  # A股主板默认涨跌停 10%
    _dm().symbols_repo.upsert(
        symbol=symbol, name=body.name or symbol, asset_type=asset_type,
        source=source, exchange=body.exchange, lot_size=body.lot_size,
        price_limit_pct=price_limit_pct, t_plus=body.t_plus, enabled=int(body.enabled),
    )
    return {"ok": True, "symbol": symbol, "asset_type": asset_type, "source": source}


@router.delete("/symbols/{symbol:path}")
def delete_symbol(symbol: str) -> dict:
    """删除标的：移除符号池记录 + 全部已存行情（parquet 与元信息）。

    {symbol:path} 转换器支持带斜杠的加密交易对（如 BTC/USDT）。
    """
    removed = _dm().delete_symbol(symbol)
    return {"ok": True, "symbol": symbol, "data_files_removed": removed}


@router.get("/status")
def data_status(refresh: bool = False) -> dict:
    """每个 symbol×timeframe 的数据状态 + 数据源健康（refresh=True 强制绕过缓存重测）。"""
    dm = _dm()
    now = datetime.now(timezone.utc)
    items: list[dict] = []
    for m in dm.market_repo.list_meta():
        item = dict(m)
        last_bar = parse_iso(m.get("last_bar_time"))
        if last_bar is not None:
            if last_bar.tzinfo is None:
                last_bar = last_bar.replace(tzinfo=timezone.utc)
            lag = (now - last_bar).total_seconds()
            tol = BAR_TOLERANCE_SECONDS.get(m["timeframe"], 3 * 86400)
            item["stale"] = bool(lag > tol)  # True = 数据可能不是最新
            item["stale_seconds"] = int(lag)
        else:
            item["stale"] = None
            item["stale_seconds"] = None
        item["file_mtime"] = _file_mtime(m.get("file_path"))
        items.append(item)
    return {
        "symbols": items,
        "health": {k: v.to_dict() for k, v in dm.health(force=refresh).items()},
    }


def _file_mtime(file_path: str | None) -> str | None:
    if not file_path:
        return None
    try:
        return datetime.fromtimestamp(Path(file_path).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return None


@router.post("/update")
def update_data(body: UpdateIn) -> dict:
    """一键更新数据：返回每个 (symbol, timeframe) 的更新明细（含实际覆盖区间）。"""
    dm = _dm()
    results = dm.update(
        symbols=body.symbols,
        timeframes=body.timeframes,
        incremental=body.incremental,
        adjust=body.adjust,
        start_date=body.start_date,
        end_date=body.end_date,
    )
    return {"results": [r.to_dict() for r in results]}


@router.get("/bars/{symbol}")
def get_bars(symbol: str, timeframe: str = "1d", start: str | None = None, end: str | None = None) -> dict:
    df = _dm().load_bars(symbol, timeframe, start, end)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": len(df),
        "data": df.reset_index().rename(columns={"index": "datetime"}).to_dict(orient="records"),
    }
