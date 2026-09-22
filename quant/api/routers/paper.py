"""模拟盘 API：start / stop / status / 账户查询。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from quant.api.state import app_state

router = APIRouter(prefix="/paper", tags=["paper"])


class PaperStartIn(BaseModel):
    account_id: int | None = None  # None = 新建
    name: str = "模拟盘"
    symbols: list[str]
    timeframe: str = "1d"
    strategy_ids: list[str]
    params: dict | None = None
    settings: dict | None = None
    portfolio: dict | None = None
    poll_seconds: int | None = None
    initial_capital: float | None = None


class PaperStopIn(BaseModel):
    account_id: int


class PaperStatusIn(BaseModel):
    account_id: int | None = None


def _engine():
    eng = app_state.get("paper_engine")
    if eng is None:
        raise HTTPException(status_code=503, detail="模拟盘引擎未就绪")
    return eng


def _repo():
    return app_state["paper_repo"]


@router.post("/start")
def paper_start(body: PaperStartIn) -> dict:
    eng = _engine()
    repo = _repo()
    if body.account_id is None:
        settings = body.settings or {}
        capital = body.initial_capital or settings.get("initial_capital", 1_000_000.0)
        account_id = repo.create_account(body.name, float(capital), settings)
    else:
        account_id = body.account_id
    return eng.start(
        account_id, body.symbols, body.timeframe, body.strategy_ids,
        params=body.params, settings=body.settings, portfolio=body.portfolio,
        poll_seconds=body.poll_seconds, name=body.name,
    )


@router.post("/stop")
def paper_stop(body: PaperStopIn) -> dict:
    return _engine().stop(body.account_id)


@router.delete("/accounts/{account_id}")
def paper_delete_account(account_id: int, cleanup: bool = True) -> dict:
    """删除模拟盘账户。

    - 仅 status=stopped 的账户可删除（运行中 → 409）
    - cleanup=True（默认）同时清理持仓/成交/净值/挂单记录；False 仅删账户行
    """
    eng = _engine()
    repo = _repo()
    account = repo.get_account(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="账户不存在")
    if account["status"] == "running":
        raise HTTPException(status_code=409, detail="账户运行中，请先停止再删除")
    if account_id in eng._state:  # 防御：内存态残留
        eng._state.pop(account_id, None)
    for job in eng.scheduler.get_jobs():
        if job.id.startswith(f"paper_{account_id}_"):
            job.remove()
    repo.delete_account(account_id, cleanup=cleanup)
    return {"ok": True, "account_id": account_id, "cleanup": cleanup}


@router.get("/status")
def paper_status(account_id: int | None = None) -> dict:
    eng = _engine()
    repo = _repo()
    if account_id is not None:
        return eng.status(account_id)
    return {"accounts": repo.list_accounts()}


@router.get("/account")
def paper_account(account_id: int) -> dict:
    return {"account": _repo().get_account(account_id)}


@router.get("/positions")
def paper_positions(account_id: int) -> dict:
    return {"positions": _repo().list_positions(account_id)}


@router.get("/trades")
def paper_trades(account_id: int, limit: int = 200) -> dict:
    return {"trades": _repo().list_trades(account_id, limit)}


@router.get("/equity")
def paper_equity(account_id: int, limit: int = 10000) -> dict:
    return {"equity": _repo().list_equity(account_id, limit)}
