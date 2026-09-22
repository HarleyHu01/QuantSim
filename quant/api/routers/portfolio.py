"""组合配置 API：模式/权重/阈值/再平衡频率持久化（第 5 步）。"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from quant.api.state import app_state
from quant.core.models import PortfolioConfig

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


class PortfolioIn(BaseModel):
    mode: str = "signal_merge"  # signal_merge | capital_split | equal
    weights: dict[str, float] = {}  # strategy_key → weight(-100..100)
    rebalance_freq: str = "none"
    signal_threshold: float = 0.0


def _repo():
    repo = app_state.get("portfolio_repo")
    if repo is None:
        raise HTTPException(status_code=503, detail="组合仓储未就绪")
    return repo


def _to_dict(row: dict) -> dict:
    return {
        "mode": row["mode"],
        "weights": json.loads(row.get("weights_json") or "{}"),
        "rebalance_freq": row["rebalance_freq"],
        "signal_threshold": row["signal_threshold"],
        "updated_at": row.get("updated_at"),
    }


@router.get("")
def get_portfolio() -> dict:
    row = _repo().get()
    if row is None:
        return {"mode": "signal_merge", "weights": {}, "rebalance_freq": "none",
                "signal_threshold": 0.0, "updated_at": None}
    return _to_dict(row)


@router.put("")
def put_portfolio(body: PortfolioIn) -> dict:
    # 校验（非法 mode/权重类型会抛 pydantic 校验错误）
    PortfolioConfig(mode=body.mode, weights=body.weights,
                    rebalance_freq=body.rebalance_freq, signal_threshold=body.signal_threshold)
    _repo().put(body.mode, body.weights, body.rebalance_freq, body.signal_threshold)
    return get_portfolio()
