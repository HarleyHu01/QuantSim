"""设置 API：交易设置 / LLM 配置持久化。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from quant.api.state import app_state
from quant.core.models import TradingSettings

router = APIRouter(prefix="/settings", tags=["settings"])


def _repo():
    repo = app_state.get("settings_repo")
    if repo is None:
        raise HTTPException(status_code=503, detail="设置仓储未就绪")
    return repo


class TradingSettingsIn(BaseModel):
    """与 TradingSettings 同构（全部有默认值，UI 表单逐字段提交）。"""
    initial_capital: float | None = None
    commission_rate: float | None = None
    commission_fixed: float | None = None
    commission_min: float | None = None
    stamp_tax_rate: float | None = None
    transfer_fee_rate: float | None = None
    slippage_bps: float | None = None
    margin_rate: float | None = None
    margin_call_threshold: float | None = None
    timeframe: str | None = None
    min_trade_interval_bars: int | None = None
    cooldown_bars: int | None = None
    rebalance_freq: str | None = None
    signal_threshold: float | None = None
    execution: str | None = None
    max_position_pct: float | None = None
    max_exposure_pct: float | None = None
    allow_short: bool | None = None
    t_plus_enabled: bool | None = None
    max_drawdown_halt: float | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    daily_loss_limit_pct: float | None = None
    max_positions: int | None = None
    benchmark_symbol: str | None = None
    periods_per_year: int | None = None


class LLMSettingsIn(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    temperature: float | None = None


def _merged(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in updates.items():
        if v is not None:
            out[k] = v
    return out


@router.get("/trading")
def get_trading_settings() -> dict:
    """返回完整交易设置（默认值 + 已保存覆盖）。"""
    saved = _repo().get_trading() or {}
    return TradingSettings(**saved).model_dump()


@router.put("/trading")
def put_trading_settings(body: TradingSettingsIn) -> dict:
    saved = _repo().get_trading() or {}
    merged = _merged(saved, body.model_dump(exclude_none=True))
    # 校验（非法字段抛 pydantic 错误）
    settings = TradingSettings(**merged)
    _repo().put_trading(settings.model_dump())
    return settings.model_dump()


@router.get("/llm")
def get_llm_settings() -> dict:
    from quant.core.config import LLMConfig
    saved = _repo().get_llm() or {}
    return LLMConfig(**saved).model_dump()


@router.put("/llm")
def put_llm_settings(body: LLMSettingsIn) -> dict:
    from quant.core.config import LLMConfig
    saved = _repo().get_llm() or {}
    merged = _merged(saved, body.model_dump(exclude_none=True))
    cfg = LLMConfig(**merged)
    _repo().put_llm(cfg.model_dump())
    return cfg.model_dump()
