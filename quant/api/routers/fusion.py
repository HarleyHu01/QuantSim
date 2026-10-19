"""模型 × 因子融合配置 API。

融合策略（``alpha.models.strategies.fusion``）把「大模型集成预测」和
「因子策略」缝成一个仓位决策。这一层只负责**配置的存取与试算**，
真正的信号合成在策略内部完成 —— 这样回测、组合、模拟盘走的是同一条
代码路径，不会出现「UI 里算一套、回测里算另一套」的漂移。

路由：
- GET  /fusion/config      读取融合配置（模型权重 / 因子权重 / 融合方式 / 门控）
- PUT  /fusion/config      合并保存融合配置
- POST /fusion/preview     用真实行情试算一次融合决策（不写库、不下单）

## 为什么配置存 kv_store 而不是 strategies 表

融合配置是「全局偏好」而不是「某个策略实例的属性」：用户在配置页调好
权重后，回测页、模拟盘页、策略管理页看到的应该是同一套。存 kv_store
（``ml.fusion`` 键）天然满足这一点，也不需要给 strategies 表加列。
策略参数里仍然可以覆盖 —— 回测时想临时试另一套权重，改参数即可，
不必动全局配置。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from quant.api.state import app_state
from quant.core.config import AppConfig

log = logging.getLogger(__name__)

router = APIRouter(prefix="/fusion", tags=["fusion"])

#: 融合配置在 kv_store 里的键（与策略内部保持一致）
CONFIG_KEY = "ml.fusion"

#: 默认配置：只开 2 个统计基线 + 2 个因子，毫秒级；Kronos 默认关（慢）
DEFAULT_CONFIG: dict[str, Any] = {
    "model_weights": {
        "kronos": 0.0,
        "random_walk": 0.0,
        "ewma": 1.0,
        "momentum": 0.8,
        "drift": 0.0,
        "reversion": 0.0,
        "chronos": 0.0,
        "timesfm": 0.0,
        "moirai": 0.0,
    },
    "factor_weights": {
        "dual_ma": 1.0,
        "macd": 0.6,
        "rsi": 0.0,
        "bollinger": 0.0,
        "momentum": 0.0,
        "mean_reversion": 0.0,
        "turtle": 0.0,
        "grid": 0.0,
    },
    "fusion_mode": "blend",
    "model_share": 0.5,
    "factor_share": 0.5,
    "confidence_weighted": True,
    "min_abs_return": 0.0,
    "min_return_sigma": 0.3,
    "min_agreement": 0.6,
    "max_dispersion": 2.0,
    "min_models": 2,
    "position": 1.0,
    "allow_short": False,
    "pred_len": 12,
    "context": 240,
}

#: 融合方式 → 中文说明（与策略内部一致，UI 直接渲染）
FUSION_MODES: dict[str, str] = {
    "blend": "加权混合：两侧同时影响仓位，按份额加权",
    "model_gate": "模型闸门：模型不开门（空仓/门控未过）时因子一律作废",
    "factor_gate": "因子闸门：因子空仓时模型说了也不算",
    "vote": "投票制：两侧按份额投票，谁的合计权重占优听谁的",
}


def _kv():
    """通用键值仓储。缺失时按需构造（与 ml.py 同一套懒加载模式）。"""
    repo = app_state.get("kv_repo")
    if repo is None:
        from quant.storage.repositories import KVRepo

        repo = KVRepo(AppConfig.load().effective_db_path)
        app_state["kv_repo"] = repo
    return repo


def _load_config() -> dict[str, Any]:
    """读融合配置：kv_store 值覆盖默认值，未知键忽略。"""
    saved = _kv().get(CONFIG_KEY)
    if not isinstance(saved, dict):
        return dict(DEFAULT_CONFIG)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({k: v for k, v in saved.items() if k in DEFAULT_CONFIG})
    return cfg


def _clean_weights(raw: Any, allowed: set[str] | None = None) -> dict[str, float]:
    """权重清洗：只留认识的 key、只留非负有限数值。

    负权重会把「加权」变成「反向加权」，稳定性很差且难以解释，
    所以这里直接截断到 0 —— 想反向用某个模型，正确做法是把它关掉。
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        key = str(k)
        if allowed is not None and key not in allowed:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        fv = float(v)
        if not (fv == fv) or fv in (float("inf"), float("-inf")):  # NaN / inf
            continue
        out[key] = max(0.0, fv)
    return out


def _model_keys() -> set[str]:
    from alpha.models.ensemble import MODEL_FACTORIES

    return set(MODEL_FACTORIES)


def _factor_keys() -> set[str]:
    """全部已注册的因子策略 key（不含模型化策略，避免「模型自己和自己融合」）。"""
    from quant.strategies import registry

    registry.ensure_loaded()
    from alpha.factors.strategies import __all__ as factor_all

    return {str(k) for k in factor_all}


def _validate(cfg: dict[str, Any]) -> list[str]:
    """配置自检，返回人读的错误列表（空 = 通过）。"""
    errors: list[str] = []
    mw = cfg.get("model_weights") or {}
    fw = cfg.get("factor_weights") or {}
    if not any(v > 0 for v in mw.values()):
        errors.append("至少启用一个大模型（权重 > 0）")
    if not any(v > 0 for v in fw.values()):
        errors.append("至少启用一个因子策略（权重 > 0）")
    if cfg.get("fusion_mode") not in FUSION_MODES:
        errors.append(f"未知融合方式：{cfg.get('fusion_mode')}，可选 {sorted(FUSION_MODES)}")
    share = float(cfg.get("model_share", 0.0))
    if not 0.0 <= share <= 1.0:
        errors.append("模型侧话语权必须在 0~1 之间")
    return errors


class FusionConfigIn(BaseModel):
    """融合配置（缺省字段沿用当前保存值，与 ml.py 的合并语义一致）。"""

    model_weights: dict[str, float] | None = None
    factor_weights: dict[str, float] | None = None
    fusion_mode: str | None = None
    model_share: float | None = Field(None, ge=0.0, le=1.0)
    confidence_weighted: bool | None = None
    min_abs_return: float | None = Field(None, ge=0.0, le=0.5)
    min_return_sigma: float | None = Field(None, ge=0.0, le=10.0)
    min_agreement: float | None = Field(None, ge=0.0, le=1.0)
    max_dispersion: float | None = Field(None, ge=0.0, le=20.0)
    min_models: int | None = Field(None, ge=1, le=9)
    position: float | None = Field(None, ge=0.0, le=1.0)
    allow_short: bool | None = None
    pred_len: int | None = Field(None, ge=1, le=120)
    context: int | None = Field(None, ge=32, le=2048)


class FusionPreviewIn(BaseModel):
    """融合试算请求。``config`` 可临时覆盖，不写库（UI 调参数即时预览用）。"""

    symbol: str
    timeframe: str = "5m"
    bars: int = Field(400, ge=64, le=4000, description="取多少根历史 K 线")
    config: dict[str, Any] | None = None


@router.get("/config")
def fusion_config() -> dict:
    """读取融合配置（含可选清单，UI 一次拿全）。"""
    cfg = _load_config()
    return {
        "config": cfg,
        "models": _model_keys(),
        "factor_keys": sorted(_factor_keys()),
        "fusion_modes": [{"key": k, "label": v} for k, v in FUSION_MODES.items()],
        "errors": _validate(cfg),
    }


@router.put("/config")
def fusion_config_put(body: FusionConfigIn) -> dict:
    """合并保存融合配置。只覆盖显式传入的字段。"""
    cfg = _load_config()
    patch = body.model_dump(exclude_none=True)

    if "model_weights" in patch:
        cfg["model_weights"] = _clean_weights(patch["model_weights"], _model_keys())
    if "factor_weights" in patch:
        cfg["factor_weights"] = _clean_weights(patch["factor_weights"], _factor_keys())
    if "fusion_mode" in patch:
        cfg["fusion_mode"] = str(patch["fusion_mode"])
    if "model_share" in patch:
        cfg["model_share"] = float(patch["model_share"])
        cfg["factor_share"] = float(1.0 - patch["model_share"])
    for k in ("confidence_weighted", "min_abs_return", "min_return_sigma",
              "min_agreement", "max_dispersion", "min_models", "position",
              "allow_short", "pred_len", "context"):
        if k in patch:
            cfg[k] = patch[k]

    errors = _validate(cfg)
    if errors:
        raise HTTPException(status_code=400, detail="；".join(errors))

    _kv().put(CONFIG_KEY, cfg)
    return {"ok": True, "config": cfg}


@router.post("/preview")
def fusion_preview(body: FusionPreviewIn) -> dict:
    """用真实行情试算一次融合决策，返回两侧信号与最终仓位。

    与回测的差别只在**范围**：这里只算最后一根，回测算整段。
    合成逻辑走的是策略内部同一个 ``_decide``，所以预览结果和回测
    最后一根的决策必然一致 —— 这是「UI 不骗人」的底线。
    """
    dm = app_state.get("data_manager")
    if dm is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    df = dm.load_bars(body.symbol, body.timeframe)
    if df is None or df.empty:
        raise HTTPException(status_code=404,
                            detail=f"无数据：{body.symbol} {body.timeframe}，请先下载")

    cfg = _load_config()
    if body.config:
        patch = {k: v for k, v in body.config.items() if k in DEFAULT_CONFIG}
        if "model_weights" in patch:
            patch["model_weights"] = _clean_weights(patch["model_weights"], _model_keys())
        if "factor_weights" in patch:
            patch["factor_weights"] = _clean_weights(patch["factor_weights"], _factor_keys())
        if "model_share" in patch:
            patch["factor_share"] = float(1.0 - float(patch["model_share"]))
        cfg.update(patch)

    errors = _validate(cfg)
    if errors:
        raise HTTPException(status_code=400, detail="；".join(errors))

    n = min(int(body.bars), len(df))
    window = df.iloc[-n:]
    if len(window) < 64:
        raise HTTPException(status_code=400,
                            detail=f"历史不足：仅 {len(window)} 根，至少需要 64 根")

    from alpha.models.strategies.fusion import FusionStrategy
    from quant.strategies.base import StrategyContext

    params = {
        "model_weights_json": json.dumps(cfg["model_weights"], ensure_ascii=False),
        "factor_weights_json": json.dumps(cfg["factor_weights"], ensure_ascii=False),
        "fusion_mode": cfg["fusion_mode"],
        "model_share": cfg["model_share"],
        "confidence_weighted": cfg["confidence_weighted"],
        "min_abs_return": cfg["min_abs_return"],
        "min_return_sigma": cfg["min_return_sigma"],
        "min_agreement": cfg["min_agreement"],
        "max_dispersion": cfg["max_dispersion"],
        "min_models": cfg["min_models"],
        "position": cfg["position"],
        "allow_short": cfg["allow_short"],
        "pred_len": cfg["pred_len"],
        "context": min(cfg["context"], len(window)),
    }
    strat = FusionStrategy(params)
    ctx = StrategyContext(symbol=body.symbol, timeframe=body.timeframe, history=window)

    t0 = datetime.now()
    try:
        weight, reason, confidence = strat._decide(window, ctx)
    except Exception as e:  # noqa: BLE001
        log.exception("融合试算失败 %s %s", body.symbol, body.timeframe)
        raise HTTPException(status_code=500,
                            detail=f"融合试算失败：{type(e).__name__}: {e}") from e
    elapsed = (datetime.now() - t0).total_seconds()

    model_w, model_reason = strat._model_signal(window)
    factor_w, factor_reason = strat._factor_signal(window, ctx)

    return {
        "ok": True,
        "symbol": body.symbol,
        "timeframe": body.timeframe,
        "context_bars": len(window),
        "last_bar": str(window.index[-1]),
        "last_close": float(window["close"].iloc[-1]),
        "elapsed_seconds": round(elapsed, 2),
        "target_weight": float(weight),
        "confidence": float(confidence),
        "reason": reason,
        "model_signal": float(model_w),
        "model_reason": model_reason,
        "factor_signal": float(factor_w),
        "factor_reason": factor_reason,
        "config": cfg,
    }
