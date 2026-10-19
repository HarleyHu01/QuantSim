"""机器学习模型 API（Kronos K 线预测 + 多模型集成）。

设计原则：
- **不阻塞主流程**。模型加载要 4~50 秒，所以状态查询（`/status`）绝不触发加载，
  只有显式 `/load` 或 `/predict` 才会加载；加载在进程内缓存，后续复用。
- **依赖可选**。缺 torch 时所有接口返回 200 + `available: false` + 安装命令，
  UI 据此显示提示，而不是抛 500。

路由：
- GET  /ml/kronos/status            Kronos 可用性/加载状态
- POST /ml/kronos/load              显式加载 Kronos
- POST /ml/kronos/predict           单模型试预测
- GET  /ml/models                   全部预测模型的可用性（Kronos/统计基线/外部模型）
- GET  /ml/ensemble/config          读取集成配置（权重 + 硬性门控）
- PUT  /ml/ensemble/config          保存集成配置
- POST /ml/ensemble/predict         集成试预测（含每个模型的票与门控明细）
- POST /ml/ensemble/auto-weights    按历史表现给建议权重
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from quant.api.state import app_state
from quant.core.config import AppConfig
from alpha.models import KronosForecaster, KronosUnavailable
from alpha.models import kronos as kr
from alpha.models.ensemble import (CONFIG_KEY, EnsembleForecaster, EnsembleSpec,
                               auto_weights, list_models)
from alpha.models.kronos import MAX_CONTEXT

log = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml"])


def _dm():
    dm = app_state.get("data_manager")
    if dm is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    return dm


def _kv():
    """通用键值仓储（存集成配置）。缺失时按需构造，避免各入口都要注册。"""
    repo = app_state.get("kv_repo")
    if repo is None:
        from quant.storage.repositories import KVRepo

        repo = KVRepo(AppConfig.load().effective_db_path)
        app_state["kv_repo"] = repo
    return repo


def _load_spec() -> EnsembleSpec:
    return EnsembleSpec.from_dict(_kv().get(CONFIG_KEY))


class KronosPredictIn(BaseModel):
    """试预测请求。"""

    symbol: str
    timeframe: str = "5m"
    bars: int = Field(400, ge=32, le=4000, description="取多少根历史 K 线作为上下文")
    pred_len: int = Field(12, ge=1, le=120, description="预测未来多少根")
    model: str | None = None
    sample_count: int = Field(1, ge=1, le=8)


@router.get("/kronos/status")
def kronos_status() -> dict:
    """Kronos 可用性与加载状态（不触发加载）。"""
    fc = KronosForecaster.default()
    return fc.status()


@router.post("/kronos/load")
def kronos_load() -> dict:
    """显式加载模型（首次含权重下载，约 4~50 秒）。加载后进程内复用。"""
    if not kr.available():
        return {
            "ok": False,
            "available": False,
            "missing": kr.missing_deps(),
            "hint": kr.install_hint(),
            "message": f"依赖未安装：{', '.join(kr.missing_deps())}",
        }
    fc = KronosForecaster.default()
    try:
        # status 不加载，这里用一次极小预测触发加载（32 根是允许的最小上下文）
        import numpy as np
        import pandas as pd

        idx = pd.date_range(end=pd.Timestamp.now(tz=fc.spec.tz), periods=32, freq="5min")
        base = np.linspace(100, 101, 32)
        bars = pd.DataFrame(
            {"open": base, "high": base * 1.001, "low": base * 0.999, "close": base, "volume": 1.0},
            index=idx,
        )
        fc.forecast(bars, pred_len=1)
    except KronosUnavailable as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"模型加载失败：{type(e).__name__}: {e}") from e
    return {"ok": True, **fc.status()}


@router.post("/kronos/predict")
def kronos_predict(body: KronosPredictIn) -> dict:
    """用真实行情做一次预测（UI「试预测」用）。不写库、不下单。"""
    if not kr.available():
        raise HTTPException(
            status_code=400,
            detail=f"Kronos 依赖未安装：{', '.join(kr.missing_deps())}。安装命令：{kr.install_hint()}",
        )
    dm = _dm()
    df = dm.load_bars(body.symbol, body.timeframe)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"无数据：{body.symbol} {body.timeframe}，请先下载")

    n = min(int(body.bars), len(df))
    window = df.iloc[-n:]
    if len(window) < 32:
        raise HTTPException(status_code=400, detail=f"历史不足：仅 {len(window)} 根，至少需要 32 根")

    fc = KronosForecaster.default()
    if body.model:
        from alpha.models.kronos import KronosSpec

        spec = KronosSpec(**{**fc.spec.__dict__, "model": body.model,
                             "max_context": min(MAX_CONTEXT.get(body.model, 512), n),
                             "pred_len": body.pred_len,
                             "sample_count": body.sample_count})
        fc = KronosForecaster(spec)
    elif body.sample_count != fc.spec.sample_count:
        from alpha.models.kronos import KronosSpec

        spec = KronosSpec(**{**fc.spec.__dict__, "sample_count": body.sample_count,
                             "pred_len": body.pred_len, "max_context": min(fc.spec.max_context, n)})
        fc = KronosForecaster(spec)

    t0 = datetime.now()
    try:
        pred = fc.forecast(window, pred_len=body.pred_len)
    except KronosUnavailable as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        log.exception("Kronos 预测失败 %s %s", body.symbol, body.timeframe)
        raise HTTPException(status_code=500, detail=f"预测失败：{type(e).__name__}: {e}") from e
    elapsed = (datetime.now() - t0).total_seconds()

    last_close = float(window["close"].iloc[-1])
    pred_close = float(pred["close"].iloc[-1])
    exp_ret = pred_close / last_close - 1.0 if last_close > 0 else 0.0
    return {
        "ok": True,
        "symbol": body.symbol,
        "timeframe": body.timeframe,
        "context_bars": len(window),
        "last_bar": str(window.index[-1]),
        "last_close": last_close,
        "elapsed_seconds": round(elapsed, 2),
        "expected_return": exp_ret,
        "model": fc.spec.resolved_model(),
        "device": fc.status().get("device"),
        "history": [
            {"t": str(ts), "close": float(c), "kind": "history"}
            for ts, c in window["close"].iloc[-min(len(window), 120):].items()
        ],
        "forecast": [
            {
                "t": str(ts),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
                "kind": "forecast",
            }
            for ts, r in pred.iterrows()
        ],
    }


# ====================== 多模型集成 ======================

class EnsembleConfigIn(BaseModel):
    """集成配置（权重 + 硬性门控）。缺省字段沿用当前保存值。"""

    weights: dict[str, float] | None = None
    min_abs_return: float | None = Field(None, ge=0.0, le=0.5)
    min_return_sigma: float | None = Field(None, ge=0.0, le=10.0)
    min_agreement: float | None = Field(None, ge=0.0, le=1.0)
    max_dispersion: float | None = Field(None, ge=0.0, le=20.0)
    min_models: int | None = Field(None, ge=1, le=9)
    position: float | None = Field(None, ge=0.0, le=1.0)
    allow_short: bool | None = None


class EnsemblePredictIn(BaseModel):
    """集成试预测请求。"""

    symbol: str
    timeframe: str = "5m"
    bars: int = Field(400, ge=32, le=4000)
    pred_len: int | None = Field(None, ge=1, le=120)
    #: 临时覆盖权重，不写库（UI 调参数时即时预览用）
    weights: dict[str, float] | None = None


class AutoWeightsIn(BaseModel):
    """按历史表现估算建议权重。"""

    symbol: str
    timeframe: str = "5m"
    pred_len: int = Field(12, ge=1, le=120)
    eval_bars: int = Field(240, ge=60, le=2000, description="回看区间长度（根）")
    step: int = Field(20, ge=1, le=200, description="每隔多少根取一个评估点")
    context: int = Field(240, ge=32, le=2048)
    keys: list[str] | None = Field(None, description="只评估这些模型（None = 默认几个便宜且稳定的）")


@router.get("/models")
def models() -> dict:
    """全部预测模型及其可用性（不触发任何权重加载）。"""
    items = list_models()
    return {
        "models": items,
        "available": [m["key"] for m in items if m.get("available")],
        "missing": [m["key"] for m in items if not m.get("available")],
    }


@router.get("/ensemble/config")
def ensemble_config() -> dict:
    """读取集成配置（权重 + 硬性门控）。"""
    spec = _load_spec()
    return {"config": spec.as_dict(), "models": list_models()}


@router.put("/ensemble/config")
def ensemble_config_put(body: EnsembleConfigIn) -> dict:
    """合并保存集成配置。只覆盖显式传入的字段。"""
    cur = _load_spec()
    data = cur.as_dict()
    patch = body.model_dump(exclude_none=True)
    if "weights" in patch:
        # 只接受数值权重；负权重会把「投票」变成「反向投票」，稳定性很差
        data["weights"] = {str(k): max(0.0, float(v))
                           for k, v in patch["weights"].items()
                           if isinstance(v, (int, float))}
    for k in ("min_abs_return", "min_return_sigma", "min_agreement",
              "max_dispersion", "position", "min_models", "allow_short"):
        if k in patch:
            data[k] = patch[k]
    spec = EnsembleSpec.from_dict(data)
    _kv().put(CONFIG_KEY, spec.as_dict())
    return {"ok": True, "config": spec.as_dict()}


@router.post("/ensemble/predict")
def ensemble_predict(body: EnsemblePredictIn) -> dict:
    """用真实行情做一次集成预测，返回每个模型的票与门控明细。

    与单模型的区别：即使最终空仓，也会把「为什么空仓」逐条列出来
    （哪个门控没过、实测值多少），这比只给一个仓位数字有用得多。
    """
    dm = _dm()
    df = dm.load_bars(body.symbol, body.timeframe)
    if df is None or df.empty:
        raise HTTPException(status_code=404,
                            detail=f"无数据：{body.symbol} {body.timeframe}，请先下载")

    spec = _load_spec()
    if body.weights is not None:
        spec.weights = {str(k): max(0.0, float(v)) for k, v in body.weights.items()
                        if isinstance(v, (int, float))}
    n = min(int(body.bars), len(df))
    window = df.iloc[-n:]
    min_needed = spec.context + spec.pred_len + 2
    if len(window) < min_needed:
        # 历史不够就自动缩短上下文，而不是直接报错——集成里的统计基线
        # 只需要几十根，Kronos 也支持 32 根起。
        spec.context = max(32, len(window) - spec.pred_len - 2)
        if len(window) < 40:
            raise HTTPException(status_code=400,
                                detail=f"历史不足：仅 {len(window)} 根，至少需要 40 根")

    t0 = datetime.now()
    try:
        result = EnsembleForecaster(spec).predict(window, pred_len=body.pred_len)
    except Exception as e:  # noqa: BLE001
        log.exception("集成预测失败 %s %s", body.symbol, body.timeframe)
        raise HTTPException(status_code=500,
                            detail=f"集成预测失败：{type(e).__name__}: {e}") from e
    elapsed = (datetime.now() - t0).total_seconds()

    payload = result.as_dict()
    payload.update({
        "ok": True,
        "symbol": body.symbol,
        "timeframe": body.timeframe,
        "context_bars": len(window),
        "last_bar": str(window.index[-1]),
        "last_close": float(window["close"].iloc[-1]),
        "elapsed_seconds": round(elapsed, 2),
        "config": spec.as_dict(),
    })
    return payload


@router.post("/ensemble/auto-weights")
def ensemble_auto_weights(body: AutoWeightsIn) -> dict:
    """按近期表现估算建议权重（不写库，由用户在 UI 里确认后再保存）。"""
    dm = _dm()
    df = dm.load_bars(body.symbol, body.timeframe)
    if df is None or df.empty:
        raise HTTPException(status_code=404,
                            detail=f"无数据：{body.symbol} {body.timeframe}，请先下载")
    t0 = datetime.now()
    out = auto_weights(df, pred_len=body.pred_len, eval_bars=body.eval_bars,
                       step=body.step, context=body.context, keys=body.keys,
                       tz="Asia/Shanghai")
    out["elapsed_seconds"] = round((datetime.now() - t0).total_seconds(), 2)
    out["symbol"] = body.symbol
    out["timeframe"] = body.timeframe
    return out

