"""策略管理 API：启用/停用、权重、参数持久化、声明式规则策略创建（第 5/7 步）。"""
from __future__ import annotations

import json
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from quant.api.state import app_state
from quant.strategies.registry import ensure_loaded, get, list_all

router = APIRouter(prefix="/strategies", tags=["strategies"])


class StrategyIn(BaseModel):
    key: str
    params: dict | None = None
    weight: float = 100.0
    enabled: bool = True


class StrategyUpdateIn(BaseModel):
    params: dict | None = None
    weight: float | None = None
    enabled: bool | None = None


class RuleStrategyIn(BaseModel):
    """手动表单 / LLM 生成的声明式策略（RuleStrategyConfig 全量字段）。"""

    name: str
    description: str = ""
    indicators: list[dict] = []
    entry_long: dict | None = None
    exit_long: dict | None = None
    entry_short: dict | None = None
    exit_short: dict | None = None
    position: dict | None = None
    params: dict | None = None
    key: str | None = None  # 缺省由英文名生成
    enabled: bool = False  # 保存为草稿（enabled=0）；回测通过后在 UI 激活


def _repo():
    repo = app_state.get("strategies_repo")
    if repo is None:
        raise HTTPException(status_code=503, detail="策略仓储未就绪")
    return repo


def _slugify(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    return s or "rule_strategy"


def _rule_meta(row: dict) -> dict:
    """声明式策略行的展示元信息（来自 config）。"""
    from quant.strategies.llm.schema import RuleParams
    from quant.strategies.rule_strategy import RuleStrategy

    try:
        cfg = json.loads(row.get("config_json") or "{}")
    except Exception:  # noqa: BLE001
        cfg = {}
    return {
        "name": cfg.get("name") or row.get("name") or "规则策略",
        "display_name": cfg.get("name") or row.get("name") or "规则策略",
        "description": cfg.get("description", ""),
        "params_schema": RuleStrategy.Params.model_json_schema(),
        "params": cfg.get("params") or {},
        "config": cfg,
    }


def _code_meta(row) -> dict:
    """代码策略的展示元信息（参数 schema 从源码静态解析，不执行）。"""
    from quant.strategies.code_strategy import inspect_source

    code = row.get("source_code") or ""
    info = inspect_source(code) if code.strip() else {"ok": False, "errors": ["无源码"]}
    try:
        cfg = json.loads(row.get("config_json") or "{}")
    except Exception:  # noqa: BLE001
        cfg = {}
    return {
        "name": row.get("name") or "AI 代码策略",
        "display_name": row.get("name") or "AI 代码策略",
        "description": row.get("description", ""),
        "params_schema": info.get("params_schema") or {},
        "params": cfg.get("params_defaults") or info.get("params_defaults") or {},
        "has_code": bool(code.strip()),
        "code_ok": bool(info.get("ok")),
        "code_errors": info.get("errors", []),
        "code_lines": len(code.splitlines()),
    }


@router.get("")
def list_strategies() -> dict:
    """DB 策略行 + 注册表/声明式元信息（params_schema 供 UI 渲染参数表单）。"""
    ensure_loaded()
    metas = {m["key"]: m for m in list_all()}
    out = []
    for row in _repo().list_all():
        if row.get("strategy_type") == "llm_generated":
            meta = _rule_meta(row)
            out.append({**row, **meta})
        elif row.get("strategy_type") == "code":
            out.append({**row, **_code_meta(row)})
        else:
            meta = metas.get(row["key"], {})
            out.append({
                **row,
                "name": row["name"] or meta.get("name", ""),
                "display_name": meta.get("display_name") or row["name"] or meta.get("name", ""),
                "description": meta.get("description", ""),
                "params_schema": meta.get("params_schema", {}),
                "params": json.loads(row.get("config_json") or "{}"),
            })
    return {"strategies": out}


@router.get("/indicators")
def list_indicators() -> dict:
    """规则策略表单元信息：指标白名单 / 操作符 / 引用字段 / 仓位模式。"""
    from quant.strategies.rule_strategy import RuleStrategy

    return RuleStrategy.config_schema()


@router.post("")
def add_strategy(body: StrategyIn) -> dict:
    ensure_loaded()
    cls = get(body.key)  # 校验注册表存在该策略
    repo = _repo()
    repo.upsert(key=body.key, name=cls.name, description=cls.description,
                strategy_type="builtin", class_name=cls.__name__,
                config_json=body.params or {}, enabled=int(body.enabled), weight=body.weight)
    return repo.get(body.key)


@router.post("/rule")
def add_rule_strategy(body: RuleStrategyIn) -> dict:
    """保存声明式策略（手动表单或 LLM 生成共用同一校验链）。

    - 校验不通过 → 400 + 错误明细
    - 缺省保存为草稿（enabled=0），回测验证通过后由 UI 激活
    """
    from quant.strategies.llm.schema import RuleStrategyConfig
    from quant.strategies.llm.validator import validate_config

    kwargs = dict(
        name=body.name, description=body.description,
        indicators=body.indicators or [],
        entry_long=body.entry_long, exit_long=body.exit_long,
        entry_short=body.entry_short, exit_short=body.exit_short,
    )
    if body.position is not None:  # 缺省用模型默认值，避免 None 覆盖
        kwargs["position"] = body.position
    if body.params is not None:
        kwargs["params"] = body.params
    try:
        cfg = RuleStrategyConfig(**kwargs)
    except Exception as e:  # noqa: BLE001  pydantic 校验失败 → 400
        raise HTTPException(status_code=400, detail={"errors": [str(e)]})
    errors = validate_config(cfg)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    key = (body.key or _slugify(body.name)).lower()
    repo = _repo()
    if body.key is None:
        # 自动生成 key：中文名 fallback 为 rule_strategy，冲突时加后缀避免覆盖
        base = key
        i = 2
        while repo.get(key) is not None:
            key = f"{base}_{i}"
            i += 1
    existing = repo.get(key)
    repo.upsert(key=key, name=cfg.name, description=cfg.description,
                strategy_type="llm_generated", class_name="RuleStrategy",
                config_json=cfg.model_dump(by_alias=True), enabled=int(body.enabled),
                weight=(existing or {}).get("weight", 100.0))
    row = repo.get(key)
    return {"strategy": {**row, **_rule_meta(row)}, "config": cfg.model_dump(by_alias=True)}


@router.put("/{strategy_id}")
def update_strategy(strategy_id: int, body: StrategyUpdateIn) -> dict:
    repo = _repo()
    row = repo.get_by_id(strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="策略不存在")
    params = body.params
    stype = row.get("strategy_type")
    if params is not None and stype == "llm_generated":
        # 声明式策略：params 只更新可调子集，保留 indicators/conditions
        try:
            cfg = json.loads(row.get("config_json") or "{}")
        except Exception:  # noqa: BLE001
            cfg = {}
        clean = {k: (None if v == "None" else v) for k, v in (params or {}).items()}
        cfg["params"] = {**(cfg.get("params") or {}), **clean}
        params = cfg
    elif params is not None and stype == "code":
        # 代码策略：参数由源码 Params 定义，存 params_defaults 供回测覆盖
        clean = {k: (None if v == "None" else v) for k, v in (params or {}).items()}
        params = {"params_defaults": clean}
    repo.update(strategy_id, config_json=params, enabled=body.enabled, weight=body.weight)
    updated = repo.get_by_id(strategy_id)
    if updated.get("strategy_type") == "llm_generated":
        return {**updated, **_rule_meta(updated)}
    if updated.get("strategy_type") == "code":
        return {**updated, **_code_meta(updated)}
    return updated


@router.post("/{strategy_id}/toggle")
def toggle_strategy(strategy_id: int) -> dict:
    repo = _repo()
    row = repo.get_by_id(strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="策略不存在")
    repo.update(strategy_id, enabled=0 if row["enabled"] else 1)
    return repo.get_by_id(strategy_id)


@router.delete("/{strategy_id}")
def delete_strategy(strategy_id: int) -> dict:
    """删除策略。

    - 内置策略（builtin）→ 403 不可删除
    - 手动/规则策略（llm_generated）→ 允许删除
    - 被组合配置引用的策略 → 级联从组合权重中移除后删除
    """
    repo = _repo()
    row = repo.get_by_id(strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="策略不存在")
    if row.get("strategy_type") == "builtin":
        raise HTTPException(status_code=403, detail="内置策略不可删除")
    removed = _remove_from_portfolio(row["key"])
    repo.delete(strategy_id)
    return {"ok": True, "strategy_id": strategy_id, "key": row["key"],
            "removed_from_portfolio": removed}


def _remove_from_portfolio(key: str) -> bool:
    """从已保存的组合配置中移除该策略权重（被引用时返回 True）。"""
    port_repo = app_state.get("portfolio_repo")
    if port_repo is None:
        return False
    saved = port_repo.get()
    if not saved:
        return False
    try:
        weights = json.loads(saved.get("weights_json") or "{}")
    except Exception:  # noqa: BLE001
        weights = {}
    if key not in weights:
        return False
    del weights[key]
    port_repo.put(mode=saved.get("mode") or "signal_merge", weights=weights,
                  rebalance_freq=saved.get("rebalance_freq") or "none",
                  signal_threshold=float(saved.get("signal_threshold") or 0.0))
    return True
