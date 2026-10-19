"""大模型策略生成 API：自然语言对话 → 策略代码 / 规则配置 → 保存 → 手改。

路由（全部挂 /api/llm）：
- GET  /llm/status          当前配置是否可用 + 模型信息
- POST /llm/ping            连通性测试
- POST /llm/chat            多轮对话生成（核心）
- GET  /llm/sessions        会话列表
- POST /llm/sessions        新建会话
- GET  /llm/sessions/{id}   会话详情（含消息）
- DELETE /llm/sessions/{id} 删除会话
- POST /llm/validate        校验代码/配置（静态检查 + 冒烟回测）
- POST /llm/save            把生成结果保存成策略（代码策略 or 规则策略）
- GET  /llm/code/{sid}      读取策略源码（手动编辑用）
- PUT  /llm/code/{sid}      保存手改后的源码（含校验，可选择立即回测）
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from quant.api.state import app_state
from quant.core.config import AppConfig
from quant.llm.client import LLMError, LLMResult, client_from_settings
from quant.llm.prompts import (INTENTS, MODES, build_messages, explanation_of,
                               extract_code, extract_json, ideas_text, parse_ideas)
from quant.strategies.code_strategy import (CodeError, check_source, inspect_source,
                                            smoke_test)
from quant.strategies.llm.schema import RuleStrategyConfig
from quant.strategies.llm.validator import validate_config

log = logging.getLogger(__name__)
router = APIRouter(prefix="/llm", tags=["llm"])

MAX_REQUEST_CHARS = 8000
MAX_CODE_CHARS = 200_000


# ====================== 依赖 ======================

def _settings_repo():
    return app_state.get("settings_repo")


def _llm_repo():
    repo = app_state.get("llm_repo")
    if repo is None:
        from quant.storage.repositories import LLMRepo

        repo = LLMRepo(AppConfig.load().effective_db_path)
        app_state["llm_repo"] = repo
    return repo


def _strategies_repo():
    repo = app_state.get("strategies_repo")
    if repo is None:
        raise HTTPException(status_code=503, detail="策略仓储未就绪")
    return repo


def _client():
    return client_from_settings(settings_repo=_settings_repo())


def _chat(messages: list[dict[str, str]], *, temperature: float | None = None,
          json_mode: bool = False, provider: str | None = None,
          model: str | None = None):
    """统一对话入口：默认走 DeepSeek/OpenAI 兼容配置，provider=workbuddy 走 WorkBuddy。

    返回一个带 .content / .model / .usage 属性的对象（LLMResult 兼容）。
    """
    if provider == "workbuddy":
        from quant.llm import workbuddy as wb

        wb_model = (model or "auto")
        r = wb.WorkBuddyClient().chat(messages, wb_model, temperature=temperature,
                                      json_mode=json_mode)
        return LLMResult(content=r["content"], model=r["model"], usage=r["usage"])
    # 默认 DeepSeek/OpenAI 兼容：若指定了 model 则临时覆盖
    if model:
        from quant.llm.client import resolve_llm_config

        cfg = resolve_llm_config(settings_repo=_settings_repo())
        cfg.model = model
        from quant.llm.client import LLMClient
        return LLMClient(cfg).chat(messages, temperature=temperature, json_mode=json_mode)
    return _client().chat(messages, temperature=temperature, json_mode=json_mode)


def _slugify(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name or "").strip("_").lower() or "ai_strategy"


def _unique_key(base: str, repo, exclude_key: str | None = None) -> str:
    base = (base or "ai_strategy")[:40]
    key, i = base, 1
    while True:
        row = repo.get(key)
        if row is None or (exclude_key and key == exclude_key):
            return key
        i += 1
        key = f"{base}_{i}"


# ====================== 请求模型 ======================

class ChatIn(BaseModel):
    message: str
    session_id: int | None = None
    mode: str = "code"          # code | rule
    intent: str = "write"       # write | revise | ideas
    code: str | None = None     # 带上则按「修改现有代码」处理
    config: dict | None = None  # 规则模式下已生成的配置（改配置时避免推倒重来）
    context: dict | None = None # {symbol, timeframe, bars_count}
    temperature: float | None = None
    provider: str | None = None  # "deepseek"(默认) | "workbuddy"
    model: str | None = None     # provider=workbuddy 时指定模型（默认 auto）


class ValidateIn(BaseModel):
    mode: str = "code"
    code: str | None = None
    config: dict | None = None
    params: dict | None = None


class SaveIn(BaseModel):
    mode: str = "code"
    name: str = ""
    description: str = ""
    key: str | None = None
    strategy_id: int | None = None   # 有则更新已有策略
    code: str | None = None
    config: dict | None = None
    params: dict | None = None
    enabled: bool = False
    session_id: int | None = None


class CodeIn(BaseModel):
    code: str
    name: str | None = None
    description: str | None = None
    params: dict | None = None
    enabled: bool | None = None


# ====================== 基础接口 ======================

@router.get("/status")
def llm_status() -> dict[str, Any]:
    """前端初始化用：是否配置了 key、模型名、支持的模式。"""
    from quant.llm.client import PROTOCOLS, resolve_llm_config

    cfg = resolve_llm_config(settings_repo=_settings_repo())
    return {
        "configured": bool((cfg.api_key or "").strip()),
        "base_url": cfg.base_url,
        "model": cfg.model,
        "api_protocol": cfg.api_protocol,
        "protocols": [{"key": k, "label": v} for k, v in PROTOCOLS.items()],
        "temperature": cfg.temperature,
        "timeout_seconds": cfg.timeout_seconds,
        "max_retries": cfg.max_retries,
        "modes": [{"key": k, "label": v} for k, v in MODES.items()],
    }


@router.post("/ping")
def llm_ping() -> dict[str, Any]:
    try:
        return _client().ping()
    except LLMError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"连通性测试失败: {e}") from e


@router.get("/models")
def llm_models() -> dict[str, Any]:
    """探测当前接口地址下可用的模型列表（OpenAI 兼容 GET /models）。"""
    try:
        models = _client().list_models()
    except LLMError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"models": models, "count": len(models)}


# ====================== 核心：对话生成 ======================

@router.post("/chat")
def llm_chat(body: ChatIn) -> dict[str, Any]:
    """一轮对话：生成代码/配置 + 校验 + 落库会话。"""
    request = (body.message or "").strip()
    if not request:
        raise HTTPException(status_code=400, detail="请输入你的策略需求")
    if len(request) > MAX_REQUEST_CHARS:
        raise HTTPException(status_code=400, detail=f"需求描述过长（上限 {MAX_REQUEST_CHARS} 字）")
    mode = body.mode if body.mode in MODES else "code"
    intent = body.intent if body.intent in INTENTS else "write"
    repo = _llm_repo()

    session = None
    if body.session_id:
        session = repo.get_session(int(body.session_id))
    if session is None:
        title = request[:24] + ("…" if len(request) > 24 else "")
        session = repo.create_session(title=title, mode=mode)
    sid = int(session["id"])

    history = [{"role": m["role"], "content": m["content"]}
               for m in repo.list_messages(sid) if m.get("content")]
    prev_code = body.code
    if not prev_code:
        # 没显式传代码时，沿用会话里最近一次生成的代码，便于「继续改」
        for m in reversed(repo.list_messages(sid)):
            if m.get("role") == "assistant" and m.get("code"):
                prev_code = m["code"]
                break
    prev_config = body.config
    if not prev_config and mode == "rule":
        for m in reversed(repo.list_messages(sid)):
            meta = m.get("meta") or {}
            if m.get("role") == "assistant" and meta.get("config"):
                prev_config = meta["config"]
                break

    messages = build_messages(mode, request, code=prev_code, history=history,
                              context=body.context, intent=intent, config=prev_config)
    repo.add_message(sid, "user", request, meta={"mode": mode, "intent": intent})

    try:
        result = _chat(messages, temperature=body.temperature,
                       json_mode=(mode == "rule" and intent != "ideas"),
                       provider=body.provider, model=body.model)
    except LLMError as e:
        repo.add_message(sid, "assistant", f"生成失败：{e}",
                         meta={"error": True, "mode": mode, "intent": intent})
        repo.log_prompt(request, model="", strategy_id=session.get("strategy_id"),
                        mode=mode, ok=0, detail=str(e)[:500])
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001  WorkBuddy 等其它 provider 的错误
        repo.add_message(sid, "assistant", f"生成失败：{e}",
                         meta={"error": True, "mode": mode, "intent": intent})
        repo.log_prompt(request, model="", strategy_id=session.get("strategy_id"),
                        mode=mode, ok=0, detail=str(e)[:500])
        raise HTTPException(status_code=400, detail=str(e)) from e

    if intent == "ideas":
        return _ideas_response(repo, sid, session, request, mode, result)

    payload = _validate_result(mode, result.content)
    explanation = explanation_of(result.content, payload.get("code") or "", mode)

    msg = repo.add_message(
        sid, "assistant", explanation or _default_explanation(mode),
        code=payload.get("code") or "",
        meta={"mode": mode, "intent": intent, "model": result.model,
              "usage": result.usage,
              "valid": payload["ok"], "errors": payload["errors"],
              "config": payload.get("config"), "smoke": payload.get("smoke")})
    # 首轮自动把会话标题改成更像样
    if sid and (session.get("title") or "").startswith("新对话"):
        repo.update_session(sid, title=request[:24])
    repo.log_prompt(request, model=result.model, strategy_id=session.get("strategy_id"),
                    mode=mode, ok=1 if payload["ok"] else 0,
                    detail=(payload["errors"][0] if payload["errors"] else "")[:500])
    return {"session_id": sid, "message": msg, "result": payload,
            "model": result.model, "usage": result.usage,
            "version": _version_of(repo, sid)}


def _ideas_response(repo, sid: int, session: dict, request: str, mode: str,
                    result) -> dict[str, Any]:
    """思路模式：只发散想法，不产出策略，所以不校验、不存 config/code。"""
    ideas = parse_ideas(result.content)
    content = ideas_text(result.content) or "下面是我的一些思路："
    msg = repo.add_message(sid, "assistant", content,
                           meta={"mode": mode, "intent": "ideas",
                                 "model": result.model, "usage": result.usage,
                                 "ideas": ideas})
    if (session.get("title") or "").startswith("新对话"):
        repo.update_session(sid, title=request[:24])
    repo.log_prompt(request, model=result.model,
                    strategy_id=session.get("strategy_id"), mode=mode, ok=1,
                    detail=f"ideas x{len(ideas)}"[:500])
    return {"session_id": sid, "message": msg, "ideas": ideas,
            "result": {"ok": False, "errors": [], "mode": mode, "code": None,
                       "config": None, "smoke": None, "ideas": ideas},
            "model": result.model, "usage": result.usage,
            "version": _version_of(repo, sid)}


def _version_of(repo, session_id: int) -> dict[str, Any]:
    """当前策略版本：会话里最近一版通过校验的代码/配置 + 版本号。

    UI 靠它做「实时查看当前版本」：每成功生成一版，version 就 +1。
    """
    version = 0
    for m in repo.list_messages(session_id):
        if m.get("role") != "assistant":
            continue
        meta = m.get("meta") or {}
        if meta.get("intent") == "ideas" or meta.get("error"):
            continue
        if m.get("code") or meta.get("config"):
            version += 1
    if version == 0:
        return {"n": 0, "code": None, "config": None, "valid": False,
                "message_id": None, "explanation": ""}
    for m in reversed(repo.list_messages(session_id)):
        if m.get("role") != "assistant":
            continue
        meta = m.get("meta") or {}
        if meta.get("intent") == "ideas" or meta.get("error"):
            continue
        if m.get("code") or meta.get("config"):
            return {"n": version, "code": m.get("code") or None,
                    "config": (meta or {}).get("config"),
                    "valid": bool(meta.get("valid")),
                    "message_id": m.get("id"),
                    "explanation": m.get("content") or ""}
    return {"n": version, "code": None, "config": None, "valid": False,
            "message_id": None, "explanation": ""}


@router.post("/validate")
def validate(body: ValidateIn) -> dict[str, Any]:
    """校验用户/AI 提供的代码或规则配置（不改库）。"""
    if body.mode == "rule":
        if not body.config:
            raise HTTPException(status_code=400, detail="缺少 config")
        try:
            cfg = RuleStrategyConfig(**_normalize_rule(body.config))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "errors": [f"配置结构不合法: {e}"]}
        errors = validate_config(cfg)
        return {"ok": not errors, "errors": errors,
                "config": cfg.model_dump(by_alias=True)}
    if not (body.code or "").strip():
        raise HTTPException(status_code=400, detail="缺少 code")
    if len(body.code) > MAX_CODE_CHARS:
        raise HTTPException(status_code=400, detail="代码过长")
    info = inspect_source(body.code)
    if not info.get("ok"):
        return {"ok": False, "errors": info.get("errors", []), "inspect": info}
    smoke = smoke_test(body.code, body.params)
    return {"ok": bool(smoke.get("ok")), "errors": smoke.get("errors", []),
            "inspect": info, "smoke": smoke}


# ====================== 会话管理 ======================

@router.get("/sessions")
def list_sessions(limit: int = 50) -> dict[str, Any]:
    return {"sessions": _llm_repo().list_sessions(limit=limit)}


@router.post("/sessions")
def create_session(mode: str = "code") -> dict[str, Any]:
    return _llm_repo().create_session(mode=mode if mode in MODES else "code")


@router.get("/sessions/{session_id}")
def get_session(session_id: int) -> dict[str, Any]:
    repo = _llm_repo()
    session = repo.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    session["messages"] = repo.list_messages(session_id)
    return session


@router.delete("/sessions/{session_id}")
def delete_session(session_id: int) -> dict[str, Any]:
    _llm_repo().delete_session(session_id)
    return {"ok": True, "session_id": session_id}


# ====================== 保存为策略 ======================

@router.post("/save")
def save_strategy(body: SaveIn) -> dict[str, Any]:
    """把生成结果落成可回测的策略；已有 strategy_id 时原地更新。"""
    repo = _strategies_repo()
    mode = body.mode if body.mode in MODES else "code"

    if mode == "rule":
        return _save_rule(repo, body)
    return _save_code(repo, body)


def _save_rule(repo, body: SaveIn) -> dict[str, Any]:
    if not body.config:
        raise HTTPException(status_code=400, detail="缺少 config")
    errors: list[str] = []
    try:
        cfg = RuleStrategyConfig(**_normalize_rule(body.config))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400,
                            detail={"errors": [f"配置结构不合法: {e}"]}) from e
    if body.name:
        cfg.name = body.name
    if body.description:
        cfg.description = body.description
    errors = validate_config(cfg)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    existing = repo.get_by_id(body.strategy_id) if body.strategy_id else None
    key = existing["key"] if existing else (body.key or _unique_key(_slugify(cfg.name), repo))
    strategy_id = repo.upsert(
        key=key, name=cfg.name, description=cfg.description,
        strategy_type="llm_generated", class_name="RuleStrategy",
        config_json=cfg.model_dump(by_alias=True),
        enabled=int(bool(body.enabled)), weight=(existing or {}).get("weight", 100.0))
    row = repo.get(key) or {"id": strategy_id, "key": key}
    if body.session_id:
        _llm_repo().update_session(body.session_id, strategy_id=int(row["id"]))
    return {"ok": True, "strategy": row, "mode": "rule"}


def _save_code(repo, body: SaveIn) -> dict[str, Any]:
    code = (body.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="缺少 code")
    info = inspect_source(code)
    if not info.get("ok"):
        raise HTTPException(status_code=400, detail={"errors": info.get("errors", [])})
    smoke = smoke_test(code, body.params)
    if not smoke.get("ok"):
        raise HTTPException(
            status_code=400,
            detail={"errors": smoke.get("errors", []),
                    "hint": "代码未通过冒烟回测，请检查 generate_signals 的返回契约"})

    name = (body.name or info.get("name") or "AI 策略").strip()
    description = (body.description or info.get("description") or "").strip()
    existing = repo.get_by_id(body.strategy_id) if body.strategy_id else None
    key = existing["key"] if existing else (body.key or _unique_key(_slugify(name), repo))
    strategy_id = repo.upsert(
        key=key, name=name, description=description,
        strategy_type="code", class_name=info.get("class_name") or "Strategy",
        config_json={"params_defaults": info.get("params_defaults") or {}},
        enabled=int(bool(body.enabled)), weight=(existing or {}).get("weight", 100.0),
        source_code=code)
    row = repo.get(key) or {"id": strategy_id, "key": key}
    if body.session_id:
        _llm_repo().update_session(body.session_id, strategy_id=int(row["id"]))
    return {"ok": True, "strategy": row, "mode": "code", "smoke": smoke,
            "inspect": info}


# ====================== 源码读取 / 手动编辑 ======================

@router.get("/code/{strategy_id}")
def get_code(strategy_id: int) -> dict[str, Any]:
    repo = _strategies_repo()
    row = repo.get_by_id(strategy_id)
    if not row:
        raise HTTPException(status_code=404, detail="策略不存在")
    code = row.get("source_code") or ""
    info = inspect_source(code) if code.strip() else {"ok": False, "errors": ["该策略没有源码"]}
    return {"strategy_id": strategy_id, "key": row.get("key"), "name": row.get("name"),
            "description": row.get("description"), "strategy_type": row.get("strategy_type"),
            "code": code, "inspect": info}


@router.put("/code/{strategy_id}")
def put_code(strategy_id: int, body: CodeIn) -> dict[str, Any]:
    """保存用户手改后的源码：先静态检查 + 冒烟回测，通过才写库。"""
    repo = _strategies_repo()
    row = repo.get_by_id(strategy_id)
    if not row:
        raise HTTPException(status_code=404, detail="策略不存在")
    if row.get("strategy_type") == "builtin":
        raise HTTPException(status_code=403, detail="内置策略不可修改源码")
    code = (body.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="代码为空")
    if len(code) > MAX_CODE_CHARS:
        raise HTTPException(status_code=400, detail="代码过长")
    errors = check_source(code)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    smoke = smoke_test(code, body.params)
    if not smoke.get("ok"):
        raise HTTPException(status_code=400,
                            detail={"errors": smoke.get("errors", []),
                                    "hint": "代码未通过冒烟回测，未保存"})
    info = inspect_source(code)
    repo.update(
        strategy_id,
        source_code=code,
        name=(body.name or info.get("name") or row.get("name")),
        description=(body.description if body.description is not None
                     else info.get("description") or row.get("description")),
        enabled=(None if body.enabled is None else int(bool(body.enabled))),
        config_json={"params_defaults": info.get("params_defaults") or {}},
    )
    return {"ok": True, "strategy_id": strategy_id, "smoke": smoke, "inspect": info}


# ====================== 内部：结果校验 ======================

def _validate_result(mode: str, content: str) -> dict[str, Any]:
    """从模型输出解析并校验，返回统一结构。"""
    out: dict[str, Any] = {"ok": False, "errors": [], "mode": mode,
                           "code": None, "config": None, "smoke": None}
    if mode == "rule":
        try:
            cfg_dict = extract_json(content)
        except ValueError as e:
            out["errors"] = [str(e)]
            return out
        try:
            cfg = RuleStrategyConfig(**_normalize_rule(cfg_dict))
        except Exception as e:  # noqa: BLE001
            out["errors"] = [f"配置结构不合法: {e}"]
            out["config"] = cfg_dict
            return out
        errors = validate_config(cfg)
        out["config"] = cfg.model_dump(by_alias=True)
        out["errors"] = errors
        out["ok"] = not errors
        return out

    code = extract_code(content)
    out["code"] = code
    errors = check_source(code)
    if errors:
        out["errors"] = errors
        return out
    info = inspect_source(code)
    if not info.get("ok"):
        out["errors"] = info.get("errors", [])
        return out
    smoke = smoke_test(code)
    out["smoke"] = smoke
    out["inspect"] = info
    if not smoke.get("ok"):
        out["errors"] = smoke.get("errors", [])
        return out
    out["ok"] = True
    return out


def _normalize_rule(raw: dict[str, Any]) -> dict[str, Any]:
    """容错处理模型常见偏差：缺字段、conditions 数组形式。"""
    cfg = dict(raw or {})
    cfg.setdefault("name", "AI 策略")
    cfg.setdefault("description", "")
    cfg.setdefault("indicators", [])
    cfg.setdefault("position", {"mode": "fixed_pct", "value": 1.0, "max_position": 1.0})
    params = dict(cfg.get("params") or {})
    params.setdefault("only_long", not params.get("allow_short", False))
    cfg["params"] = params
    for g in ("entry_long", "exit_long", "entry_short", "exit_short"):
        cfg[g] = _normalize_group(cfg.get(g))
    return cfg


def _normalize_group(group: Any) -> Any:
    """{"all": [...]} / {"any": [...]} / {"logic","conditions"} → 标准结构。"""
    if not group:
        return None
    if isinstance(group, list):
        return {"logic": "all", "conditions": group}
    if isinstance(group, dict):
        if "conditions" in group:
            return group
        for logic in ("all", "any"):
            if logic in group and isinstance(group[logic], list):
                return {"logic": logic, "conditions": group[logic]}
    return None


def _default_explanation(mode: str) -> str:
    return "已生成策略代码。" if mode == "code" else "已生成规则配置。"
