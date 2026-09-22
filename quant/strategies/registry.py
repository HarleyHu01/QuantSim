"""策略注册表：key → 策略类。内置策略在 import 时自动注册。"""
from __future__ import annotations

from typing import Any

from quant.strategies.base import BaseStrategy

_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """类装饰器：注册策略。"""
    _REGISTRY[cls.key] = cls
    return cls


def get(key: str) -> type[BaseStrategy]:
    if key not in _REGISTRY:
        raise KeyError(f"未知策略: {key}，可用: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def create(key: str, params: dict | None = None) -> BaseStrategy:
    """按 key 创建策略实例；未注册 key 走规则策略加载器（DB 中保存的声明式策略）。"""
    if key in _REGISTRY:
        return _REGISTRY[key](params)
    if _rule_loader is not None:
        s = _rule_loader(key, params)
        if s is not None:
            return s
    raise KeyError(f"未知策略: {key}")


_rule_loader = None


def set_rule_loader(fn) -> None:
    """注册规则策略加载器：key 不在内置注册表时，从这里按 key 从存储加载。"""
    global _rule_loader
    _rule_loader = fn


def _default_rule_loader(key: str, params: dict | None = None):
    """默认加载器：从策略表读取 strategy_type='llm_generated' 的声明式配置。"""
    try:
        import json

        from quant.core.config import AppConfig
        from quant.storage.repositories import StrategiesRepo
        from quant.strategies.rule_strategy import RuleStrategy

        row = StrategiesRepo(AppConfig.load().effective_db_path).get(key)
        if row and row.get("strategy_type") == "llm_generated" and row.get("config_json"):
            return RuleStrategy(config=json.loads(row["config_json"]))
    except Exception:  # noqa: BLE001
        return None
    return None


set_rule_loader(_default_rule_loader)


def list_all() -> list[dict[str, Any]]:
    """策略元信息（含参数默认值与中英文显示名），UI 展示用。"""
    out = []
    for key, cls in sorted(_REGISTRY.items()):
        out.append({
            "key": key,
            "name": cls.name,
            "display_name": getattr(cls, "display_name", cls.name),
            "description": cls.description,
            "params_schema": cls.params_schema(),
            "params_defaults": cls.Params().model_dump(),
        })
    return out


def ensure_loaded() -> None:
    """确保内置策略包被 import（触发注册）。"""
    from quant.strategies import builtin  # noqa: F401
