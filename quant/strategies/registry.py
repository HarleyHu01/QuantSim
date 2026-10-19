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
    """默认加载器：从策略表按 key 还原策略实例。

    支持两种存储形态：
    - strategy_type='llm_generated'：声明式 JSON 配置 → RuleStrategy
    - strategy_type='code'：Python 源码 → CodeStrategy（AI 生成或用户手改）

    加载后强制用数据库行的 key 覆写实例 key，保证同一批回测里多个自定义策略
    的子账户/信号 key 唯一。
    """
    try:
        import json

        from quant.core.config import AppConfig
        from quant.storage.repositories import StrategiesRepo

        row = StrategiesRepo(AppConfig.load().effective_db_path).get(key)
        if not row:
            return None
        stype = row.get("strategy_type")

        if stype == "code" and (row.get("source_code") or "").strip():
            from quant.strategies.code_strategy import CodeStrategy

            strat = CodeStrategy(
                row["source_code"], key=key,
                name=row.get("name") or None,
                description=row.get("description") or None,
                params=params)
        elif stype == "llm_generated" and row.get("config_json"):
            from quant.strategies.rule_strategy import RuleStrategy

            strat = RuleStrategy(config=json.loads(row["config_json"]))
        else:
            return None
        strat.key = key
        return strat
    except Exception:  # noqa: BLE001 任何加载失败都视为「未找到」，由调用方报错
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
    """确保所有内置策略包被 import（触发注册）。

    两类策略分居两处，都要加载：

    - ``alpha.factors.strategies``：因子化经典策略（双均线、MACD、RSI…），
      零依赖，必须成功；
    - ``alpha.models.strategies``：模型化策略（Kronos 预测、多模型集成），
      依赖 torch 等可选组件，导入失败只记 debug 日志，不让注册表变空。
    """
    from alpha.factors import strategies as _factor_strategies  # noqa: F401

    try:
        from alpha.models import strategies as _model_strategies  # noqa: F401
    except Exception:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).debug("alpha.models.strategies 未加载", exc_info=True)


