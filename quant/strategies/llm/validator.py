"""声明式策略配置校验链（手动表单与 LLM 生成共用）。

校验：指标白名单 / 参数完整性 / 输出名唯一 / 条件引用存在 / 操作符白名单 /
between 区间格式 / crosses 右值 / 仓位模式 / 做空互斥。返回错误列表（空 = 通过）。
"""
from __future__ import annotations

from quant.strategies.llm.schema import (BUILTIN_REFS, OPERATORS, POSITION_MODES,
                                         INDICATOR_WHITELIST, ConditionGroup,
                                         RuleStrategyConfig)


def validate_config(cfg: RuleStrategyConfig) -> list[str]:
    errors: list[str] = []
    seen_outputs: set[str] = set()

    for spec in cfg.indicators:
        meta = INDICATOR_WHITELIST.get(spec.name)
        if meta is None:
            errors.append(f"未知指标: {spec.name}")
            continue
        for p in meta["params"]:
            if p["name"] not in spec.params:
                errors.append(f"指标 {spec.name} 缺少参数 {p['name']}")
        for out in meta["outputs"]:
            name = out.format(**{"as": spec.as_})
            if name in seen_outputs:
                errors.append(f"输出名重复: {name}")
            seen_outputs.add(name)

    _validate_group(cfg.entry_long, "买入", seen_outputs, errors)
    _validate_group(cfg.exit_long, "卖出", seen_outputs, errors)
    _validate_group(cfg.entry_short, "做空", seen_outputs, errors)
    _validate_group(cfg.exit_short, "平空", seen_outputs, errors)

    if cfg.position.mode not in POSITION_MODES:
        errors.append(f"未知仓位模式: {cfg.position.mode}")

    if cfg.params.allow_short and cfg.params.only_long:
        errors.append("允许做空与仅做多互斥，不能同时开启")

    return errors


def _validate_group(group: ConditionGroup | None, label: str,
                    seen_outputs: set[str], errors: list[str]) -> None:
    if group is None:
        return
    if group.logic not in ("all", "any"):
        errors.append(f"{label} 条件组合逻辑非法: {group.logic}")
    if not group.conditions:
        errors.append(f"{label} 条件为空，至少需要一条")
        return
    for c in group.conditions:
        if c.lhs not in seen_outputs and c.lhs not in BUILTIN_REFS:
            errors.append(f"{label} 条件左侧引用未定义: {c.lhs}")
        if c.op not in OPERATORS:
            errors.append(f"{label} 未知操作符: {c.op}")
            continue
        if c.op in ("between", "not_between"):
            if not isinstance(c.rhs, list) or len(c.rhs) != 2:
                errors.append(f"{label} 条件 {c.op} 需要区间 [下限, 上限]")
        elif c.op in ("crosses_above", "crosses_below"):
            if isinstance(c.rhs, str) and c.rhs not in seen_outputs and c.rhs not in BUILTIN_REFS:
                errors.append(f"{label} 条件 {c.op} 的右值引用未定义: {c.rhs}")
        else:
            if isinstance(c.rhs, str) and c.rhs not in seen_outputs and c.rhs not in BUILTIN_REFS:
                errors.append(f"{label} 条件右值引用未定义: {c.rhs}")
