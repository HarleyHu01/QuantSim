"""用户代码策略：把一段 Python 源码变成可回测的策略实例。

设计目标（对应本步需求「用户可以直接修改代码」）：
- AI 或用户写出的源码无需注册、无需重启，存库后按 key 动态加载
- 加载前做静态安全扫描（AST）+ 冒烟回测（跑一遍 generate_signals）
- 加载失败只影响该策略，不影响整个回测/进程

约定的源码形态（AI 提示词里会强制说明）：

    class Strategy(BaseStrategy):
        key = "my_strategy"
        name = "我的策略"
        class Params(BaseModel):
            window: int = Field(20, ge=2, le=500)
        def generate_signals(self, bars, ctx):
            ...
            return pd.DataFrame({"target_weight": w, "reason": r, "confidence": 1.0})

类名不必叫 Strategy，只要模块里有一个带 generate_signals 的类即可（优先 BaseStrategy 子类）。
"""
from __future__ import annotations

import ast
import math
import types
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.core import indicators
from quant.strategies.base import BaseStrategy

# ---- 静态安全扫描配置 ----
ALLOWED_IMPORTS = {
    "math", "statistics", "datetime", "random", "typing", "dataclasses",
    "pandas", "numpy", "pydantic",
    "quant.core.indicators", "quant.strategies.base",
}

FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "memoryview", "setattr", "delattr",
    "exit", "quit",
}

FORBIDDEN_ATTRS = {
    "__globals__", "__builtins__", "__subclasses__", "__bases__", "__mro__",
    "__code__", "__closure__", "__getattribute__", "__reduce__", "__reduce_ex__",
}

REQUIRED_COLUMNS = ["target_weight", "reason", "confidence"]


class CodeError(ValueError):
    """源码不可用（消息可直接展示给用户）。"""


# ====================== 源码检查 ======================

def check_source(source_code: str) -> list[str]:
    """静态检查源码，返回错误列表（空列表=通过）。不做运行时执行。"""
    errors: list[str] = []
    if not (source_code or "").strip():
        return ["策略源码为空"]
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return [f"语法错误（第 {e.lineno} 行）：{e.msg}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in {a.split(".")[0] for a in ALLOWED_IMPORTS}:
                    errors.append(f"第 {node.lineno} 行：不允许导入 `{alias.name}`")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod not in ALLOWED_IMPORTS:
                errors.append(f"第 {node.lineno} 行：不允许从 `{mod}` 导入")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_NAMES:
                errors.append(f"第 {node.lineno} 行：不允许调用 `{node.func.id}()`")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            errors.append(f"第 {node.lineno} 行：不允许访问 `{node.attr}`")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"第 {node.lineno} 行：不允许使用 `{node.id}`")
    if not _find_strategy_class(tree):
        errors.append("未找到策略类：请定义 `class Strategy(BaseStrategy)` 并实现 generate_signals")
    # 去重保序
    seen, out = set(), []
    for e in errors:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _find_strategy_class(tree: ast.Module) -> ast.ClassDef | None:
    """找一个带 generate_signals 方法的类（优先 BaseStrategy 子类）。"""
    candidates = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    with_signals = [
        c for c in candidates
        if any(isinstance(b, ast.FunctionDef) and b.name == "generate_signals"
               for b in c.body)
    ]
    if not with_signals:
        return None
    # 优先显式继承 BaseStrategy 的类
    for c in with_signals:
        bases = " ".join(ast.unparse(b) for b in c.bases)
        if "BaseStrategy" in bases:
            return c
    named = [c for c in with_signals if c.name in ("Strategy", "MyStrategy")]
    return named[0] if named else with_signals[0]


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
    """受控 import：只允许 ALLOWED_IMPORTS 白名单内的模块。

    check_source 已静态拒绝白名单外的 import，这里再兜底一层（防止动态拼模块名绕过）。
    """
    root = (name or "").split(".")[0]
    if name not in ALLOWED_IMPORTS and root not in {a.split(".")[0] for a in ALLOWED_IMPORTS}:
        raise ImportError(f"不允许导入 `{name}`")
    import builtins
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    import builtins

    allow = [
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
        "format", "frozenset", "getattr", "hasattr", "hash", "int", "isinstance",
        "issubclass", "iter", "len", "list", "map", "max", "min", "next", "object",
        "pow", "property", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "staticmethod", "classmethod", "str", "sum", "super", "tuple",
        "type", "zip", "__build_class__", "Exception", "ValueError", "TypeError",
        "ZeroDivisionError", "KeyError", "IndexError", "RuntimeError", "True", "False",
        "None", "NotImplemented", "print",
    ]
    out = {k: getattr(builtins, k) for k in allow if hasattr(builtins, k)}
    out["__import__"] = _safe_import
    return out


def exec_source(source_code: str) -> dict[str, Any]:
    """执行源码，返回命名空间。调用前应先 check_source。"""
    ns: dict[str, Any] = {
        "__name__": "user_strategy",
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "np": np,
        "math": math,
        "BaseStrategy": BaseStrategy,
        "BaseModel": BaseModel,
        "Field": Field,
        "ClassVar": ClassVar,
        "indicators": indicators,
    }
    try:
        exec(compile(source_code, "<user_strategy>", "exec"), ns)  # noqa: S102 受控沙箱
    except Exception as e:  # noqa: BLE001
        raise CodeError(f"执行策略代码失败：{type(e).__name__}: {e}") from e
    return ns


def pick_class(ns: dict[str, Any]) -> type:
    """从命名空间挑出策略类。"""
    classes = [
        v for k, v in ns.items()
        if isinstance(v, type) and not k.startswith("_")
        and hasattr(v, "generate_signals")
        and getattr(v, "__module__", "") == "user_strategy"
    ]
    if not classes:
        raise CodeError("未找到策略类：请定义 `class Strategy(BaseStrategy)` 并实现 generate_signals")
    for c in classes:
        if issubclass(c, BaseStrategy):
            return c
    return classes[0]


# ====================== 策略包装 ======================

class CodeStrategy(BaseStrategy):
    """把用户源码里的策略类包装成可注册实例（key 由数据库行决定）。"""

    key: ClassVar[str] = "code"
    name: ClassVar[str] = "代码策略"
    display_name: ClassVar[str] = "代码策略 (Code Strategy)"
    description: ClassVar[str] = "用 Python 代码编写的自定义策略（AI 生成或手动编辑）"

    class Params(BaseModel):
        pass

    def __init__(self, source_code: str, key: str = "code",
                 name: str | None = None, description: str | None = None,
                 params: dict | None = None):
        self.source_code = source_code
        errors = check_source(source_code)
        if errors:
            raise CodeError("；".join(errors))
        ns = exec_source(source_code)
        cls = pick_class(ns)
        self.inner_class = cls
        self.key = key
        self.name = name or getattr(cls, "name", None) or key
        self.display_name = getattr(cls, "display_name", None) or self.name
        self.description = description or getattr(cls, "description", "") or ""
        try:
            self.inner = cls(params or {})
        except Exception as e:  # noqa: BLE001
            raise CodeError(f"策略实例化失败（参数不匹配？）：{type(e).__name__}: {e}") from e
        if not isinstance(self.inner, BaseStrategy):
            # 非 BaseStrategy 子类：包一层，保证 generate_signals 可调用
            self.inner = _BareWrapper(self.inner, params or {})
        self.params = self.inner.params

    @classmethod
    def params_schema(cls) -> dict:
        """CodeStrategy 自身的 Params 为空；真正暴露给 UI 的是源码里的 Params。"""
        return {}

    def schema(self) -> dict:
        """当前策略实例的参数 JSON Schema（来自源码内 Params 类）。"""
        params_cls = getattr(self.inner_class, "Params", None)
        if isinstance(params_cls, type) and issubclass(params_cls, BaseModel):
            return params_cls.model_json_schema()
        return {}

    def generate_signals(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        return self.inner.generate_signals(bars, ctx)


class _BareWrapper:
    """非 BaseStrategy 子类的兜底包装（必须自带 generate_signals）。"""

    def __init__(self, inst: Any, params: dict):
        self._inst = inst
        try:
            self.params = inst.build_params(params)
        except Exception:  # noqa: BLE001
            self.params = types.SimpleNamespace(**(params or {}))

    def generate_signals(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        return self._inst.generate_signals(bars, ctx)


# ====================== 对外工具 ======================

def inspect_source(source_code: str) -> dict[str, Any]:
    """读源码元信息（名称/描述/参数 schema/默认值），供 UI 与 API 使用。"""
    errors = check_source(source_code)
    if errors:
        return {"ok": False, "errors": errors}
    try:
        ns = exec_source(source_code)
        cls = pick_class(ns)
    except CodeError as e:
        return {"ok": False, "errors": [str(e)]}
    schema: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    params_cls = getattr(cls, "Params", None)
    if isinstance(params_cls, type) and issubclass(params_cls, BaseModel):
        schema = params_cls.model_json_schema()
        try:
            defaults = params_cls().model_dump()
        except Exception:  # noqa: BLE001
            defaults = {}
    return {
        "ok": True,
        "errors": [],
        "class_name": cls.__name__,
        "name": getattr(cls, "name", None) or cls.__name__,
        "description": getattr(cls, "description", "") or "",
        "params_schema": schema,
        "params_defaults": defaults,
    }


def smoke_test(source_code: str, params: dict | None = None,
               bars: pd.DataFrame | None = None) -> dict[str, Any]:
    """冒烟回测：用合成行情跑一遍 generate_signals，校验输出契约。"""
    errors = check_source(source_code)
    if errors:
        return {"ok": False, "errors": errors}
    bars = _synthetic_bars() if bars is None else bars
    try:
        strat = CodeStrategy(source_code, key="smoke", params=params)
    except CodeError as e:
        return {"ok": False, "errors": [str(e)]}
    try:
        df = strat.generate_signals(bars, None)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "errors": [f"generate_signals 运行出错：{type(e).__name__}: {e}"]}
    errs = validate_signal_output(df, len(bars))
    if errs:
        return {"ok": False, "errors": errs}
    w = pd.to_numeric(df["target_weight"], errors="coerce").fillna(0.0)
    return {
        "ok": True,
        "errors": [],
        "bars": int(len(bars)),
        "signal_count": int((w != 0).sum()),
        "max_weight": float(w.max()) if len(w) else 0.0,
        "min_weight": float(w.min()) if len(w) else 0.0,
        "last_reason": str(df["reason"].iloc[-1]) if len(df) else "",
    }


def validate_signal_output(df: Any, n_bars: int | None = None) -> list[str]:
    """校验 generate_signals 的输出契约。"""
    if not isinstance(df, pd.DataFrame):
        return [f"generate_signals 必须返回 DataFrame，实际是 {type(df).__name__}"]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"返回的 DataFrame 缺少列：{', '.join(missing)}（需要 "
                f"{', '.join(REQUIRED_COLUMNS)}）"]
    if n_bars is not None and len(df) != n_bars:
        return [f"返回行数 {len(df)} 与行情根数 {n_bars} 不一致"]
    if len(df) == 0:
        return ["返回的 DataFrame 为空"]
    w = pd.to_numeric(df["target_weight"], errors="coerce")
    if w.isna().all():
        return ["target_weight 不是数值"]
    return []


def _synthetic_bars(n: int = 240) -> pd.DataFrame:
    """合成一段有趋势+波动的行情，用于冒烟测试。"""
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    base = 100.0
    drift = np.linspace(0, 25, n)
    wave = 6.0 * np.sin(np.linspace(0, 8 * np.pi, n))
    noise = np.random.default_rng(7).normal(0, 0.6, n)
    close = base + drift + wave + noise
    return pd.DataFrame({
        "open": np.roll(close, 1),
        "high": close * 1.012,
        "low": close * 0.988,
        "close": close,
        "volume": np.random.default_rng(11).uniform(8e3, 2e4, n),
    }, index=idx).assign(open=lambda d: d["open"].fillna(d["close"]))
