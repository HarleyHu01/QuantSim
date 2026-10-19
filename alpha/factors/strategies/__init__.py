"""因子化经典策略集合（import 即注册到 registry）。

导入本包会把这 8 个经典策略注册进 ``quant.strategies.registry``，
因此 ``create("dual_ma")`` 之类的调用在任何 import 顺序下都能工作。

策略 key 与模块名一一对应，新增策略时照着已有文件的结构写：
继承 ``BaseStrategy``、定义 ``Params``、实现 ``generate_signals``、
加 ``@register`` 装饰器，然后在这里补一行导入。
"""
from alpha.factors.strategies import bollinger  # noqa: F401
from alpha.factors.strategies import dual_ma  # noqa: F401
from alpha.factors.strategies import grid  # noqa: F401
from alpha.factors.strategies import macd  # noqa: F401
from alpha.factors.strategies import mean_reversion  # noqa: F401
from alpha.factors.strategies import momentum  # noqa: F401
from alpha.factors.strategies import rsi  # noqa: F401
from alpha.factors.strategies import turtle  # noqa: F401

__all__ = [
    "bollinger",
    "dual_ma",
    "grid",
    "macd",
    "mean_reversion",
    "momentum",
    "rsi",
    "turtle",
]
