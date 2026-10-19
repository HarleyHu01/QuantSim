"""模型化预测：所有「需要加载权重或做统计外推」的东西。

统一接口是 ``BaseForecaster``（见 ``alpha.models.base``）：

.. code-block:: python

    class MyForecaster(BaseForecaster):
        key = "mymodel"
        label = "我的模型"
        kind = "dl"                    # stat | dl | external

        def status(self) -> dict: ...  # 只报状态，绝不触发权重下载
        def forecast(self, bars, pred_len) -> pd.DataFrame: ...

上层只消费 ``expected_return(bars, pred_len)``（预测末根 close 相对当前
close 的涨跌幅），所以内部怎么做完全自由。

## 三个层次

- ``baselines.py``：5 个零依赖统计基线（随机游走 / 漂移 / 动量 /
  均值回归 / EWMA）。它们不是「凑数」的——**随机游走是必需的零假设**，
  任何模型跑不赢它，就说明这套预测没有信息量。
- ``kronos.py``：Kronos 金融 K 线基础模型（AAAI 2026）。只吃 OHLCV，
  是这个项目里最贴合的模型，也是唯一默认权重非零的深度学习模型。
- ``external.py``：Chronos / TimesFM / Moirai 等通用时序基础模型的
  **适配器**。这些库 API 各异且版本在变，所以每个都是惰性导入 +
  缺库时优雅降级，不装就不出现。

## 集成

``ensemble.py`` 把上述所有模型加权成单个仓位决策，并加一组硬性门控
（幅度下限、波动率归一、一致性、分歧度、最少模型数）。**默认权重只
开启 kronos + ewma + momentum**，其余为 0——装了才会被用上。

## 加新模型

1. 在 ``alpha/models/`` 新建模块，定义继承 ``BaseForecaster`` 的类；
2. 在 ``ensemble.py`` 的 ``MODEL_FACTORIES`` 里登记 ``key → 类``；
3. 在 ``DEFAULT_WEIGHTS`` 里给个默认权重（不想默认启用就给 0.0）；
4. UI 会自动出现该模型（``list_models()`` 动态枚举）。

需要输出仓位而不只是预测时，在 ``alpha/models/strategies/`` 加一个
``BaseStrategy`` 子类，通过 ``MODEL_FACTORIES`` 拿预测。
"""
from alpha.models.base import (  # noqa: F401
    BaseForecaster,
    ForecasterUnavailable,
    future_index,
    path_to_ohlcv,
    prepare_bars,
    return_sigma,
)
from alpha.models.kronos import (  # noqa: F401
    KronosForecaster,
    KronosUnavailable,
    available,
    default_spec,
    install_hint,
    missing_deps,
)

__all__ = [
    "BaseForecaster",
    "ForecasterUnavailable",
    "KronosForecaster",
    "KronosUnavailable",
    "available",
    "default_spec",
    "future_index",
    "install_hint",
    "missing_deps",
    "path_to_ohlcv",
    "prepare_bars",
    "return_sigma",
]
