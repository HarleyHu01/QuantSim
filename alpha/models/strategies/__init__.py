"""基于预测模型的策略（导入即注册）。

这些策略和 ``alpha.factors.strategies`` 里的经典策略满足同一个
``BaseStrategy`` 契约，所以能无缝进回测、组合、模拟盘。它们的差别
只在信号来源：不是算均线，而是**调用模型做预测**。

- ``kronos_forecast``：单模型。Kronos 预测未来 N 根，涨跌幅超过
  ``threshold_sigma`` 倍波动率就建仓。
- ``ensemble_forecast``：多模型。Kronos + 统计基线 + 可选外部时序模型
  加权投票，还要通过一致性/分歧度/最少模型数等硬性门控才建仓。
- ``fusion``：**融合策略**。把上面的模型集成和 ``alpha.factors``
  的因子策略（双均线、MACD…）自由组合：用户选模型与权重、选因子
  与权重、选结合方式（加权混合 / 模型闸门 / 因子闸门 / 投票制）。

导入本包时如果 ``torch`` 等依赖缺失，策略仍会被注册（类定义本身不依赖
torch），只是运行到预测那一步才会抛 ``KronosUnavailable`` 并把安装命令
写进 reason——这样 UI 里能看到策略、也能提前知道要装什么。
"""
from alpha.models.strategies import ensemble_forecast  # noqa: F401
from alpha.models.strategies import fusion  # noqa: F401
from alpha.models.strategies import kronos_forecast  # noqa: F401

__all__ = ["ensemble_forecast", "fusion", "kronos_forecast"]
