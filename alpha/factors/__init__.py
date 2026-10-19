"""因子化策略：只用历史数据算出的规则型信号。

这里的策略共同点是 **零外部权重、完全确定性**——同样的输入永远得到
同样的输出，毫秒级完成，不需要 torch/numpy 之外的任何东西。它们的
逻辑可以直接写成一行条件（``close > sma(close, 20)``），因此可解释、
可审计、可手工推演。

## 目录约定

- 具体策略文件放 ``alpha/factors/strategies/``，用 ``@register`` 注册，
  继承 ``quant.strategies.base.BaseStrategy``。
- 指标计算统一走 ``quant.core.indicators``（sma / ema / macd / rsi /
  bollinger / atr / donchian / roc / zscore），避免每个策略重复实现。
- 想加新因子：先看 ``quant.core.indicators`` 里有没有，没有就在这里
  加纯净函数（输入 Series、输出 Series/DataFrame，不碰 IO、不读全局状态）。

## 和 alpha/models 的边界

如果一个策略需要在运行前加载权重、或者行为依赖随机采样
（例如神经网络推理、采样路径平均），那它属于 ``alpha/models``，
不属于这里——哪怕它最终也只是输出一个 ``target_weight``。
"""
