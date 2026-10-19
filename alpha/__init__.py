"""alpha —— 策略研究与预测模型的统一入口。

项目把「能产生交易信号的东西」分成两类，都放在这里，与 ``quant/``（引擎、
数据、存储等基础设施）、``web/``（界面）解耦：

.. code-block:: text

    alpha/
    ├── factors/          因子化：只用历史数据算的规则型指标/策略
    │   └── strategies/   已注册的内置经典策略（双均线、MACD、RSI…）
    └── models/           模型化：需要加载权重或统计外推的预测模型
        ├── baselines.py  零依赖统计基线（随机游走、动量、均值回归…）
        ├── kronos.py     Kronos 金融 K 线基础模型封装
        ├── external.py   Chronos / TimesFM / Moirai 通用时序模型适配器
        ├── ensemble.py   多模型加权集成 + 硬性门控
        └── strategies/   把模型预测转成仓位的策略

## 为什么这么分

**因子化**是「人先写死逻辑，机器只负责计算」：``close > sma(20)`` 就是
一个因子，行为完全可预测、可解释、毫秒级。**模型化**是「机器从数据里
学出映射」：给它 240 根 K 线，它吐未来 12 根的价格路径——你没法定量
解释它为什么这么预测，只能看历史表现。

两者需要的运行环境也完全不同：因子化零依赖，模型化可能要几百 MB 到
数 GB 的权重和一个 GPU。放在同一个目录下按子包隔离，就能做到
**没装 torch 的用户照样能用全部经典策略**。

## 怎么用

策略注册走 ``quant.strategies.registry``，两条路径都会在
``ensure_loaded()`` 里被 import：

.. code-block:: python

    from quant.strategies.registry import create, ensure_loaded

    ensure_loaded()
    strat = create("dual_ma", {"fast": 10, "slow": 30})      # 因子化
    strat = create("ensemble_forecast", {"weights_json": "{}"})  # 模型化

预测模型本身也可以单独用：

.. code-block:: python

    from alpha.models import EnsembleForecaster, EnsembleSpec

    ens = EnsembleForecaster(EnsembleSpec())
    result = ens.predict(bars, pred_len=12)
    print(result.reason)     # 为什么做多/空仓（含每条硬门控的实测值）

## 加新东西放哪里

- 一个新指标组合 / 手工规则 → ``alpha/factors/strategies/``
- 一个新预测模型（LSTM、Transformer、某篇论文的实现）→ ``alpha/models/``，
  继承 ``alpha.models.base.BaseForecaster`` 实现 ``forecast()``，
  然后在 ``alpha/models/ensemble.py`` 的 ``MODEL_FACTORIES`` 里登记
- 想让它参与集成投票 → 同上登记即可，权重在 UI 里调
"""

__all__ = ["factors", "models"]
