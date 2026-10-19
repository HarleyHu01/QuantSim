# 来源与改动说明

本目录是 [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) 的**最小化内联副本**，
用于在本项目内直接做推理，无需 pip 安装外部包、也不依赖上游仓库的目录布局。

- 上游许可证：MIT（见同目录 `LICENSE`，Copyright (c) 2025 ShiYu）
- 模型权重：Hugging Face `NeoQuasar/Kronos-small` + `NeoQuasar/Kronos-Tokenizer-base`
  （通过 `huggingface_hub` 在首次使用时下载并缓存到 `~/.cache/huggingface`）
- 论文：Kronos: A Foundation Model for the Language of Financial Markets
  （AAAI 2026, arXiv:2508.02739）

## 相对上游源码的改动（仅一处）

`kronos.py` 顶部的导入：

```python
# 上游（依赖仓库自身布局，作为包使用时会 ImportError）
sys.path.append("../")
from model.module import *

# 本目录
from .module import *
```

其余 `kronos.py` / `module.py` 内容与上游 `master` 分支一致，未做逻辑修改。

## 本地验证过的版本

CPU 推理（`torch` 2.14.1+cpu、`einops` 0.8.2、`huggingface_hub` 2.2.0）：

- 模型加载约 50 秒（首次含权重下载，之后走本地缓存）
- 400 根上下文 + 预测 12 根：约 0.7 秒/次（`sample_count=1`）
- 参数量：`Kronos-small` 24.7M
