"""外部时序基础模型适配器（Chronos / TimesFM / Moirai）。

这三个模型和 Kronos 属于同一类思路：在**大规模时序语料**上预训练，
推理时给一段历史就输出未来分布或点预测。差别在于数据域——

- **Kronos**：只吃金融 K 线（OHLCV token 化），最贴合本项目
- **Chronos**（Amazon）：通用时序，T5 编码器做「数值分箱 + 语言化」
- **TimesFM**（Google）：通用时序，decoder-only patch 模型
- **Moirai**（Salesforce）：通用时序，任意变量/频率的统一掩码建模

为什么用「适配器 + 惰性导入」而不是直接写死：

1. 这些库的 API 各不相同、版本间还在变，硬依赖会拖累所有用户；
2. 每个都要几百 MB 到数 GB 权重，全装一遍不现实；
3. 用户可能只装了其中一个。

所以每个适配器都是：**探测到库 → 用最小公共接口调用 → 折算出统一的
「预期收益」**；库不在就 ``available=False`` + `install_hint`，UI 显示
「未安装」，绝不抛异常炸掉整个页面。

## 统一接口约定

所有适配器继承 :class:`BaseForecaster`，实现两个方法：

- ``status()``：只查状态，**绝不触发权重下载**
- ``forecast(bars, pred_len)``：返回 ``DataFrame[open, high, low, close, volume, amount]``

集成层只消费 ``expected_return(bars, pred_len)``，所以适配器内部怎么算
（点预测 / 分位数中位数 / 采样均值）对上层透明。
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Any

import numpy as np
import pandas as pd

from alpha.models.base import (BaseForecaster, ForecasterUnavailable, future_index,
                           path_to_ohlcv, prepare_bars, return_sigma)

log = logging.getLogger(__name__)


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _pick_torch_device(wanted: str = "auto") -> str:
    """tensorflow/torch 都可能没有，这里只处理 torch 侧。"""
    if wanted and wanted != "auto":
        return wanted
    if not _has_module("torch"):
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - 探测失败就当 CPU，不阻塞
        pass
    return "cpu"


class _ExternalForecaster(BaseForecaster):
    """外部模型的公共骨架：可用性检查 + 点预测转 OHLCV。"""

    kind = "external"
    #: 需要的 python 模块
    requires: tuple[str, ...] = ()
    #: 安装命令（缺模块时给用户看）
    install_cmd: str = ""
    #: 说明
    summary_text: str = ""

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self._model = None
        self._load_error: str | None = None
        self._load_seconds: float | None = None

    # ---- 可用性 ----
    def missing(self) -> list[str]:
        return [m for m in self.requires if not _has_module(m)]

    @property
    def available(self) -> bool:
        return not self.missing()

    def status(self) -> dict[str, Any]:
        miss = self.missing()
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "available": not miss,
            "loaded": self._model is not None,
            "requires": list(self.requires),
            "missing": miss,
            "install_cmd": self.install_cmd if miss else "",
            "device": _pick_torch_device(str(self.params.get("device", "auto"))),
            "load_seconds": round(self._load_seconds, 2) if self._load_seconds else None,
            "error": self._load_error,
            "params": {k: v for k, v in self.params.items() if k != "device"},
            "summary": self.summary_text,
        }

    # ---- 加载 ----
    def _ensure(self):
        if self._model is not None:
            return self._model
        miss = self.missing()
        if miss:
            raise ForecasterUnavailable(
                f"{self.label} 依赖未安装：{', '.join(miss)}。安装命令：{self.install_cmd}"
            )
        import time as _t

        t0 = _t.time()
        try:
            self._model = self._load()
        except ForecasterUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 - 权重下载失败等，转成可读错误
            self._load_error = f"{type(e).__name__}: {e}"
            raise ForecasterUnavailable(f"{self.label} 加载失败：{self._load_error}") from e
        self._load_seconds = _t.time() - t0
        log.info("%s 加载完成 %.1fs", self.label, self._load_seconds)
        return self._model

    def _load(self):
        """真正加载权重。子类实现。"""
        raise NotImplementedError

    def _point_forecast(self, ctx: pd.DataFrame, pred_len: int) -> np.ndarray:
        """返回长度为 pred_len 的未来收盘价数组。子类实现。"""
        raise NotImplementedError

    # ---- 推理 ----
    def forecast(self, bars: pd.DataFrame, pred_len: int) -> pd.DataFrame:
        n = int(pred_len)
        if n <= 0:
            raise ValueError(f"pred_len 必须为正整数，收到 {pred_len}")
        ctx = prepare_bars(bars, min_bars=8)
        max_ctx = int(self.params.get("max_context", 512))
        if len(ctx) > max_ctx:
            ctx = ctx.iloc[-max_ctx:]

        self._ensure()
        path = np.asarray(self._point_forecast(ctx, n), dtype=float)
        if path.size != n:
            if path.size == 0:
                raise ValueError(f"{self.label} 未返回预测值")
            n = int(path.size)

        idx = future_index(ctx.index, n)
        sigma = return_sigma(ctx["close"])
        last_close = float(ctx["close"].iloc[-1])
        last_vol = float(ctx["volume"].iloc[-1]) if "volume" in ctx else 0.0
        closes = pd.Series(path, index=idx, dtype=float)
        closes = closes.replace([np.inf, -np.inf], np.nan).ffill().fillna(last_close)
        return path_to_ohlcv(closes, last_close=last_close, sigma=sigma, last_volume=last_vol)


class ChronosForecaster(_ExternalForecaster):
    """Amazon Chronos：把数值序列分箱成 token，用 T5 做语言模型式预测。

    论文：*Chronos: Learning the Language of Time Series*（arXiv:2403.07815）。

    接口（amazon/chronos 及后续 chronos-forecasting 包）::

        pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small",
                                                   device_map="cpu", torch_dtype=torch.bfloat16)
        samples = pipeline.predict(context_tensor, prediction_length=n)  # [batch, samples, n]

    我们取样本**中位数**作为点预测：比均值抗异常路径，也比单次采样稳。
    """

    key = "chronos"
    label = "Chronos（Amazon 通用时序）"
    requires = ("torch", "chronos")
    install_cmd = "pip install chronos-forecasting"
    summary_text = "Amazon 的通用时序基础模型（T5 分箱 + 语言建模），取采样本中位数作点预测。"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.params.setdefault("repo_id", "amazon/chronos-t5-small")
        self.params.setdefault("num_samples", 20)
        self.params.setdefault("max_context", 512)

    def _load(self):
        import torch
        from chronos import BaseChronosPipeline

        device = _pick_torch_device(str(self.params.get("device", "auto")))
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        return BaseChronosPipeline.from_pretrained(
            str(self.params["repo_id"]), device_map=device, torch_dtype=dtype
        )

    def _point_forecast(self, ctx: pd.DataFrame, pred_len: int) -> np.ndarray:
        import torch

        pipe = self._ensure()
        series = torch.tensor(ctx["close"].astype(float).to_numpy(), dtype=torch.float32)
        samples = pipe.predict(series, prediction_length=int(pred_len),
                              num_samples=int(self.params.get("num_samples", 20)))
        arr = samples.detach().cpu().numpy() if hasattr(samples, "detach") else np.asarray(samples)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            return arr
        # 允许 [samples, n] 与 [1, samples, n]
        arr = arr.reshape(-1, arr.shape[-1])
        return np.median(arr, axis=0)


class TimesFMForecaster(_ExternalForecaster):
    """Google TimesFM：decoder-only patch 模型，零样本点预测。

    论文：*A Decoder-only Foundation Model for Time-series Forecasting*（arXiv:2310.10688）。

    接口::

        model = timesfm.TimesFm(hparams, checkpoint)
        model.load_checkpoint(...)  # 或 from_pretrained（新版本）
        point, _ = model.forecast(horizon=n, inputs=[array])
    """

    key = "timesfm"
    label = "TimesFM（Google 通用时序）"
    requires = ("torch", "timesfm")
    install_cmd = "pip install timesfm"
    summary_text = "Google 的 decoder-only patch 时序基础模型，零样本直接给点预测。"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.params.setdefault("max_context", 512)
        self.params.setdefault("repo_id", "google/timesfm-1.0-200m-pytorch")

    def _load(self):
        import timesfm

        # 新旧版本 API 不一致：优先 from_pretrained，退回手工构造
        if hasattr(timesfm, "TimesFm"):
            try:
                return timesfm.TimesFm.from_pretrained(str(self.params["repo_id"]))
            except Exception:  # noqa: BLE001 - 老版本没有 from_pretrained
                hparams = timesfm.TimesFmHparams(
                    backend=_pick_torch_device(str(self.params.get("device", "auto"))),
                    per_core_batch_size=1, horizon_len=128,
                    context_len=int(self.params.get("max_context", 512)),
                    num_layers=50, use_positional_embedding=False,
                )
                checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=str(self.params["repo_id"]))
                return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
        raise ForecasterUnavailable("timesfm 已安装但找不到 TimesFm 接口，版本可能不兼容")

    def _point_forecast(self, ctx: pd.DataFrame, pred_len: int) -> np.ndarray:
        model = self._ensure()
        values = ctx["close"].astype(float).to_numpy()
        out = model.forecast(horizon=int(pred_len), inputs=[values])
        # 新版返回 (point, quantiles)，老版只返回 point
        point = out[0] if isinstance(out, tuple) else out
        arr = np.asarray(point, dtype=float).reshape(-1)
        return arr[: int(pred_len)]


class MoiraiForecaster(_ExternalForecaster):
    """Salesforce Moirai：统一掩码建模，支持任意变量数/频率。

    论文：*Unified Training of Universal Time Series Forecasting Transformers*（arXiv:2402.02592）。
    通过官方 ``uni2ts`` 包调用。
    """

    key = "moirai"
    label = "Moirai（Salesforce 通用时序）"
    requires = ("torch", "uni2ts")
    install_cmd = "pip install uni2ts"
    summary_text = "Salesforce 的统一掩码时序模型，取 P50 分位数作为点预测。"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.params.setdefault("repo_id", "Salesforce/moirai-1.0-R-small")
        self.params.setdefault("max_context", 512)

    def _load(self):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        module = MoiraiModule.from_pretrained(str(self.params["repo_id"]))
        return MoiraiForecast(
            module=module,
            prediction_length=int(self.params.get("pred_len", 12)),
            context_length=int(self.params.get("max_context", 512)),
            patch_size="auto",
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

    def _point_forecast(self, ctx: pd.DataFrame, pred_len: int) -> np.ndarray:
        import torch

        model = self._ensure()
        values = torch.tensor(ctx["close"].astype(float).to_numpy(), dtype=torch.float32)
        past_target = values.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        observed = torch.ones_like(past_target, dtype=torch.bool)
        with torch.no_grad():
            pred = model(past_target=past_target, past_observed_target=observed,
                         past_is_pad=~observed)
        samples = np.asarray(pred.detach().cpu().numpy()).reshape(-1, int(pred_len))
        return np.median(samples, axis=0)


#: 全部外部模型（顺序即 UI 展示顺序）
EXTERNAL_FORECASTERS: dict[str, type[_ExternalForecaster]] = {
    cls.key: cls
    for cls in (ChronosForecaster, TimesFMForecaster, MoiraiForecaster)
}


def external_status() -> list[dict[str, Any]]:
    """一次性返回所有外部模型状态（不触发任何加载/下载）。"""
    return [cls().status() for cls in EXTERNAL_FORECASTERS.values()]
