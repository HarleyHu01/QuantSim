"""Kronos K 线基础模型推理封装（可选组件，按需加载）。

Kronos 是首个面向金融 K 线序列的开源基础模型（AAAI 2026，arXiv:2508.02739），
用两阶段框架做多步 OHLCV 预测：
  1. Tokenizer 把连续的多维 K 线量化成分层离散 token；
  2. 自回归 Transformer 在这些 token 上预训练，输出未来若干根 K 线。

## 为什么按需加载

`torch` 的 CPU 轮子就有 200MB+，不是所有用户都需要。所以：

- 模块顶层**不 import torch**，只有 `available()` 用 `importlib.util.find_spec` 探测；
- 真正推理时才 `import` 并加载权重（`KronosForecaster.load()`）；
- 依赖缺失抛 `KronosUnavailable`，调用方（API/UI）转成可读提示，不影响其他功能运转。

## 实测性能（RTX 4060 机器上的 CPU 推理）

- 首次加载约 50 秒（含 Hugging Face 权重下载；之后走 `~/.cache/huggingface` 缓存约 3 秒）
- 400 根上下文 + 预测 12 根，`sample_count=1`：约 0.7 秒/次
- 参数量：`Kronos-small` 24.7M / `Kronos-base` 102.3M / `Kronos-mini` 4.1M

## 用法

```python
from alpha.models import KronosForecaster
f = KronosForecaster.load_default()
pred = f.forecast(bars, pred_len=12)   # DataFrame[open, high, low, close, volume, amount]
```
"""
from __future__ import annotations

import importlib.util
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# 依赖清单：(模块名, 安装提示)
_REQUIRED = (
    ("torch", "torch"),
    ("einops", "einops"),
    ("huggingface_hub", "huggingface_hub"),
    ("safetensors", "safetensors"),
    ("tqdm", "tqdm"),
)
#: 公开别名：外部模块（如集成层）要展示「需要装什么」，不该去摸私有名
REQUIRED_DEPS = _REQUIRED

MODEL_CHOICES = {
    "Kronos-mini": "NeoQuasar/Kronos-mini",
    "Kronos-small": "NeoQuasar/Kronos-small",
    "Kronos-base": "NeoQuasar/Kronos-base",
}
TOKENIZER_CHOICES = {
    "Kronos-mini": "NeoQuasar/Kronos-Tokenizer-2k",
    "Kronos-small": "NeoQuasar/Kronos-Tokenizer-base",
    "Kronos-base": "NeoQuasar/Kronos-Tokenizer-base",
}
# 各模型的上下文上限（上游 README）：mini 2048，small/base 512
MAX_CONTEXT = {"Kronos-mini": 2048, "Kronos-small": 512, "Kronos-base": 512}


class KronosUnavailable(RuntimeError):
    """依赖缺失或模型加载失败。"""


def missing_deps() -> list[str]:
    """返回缺失的依赖模块名（空列表 = 齐备）。"""
    return [mod for mod, _ in _REQUIRED if importlib.util.find_spec(mod) is None]


def available() -> bool:
    return not missing_deps()


def install_hint() -> str:
    """缺失依赖时的安装命令（CPU 版 torch，体积最小）。"""
    names = missing_deps()
    if not names:
        return ""
    pkgs = " ".join(n for n in names if n != "torch")
    parts = []
    if "torch" in names:
        parts.append("pip install --index-url https://download.pytorch.org/whl/cpu torch")
    if pkgs:
        parts.append(f"pip install {pkgs}")
    return " && ".join(parts)


@dataclass
class KronosSpec:
    """一次推理的配置。"""

    enabled: bool = True  # 配置里关掉后策略/接口统一报「未启用」
    model: str = "Kronos-small"
    tokenizer: str = ""  # 空 → 按 model 自动配对
    device: str = "auto"  # auto | cpu | cuda:0
    max_context: int = 512
    pred_len: int = 12
    temperature: float = 1.0
    top_p: float = 0.9
    sample_count: int = 1
    tz: str = "Asia/Shanghai"  # 时间特征按本地时区计算（模型吃的是「几点」这类特征）

    def resolved_tokenizer(self) -> str:
        return self.tokenizer or TOKENIZER_CHOICES.get(self.model, "NeoQuasar/Kronos-Tokenizer-base")

    def resolved_model(self) -> str:
        """短名（Kronos-small）或完整 repo id（NeoQuasar/Kronos-small）都接受。

        配置里写短名更友好，但 HF 上真正的 repo id 带 `NeoQuasar/` 前缀，
        直接拿短名去 from_pretrained 会 401 Repository Not Found。
        含 "/" 的一律当完整 id 原样使用（便于换自定义/本地微调模型）。
        """
        if "/" in self.model:
            return self.model
        return MODEL_CHOICES.get(self.model, f"NeoQuasar/{self.model}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.resolved_model(),
            "tokenizer": self.resolved_tokenizer(),
            "device": self.device,
            "max_context": self.max_context,
            "pred_len": self.pred_len,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "sample_count": self.sample_count,
            "tz": self.tz,
        }


def default_spec() -> KronosSpec:
    return KronosSpec()


@dataclass
class _Loaded:
    """已加载的模型对（tokenizer + model + predictor），按 (model, tokenizer, device) 缓存。"""

    model: Any
    tokenizer: Any
    predictor: Any
    device: str
    load_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


_CACHE: dict[tuple[str, str, str], _Loaded] = {}
_LOCK = threading.Lock()


def _pick_device(wanted: str) -> str:
    if wanted and wanted != "auto":
        return wanted
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - 探测失败就当 CPU
        pass
    return "cpu"


def _model_config(repo_id: str, retries: int = 3) -> dict[str, Any]:
    """下载并读取 HF 仓库的 config.json（带重试）。

    为什么要自己读：`PyTorchModelHubMixin.from_pretrained` 内部把 config 下载失败
    的异常 `except` 掉了（`except HfHubHTTPError`），然后拿**空 kwargs** 去构造模型，
    报出来的是 `TypeError: Kronos.__init__() missing 11 required positional arguments`，
    与真实原因（网络/缓存瞬时故障）完全无关。这里显式取 config，绕开那个静默分支。
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import CONFIG_NAME

    last: Exception | None = None
    for i in range(max(1, retries)):
        try:
            path = hf_hub_download(repo_id=repo_id, filename=CONFIG_NAME)
            import json

            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001 - 网络抖动，退避重试
            last = e
            log.warning("读取 %s 的 config.json 失败（第 %d 次）：%s", repo_id, i + 1, e)
            time.sleep(min(4.0, 0.8 * 2**i))
    raise KronosUnavailable(f"无法读取 {repo_id} 的 config.json：{last}")


def _load_pretrained(cls: Any, repo_id: str) -> Any:
    """按显式 config 构造 `nn.Module` 并加载权重。"""
    cfg = _model_config(repo_id)
    # 只保留该类 __init__ 认识的键（两个类共用同一份 config 字段名约定）
    init_params = getattr(cls, "_hub_mixin_init_parameters", {})
    kwargs = {k: v for k, v in cfg.items() if k in init_params} if init_params else dict(cfg)
    try:
        return cls.from_pretrained(repo_id, **kwargs)
    except Exception as e:  # noqa: BLE001 - 转成带上下文的可读错误
        raise KronosUnavailable(f"加载 {repo_id} 失败：{type(e).__name__}: {e}") from e


class KronosForecaster:
    """Kronos 预测器封装（线程安全地惰性加载，进程内复用同一份权重）。"""

    def __init__(self, spec: KronosSpec | None = None):
        self.spec = spec or default_spec()

    # ---- 构造/加载 ----
    @classmethod
    def default(cls) -> "KronosForecaster":
        """按配置文件里的 ml.kronos 段构造（未配置则用默认 spec）。"""
        try:
            from quant.core.config import AppConfig

            cfg = AppConfig.load().ml.kronos
            spec = KronosSpec(
                enabled=bool(cfg.enabled),
                model=cfg.model,
                tokenizer=cfg.tokenizer,
                device=cfg.device,
                max_context=cfg.max_context,
                pred_len=cfg.pred_len,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                sample_count=cfg.sample_count,
                tz=cfg.tz,
            )
        except Exception:  # noqa: BLE001 - 配置读不到就用默认，不让预测功能因配置挂掉
            spec = default_spec()
        return cls(spec)

    def _load(self) -> _Loaded:
        if not self.spec.enabled:
            raise KronosUnavailable("Kronos 已在配置中关闭（ml.kronos.enabled = false）")
        if not available():
            raise KronosUnavailable(
                f"Kronos 依赖未安装：{', '.join(missing_deps())}。安装命令：{install_hint()}"
            )

        device = _pick_device(self.spec.device)
        tokenizer_name = self.spec.resolved_tokenizer()
        model_id = self.spec.resolved_model()
        key = (model_id, tokenizer_name, device)
        with _LOCK:
            hit = _CACHE.get(key)
            if hit is not None:
                return hit

            import time as _t

            from alpha.models._kronos import Kronos, KronosPredictor, KronosTokenizer

            t0 = _t.time()
            log.info("加载 Kronos 模型 model=%s tokenizer=%s device=%s", model_id, tokenizer_name, device)
            tokenizer = _load_pretrained(KronosTokenizer, tokenizer_name)
            model = _load_pretrained(Kronos, model_id)
            predictor = KronosPredictor(model, tokenizer, device=device, max_context=self.spec.max_context)
            loaded = _Loaded(
                model=model,
                tokenizer=tokenizer,
                predictor=predictor,
                device=device,
                load_seconds=_t.time() - t0,
                extra={"params_million": sum(p.numel() for p in model.parameters()) / 1e6},
            )
            _CACHE[key] = loaded
            log.info("Kronos 加载完成 %.1fs 参数量 %.1fM", loaded.load_seconds, loaded.extra["params_million"])
            return loaded

    # ---- 状态 ----
    def status(self) -> dict[str, Any]:
        """不触发加载的状态查询（UI/API 用）。"""
        device = _pick_device(self.spec.device)
        key = (self.spec.resolved_model(), self.spec.resolved_tokenizer(), device)
        hit = _CACHE.get(key)
        return {
            "enabled": self.spec.enabled,
            "available": available(),
            "missing": missing_deps(),
            "install_hint": install_hint(),
            "loaded": hit is not None,
            "device": hit.device if hit else device,
            "load_seconds": round(hit.load_seconds, 2) if hit else None,
            "params_million": round(hit.extra.get("params_million", 0.0), 1) if hit else None,
            "spec": self.spec.as_dict(),
            "model_choices": list(MODEL_CHOICES),
            "max_context_by_model": MAX_CONTEXT,
        }

    # ---- 推理 ----
    def forecast(
        self,
        bars: pd.DataFrame,
        pred_len: int | None = None,
        *,
        future_index: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """对 `bars` 之后 `pred_len` 根 K 线做预测。

        bars: 需含 open/high/low/close，volume 可选（缺失按 0），索引须是 DatetimeIndex。
        返回：index = 未来时间戳，列 = [open, high, low, close, volume, amount]。
        """
        # 注意不能用 pred_len or default：pred_len=0 会被 or 吞掉，
        # 变成「静默用默认值」，调用方以为改了长度其实没改。
        n_pred = int(self.spec.pred_len if pred_len is None else pred_len)
        if n_pred <= 0:
            raise ValueError(f"pred_len 必须为正整数，收到 {pred_len}")
        if not isinstance(bars, pd.DataFrame) or bars.empty:
            raise ValueError("bars 为空，无法预测")

        need = {"open", "high", "low", "close"}
        missing = need - set(bars.columns)
        if missing:
            raise ValueError(f"bars 缺少列：{sorted(missing)}")

        df = bars.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("bars 的索引必须是 DatetimeIndex")
        # 上游按「本地时间」提取 minute/hour/weekday 特征，UTC 索引会喂错特征
        if df.index.tz is not None:
            df.index = df.index.tz_convert(self.spec.tz)
        else:
            df.index = df.index.tz_localize(self.spec.tz)

        if "volume" not in df.columns:
            df["volume"] = 0.0
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
        # 上游对 NaN 直接抛错，这里先清理（停牌等场景常见）
        df = df[["open", "high", "low", "close", "volume", "amount"]].astype(float)
        df = df.dropna(subset=["open", "high", "low", "close"])
        if df.empty:
            raise ValueError("bars 清洗后为空（价格列含 NaN）")

        max_ctx = int(self.spec.max_context or MAX_CONTEXT.get(self.spec.model, 512))
        if len(df) > max_ctx:
            df = df.iloc[-max_ctx:]

        loaded = self._load()
        if future_index is None:
            y_index = self._future_index(df.index, n_pred)
        else:
            y_index = pd.DatetimeIndex(future_index)
            if y_index.tz is None:
                y_index = y_index.tz_localize(self.spec.tz)
            n_pred = len(y_index)

        out = loaded.predictor.predict(
            df=df,
            x_timestamp=pd.Series(df.index),
            y_timestamp=pd.Series(y_index),
            pred_len=n_pred,
            T=float(self.spec.temperature),
            top_p=float(self.spec.top_p),
            sample_count=int(self.spec.sample_count),
            verbose=False,
        )
        return out

    def _future_index(self, last_index: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
        """按已有索引的间隔外推未来时间戳（等间隔采集，避开交易日历的复杂度）。"""
        if len(last_index) >= 2:
            deltas = pd.Series(last_index).diff().dropna()
            step = deltas.median()
            if pd.isna(step) or step <= pd.Timedelta(0):
                step = pd.Timedelta(days=1)
        else:
            step = pd.Timedelta(days=1)
        return pd.DatetimeIndex([last_index[-1] + step * (i + 1) for i in range(n)])
