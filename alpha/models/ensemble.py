"""多模型集成：把多个预测器的结论加权成一个仓位决策。

## 为什么需要集成，而不是「挑最好的那个」

单模型的问题是**它错的时候你不知道**。集成的价值不在于平均掉噪声，
而在于「模型之间分歧有多大」本身就是信息：

- 四个模型都说涨 → 高置信度做多
- 两个说涨、两个说跌 → 分歧极大，这条行情大概率没有边际，不该下注
- 全都说不涨不跌 → 空仓等待

所以本模块分两步：**加权合成** → **硬性门控**。

## 第一步：加权合成

每个模型的输出先折算成同一个量 —— 「预测末端收益」``r_i``。然后：

    r_combined = Σ w_i · r_i / Σ w_i

权重可以手调（UI 滑块），也可以按历史表现自动算（``auto_weight``：
用最近 ``eval_bars`` 根做滚动回看，按 |相关性| × 胜率给分）。

## 第二步：硬性门控（hard gates）

这是「不改主意就不下注」的部分。以下任一条件不满足 → 强制空仓：

========================  ==================================================
门                        作用
========================  ==================================================
``min_abs_return``        预测幅度绝对下限，滤掉噪声
``min_return_sigma``      预测幅度 ≥ N 倍近期波动率（按波动率归一，跨周期通用）
``min_agreement``         至少多大比例的（有效权重的）模型方向一致
``max_dispersion``        模型间预测的标准差 ≤ N 倍波动率（分歧太大就弃权）
``min_models``            至少几个模型给出了有效预测
========================  ==================================================

**为什么要 ``max_dispersion``**：如果四个模型预测 +3%、-2%、+0.1%、-5%，
加权平均可能刚好是 +0.02% 看似中性，但真实情况是「没人知道」。
用标准差把这种情形识别出来并弃权，比让平均值蒙混过关诚实得多。

## 明确不做的事

- **不做「模型投票 + 多数决」**：多数决会丢掉幅度信息，
  三个模型都只预测 +0.01% 也能压过两个模型预测 +5%。
- **不做在线学习**：``auto_weight`` 只是把历史 IC 当权重，
  不更新任何模型参数。可解释、可复现，也能被硬门控兜住。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from alpha.models.base import (BaseForecaster, ForecasterUnavailable, return_sigma)
from alpha.models.baselines import STAT_FORECASTERS
from alpha.models.external import EXTERNAL_FORECASTERS

log = logging.getLogger(__name__)

#: 集成配置在 kv_store 里的键
CONFIG_KEY = "ml.ensemble"


class KronosAdapter(BaseForecaster):
    """把 `KronosForecaster` 包装成统一的预测器接口。

    Kronos 是本项目最贴合的模型（只吃金融 K 线），但它依赖 torch，
    所以 `available` 要动态探测而不是写死。
    """

    key = "kronos"
    label = "Kronos（金融 K 线基础模型）"
    kind = "dl"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self._fc = None
        self._error: str | None = None

    def _status_module(self):
        from alpha.models import kronos as kr

        return kr

    def status(self) -> dict[str, Any]:
        kr = self._status_module()
        miss = kr.missing_deps()
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "available": not miss,
            "loaded": self._fc is not None,
            "requires": [m for m, _ in kr.REQUIRED_DEPS],
            "missing": miss,
            "install_cmd": kr.install_hint() if miss else "",
            "device": "auto",
            "load_seconds": None,
            "error": self._error,
            "params": dict(self.params),
            "summary": "只吃金融 K 线的两阶段模型（tokenizer + 自回归 Transformer），"
                       "预测多步 OHLCV。CPU 约 0.7s/次。",
        }

    def _forecaster(self):
        if self._fc is not None:
            return self._fc
        kr = self._status_module()
        if not kr.available():
            raise ForecasterUnavailable(
                f"Kronos 依赖未安装：{', '.join(kr.missing_deps())}。安装命令：{kr.install_hint()}"
            )
        from alpha.models.kronos import MAX_CONTEXT, KronosSpec

        model = str(self.params.get("model", "Kronos-small"))
        spec = KronosSpec(
            enabled=True,
            model=model,
            max_context=min(MAX_CONTEXT.get(model, 512),
                            max(int(self.params.get("context", 240)), 32)),
            pred_len=int(self.params.get("pred_len", 12)),
            sample_count=int(self.params.get("sample_count", 1)),
        )
        self._fc = kr.KronosForecaster(spec)
        return self._fc

    def forecast(self, bars: pd.DataFrame, pred_len: int) -> pd.DataFrame:
        try:
            return self._forecaster().forecast(bars, pred_len=int(pred_len))
        except ForecasterUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            self._error = f"{type(e).__name__}: {e}"
            raise ForecasterUnavailable(f"Kronos 预测失败：{self._error}") from e


#: 全部可选模型（key → 构造器）。顺序 = UI 展示顺序。
MODEL_FACTORIES: dict[str, Any] = {
    "kronos": KronosAdapter,
    **{k: v for k, v in STAT_FORECASTERS.items()},
    **{k: v for k, v in EXTERNAL_FORECASTERS.items()},
}

#: 默认权重：Kronos 最贴合金融但最慢，统计基线做锚，外部模型默认 0（未装）
DEFAULT_WEIGHTS: dict[str, float] = {
    "kronos": 1.0,
    "random_walk": 0.5,
    "ewma": 1.0,
    "momentum": 0.8,
    "drift": 0.5,
    "reversion": 0.5,
    "chronos": 0.0,
    "timesfm": 0.0,
    "moirai": 0.0,
}


@dataclass
class EnsembleSpec:
    """一次集成预测的配置。"""

    # ---- 权重 ----
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # ---- 硬性门控 ----
    min_abs_return: float = 0.0          # 预测幅度绝对下限（0 = 不限）
    min_return_sigma: float = 0.3        # 预测幅度 ≥ N × 波动率
    min_agreement: float = 0.6           # 同向模型权重占比
    max_dispersion: float = 2.0          # 模型间标准差 ≤ N × 波动率
    min_models: int = 2                  # 有效模型数下限
    # ---- 仓位 ----
    position: float = 1.0
    allow_short: bool = False
    # ---- 运行参数 ----
    pred_len: int = 12
    context: int = 240
    tz: str = "Asia/Shanghai"

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "min_abs_return": self.min_abs_return,
            "min_return_sigma": self.min_return_sigma,
            "min_agreement": self.min_agreement,
            "max_dispersion": self.max_dispersion,
            "min_models": self.min_models,
            "position": self.position,
            "allow_short": self.allow_short,
            "pred_len": self.pred_len,
            "context": self.context,
            "tz": self.tz,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "EnsembleSpec":
        """容错构造：未知键忽略、类型不对回退默认值。"""
        raw = raw or {}
        spec = cls()
        w = raw.get("weights")
        if isinstance(w, dict):
            # 只留认识的模型：配置手改出拼错的 key 时，若不清理，
            # 集成层每轮都会为它记一条「构造失败」的票，噪音大于信息。
            spec.weights = {str(k): float(v) for k, v in w.items()
                            if k in MODEL_FACTORIES and isinstance(v, (int, float))}
        for name in ("min_abs_return", "min_return_sigma", "min_agreement",
                     "max_dispersion", "position"):
            if isinstance(raw.get(name), (int, float)):
                setattr(spec, name, float(raw[name]))
        for name in ("min_models", "pred_len", "context"):
            if isinstance(raw.get(name), (int, float)):
                setattr(spec, name, int(raw[name]))
        if isinstance(raw.get("allow_short"), bool):
            spec.allow_short = raw["allow_short"]
        if isinstance(raw.get("tz"), str):
            spec.tz = raw["tz"]
        return spec


def list_models() -> list[dict[str, Any]]:
    """所有模型的可用性（**不触发任何权重加载**）。"""
    out = []
    for key, factory in MODEL_FACTORIES.items():
        try:
            out.append(factory().status())
        except Exception as e:  # noqa: BLE001 - 单个模型探测失败不影响其他
            out.append({"key": key, "label": key, "available": False,
                        "missing": [], "error": f"{type(e).__name__}: {e}",
                        "kind": "unknown"})
    return out


def _build(key: str, spec: EnsembleSpec) -> BaseForecaster | None:
    factory = MODEL_FACTORIES.get(key)
    if factory is None:
        return None
    params: dict[str, Any] = {}
    if key == "kronos":
        params = {"context": spec.context, "pred_len": spec.pred_len}
    elif key in STAT_FORECASTERS:
        params = {}
    else:
        params = {"max_context": spec.context, "pred_len": spec.pred_len, "tz": spec.tz}
    try:
        return factory(params)
    except Exception as e:  # noqa: BLE001
        log.warning("构造预测器 %s 失败：%s", key, e)
        return None


@dataclass
class ModelVote:
    key: str
    label: str
    weight: float
    expected_return: float | None
    ok: bool
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "weight": self.weight,
                "expected_return": self.expected_return, "ok": self.ok,
                "error": self.error}


@dataclass
class EnsembleResult:
    """一次集成预测的完整结果（含每个模型的票与门控判定）。"""

    expected_return: float | None
    target_weight: float
    confidence: float
    sigma: float
    reason: str
    votes: list[ModelVote]
    agreement: float
    dispersion: float
    gates: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_return": self.expected_return,
            "target_weight": self.target_weight,
            "confidence": self.confidence,
            "sigma": self.sigma,
            "reason": self.reason,
            "agreement": self.agreement,
            "dispersion": self.dispersion,
            "gates": self.gates,
            "votes": [v.as_dict() for v in self.votes],
        }


class EnsembleForecaster:
    """加权集成 + 硬性门控。线程内复用模型实例（避免每次重建）。"""

    def __init__(self, spec: EnsembleSpec | None = None):
        self.spec = spec or EnsembleSpec()
        self._instances: dict[str, BaseForecaster] = {}

    # ---- 模型池 ----
    def _get(self, key: str) -> BaseForecaster | None:
        if key in self._instances:
            return self._instances[key]
        inst = _build(key, self.spec)
        if inst is not None:
            self._instances[key] = inst
        return inst

    # ---- 核心 ----
    def predict(self, bars: pd.DataFrame, pred_len: int | None = None) -> EnsembleResult:
        spec = self.spec
        n = int(spec.pred_len if pred_len is None else pred_len)
        sigma = return_sigma(bars["close"]) if "close" in bars.columns else 0.0

        votes: list[ModelVote] = []
        for key, weight in spec.weights.items():
            w = float(weight)
            if w <= 0:
                continue
            inst = self._get(key)
            if inst is None:
                votes.append(ModelVote(key, key, w, None, False, "构造失败"))
                continue
            label = getattr(inst, "label", key)
            try:
                ret = inst.expected_return(bars, n)
            except ForecasterUnavailable as e:
                votes.append(ModelVote(key, label, w, None, False, str(e)))
                continue
            except Exception as e:  # noqa: BLE001 - 单模型失败不该拖垮集成
                log.warning("集成模型 %s 预测失败：%s", key, e)
                votes.append(ModelVote(key, label, w, None, False, f"{type(e).__name__}: {e}"))
                continue
            if ret is None or not np.isfinite(ret):
                votes.append(ModelVote(key, label, w, None, False, "未返回有效预测"))
                continue
            votes.append(ModelVote(key, label, w, float(ret), True))

        return self._combine(votes, sigma)

    def _combine(self, votes: list[ModelVote], sigma: float) -> EnsembleResult:
        spec = self.spec
        ok = [v for v in votes if v.ok and v.expected_return is not None]
        total_w = sum(v.weight for v in ok)

        gates: dict[str, Any] = {
            "min_models": {"need": spec.min_models, "got": len(ok),
                           "pass": len(ok) >= spec.min_models},
        }

        if not ok or total_w <= 0:
            gates["override"] = "无有效模型预测"
            return EnsembleResult(None, 0.0, 0.0, sigma, "无有效模型预测", votes,
                                  0.0, 0.0, gates)

        rets = np.array([v.expected_return for v in ok], dtype=float)
        ws = np.array([v.weight for v in ok], dtype=float)
        combined = float(np.dot(rets, ws) / ws.sum())

        # 同向占比（按权重）：以合成结果的符号为准
        sign = np.sign(combined) if combined != 0 else 0.0
        same = [v.weight for v in ok if np.sign(v.expected_return) == sign and sign != 0]
        agreement = float(sum(same) / total_w) if total_w > 0 else 0.0
        dispersion = float(np.sqrt(np.dot(ws, (rets - combined) ** 2) / ws.sum()))

        # ---- 硬性门控 ----
        need = spec.min_return_sigma * sigma
        gates["min_abs_return"] = {
            "need": spec.min_abs_return, "got": abs(combined),
            "pass": abs(combined) >= spec.min_abs_return,
        }
        gates["min_return_sigma"] = {
            "need": need, "got": abs(combined), "sigma": sigma,
            "pass": abs(combined) >= need,
        }
        gates["min_agreement"] = {
            "need": spec.min_agreement, "got": agreement,
            "pass": agreement >= spec.min_agreement,
        }
        gates["max_dispersion"] = {
            "need": spec.max_dispersion * sigma, "got": dispersion, "sigma": sigma,
            "pass": dispersion <= spec.max_dispersion * sigma if sigma > 0 else True,
        }
        gates["direction"] = {
            "pass": spec.allow_short or combined > 0,
            "allow_short": spec.allow_short,
        }

        blocked = [k for k, v in gates.items()
                   if isinstance(v, dict) and v.get("pass") is False]
        pct = combined * 100
        thr = need * 100

        if blocked:
            if blocked == ["direction"]:
                # 单独给这条门控一句人话：不是「没通过」，而是「方向不允许」
                return EnsembleResult(
                    combined, 0.0, 0.0, sigma,
                    f"集成预测 {pct:+.2f}%（看空），但策略只做多 → 空仓",
                    votes, agreement, dispersion, gates)
            why = "、".join(blocked)
            return EnsembleResult(
                combined, 0.0, 0.0, sigma,
                f"集成预测 {pct:+.2f}%，但未通过硬性门控（{why}）→ 空仓",
                votes, agreement, dispersion, gates)

        w = spec.position if combined > 0 else -spec.position
        conf = float(np.clip(abs(combined) / max(thr, abs(spec.min_abs_return), 1e-9) / 3.0,
                             0.0, 1.0))
        side = "做多" if w > 0 else "做空"
        return EnsembleResult(
            combined, w, conf, sigma,
            f"集成预测 {pct:+.2f}%（阈值 {thr:+.2f}%，一致度 {agreement:.0%}，"
            f"分歧 {dispersion * 100:.2f}%）→ {side}",
            votes, agreement, dispersion, gates)


# ====================== 自动权重（按历史 IC） ======================

def auto_weights(bars: pd.DataFrame, pred_len: int = 12, *,
                 eval_bars: int = 240, step: int = 20,
                 context: int = 240, keys: list[str] | None = None,
                 tz: str = "Asia/Shanghai") -> dict[str, Any]:
    """用滚动回看估每个模型的历史表现，给出建议权重。

    做法：在最近 ``eval_bars`` 根上每隔 ``step`` 根取一个评估点，
    用「截至该点的数据」预测 ``pred_len`` 根后的收益，与实际走势比对：

    - ``hit``：方向命中率（预测符号 == 实际符号）
    - ``ic``：预测值与实际值的 Pearson 相关

    建议权重 = ``max(0, ic) × (2·hit - 1) × 2``，再归一化到合计 1。
    **负相关/低于 50% 命中直接给 0** —— 一个反向指标不该被「加权融合」，
    要加权也该是负权重，而负权重会带来一堆稳定性问题，不如弃用。

    这不是严格的样本外评估（同一段数据既看又评），只能当「初始猜测」，
    真正的验证要靠 :func:`quant.engine.backtest` 跑完整回测。
    """
    keys = keys or [k for k in MODEL_FACTORIES
                    if k in ("kronos", "ewma", "momentum", "drift", "reversion")]
    spec = EnsembleSpec(pred_len=pred_len, context=context, tz=tz,
                        weights={k: 1.0 for k in keys})
    insts = {k: _build(k, spec) for k in keys}

    if not isinstance(bars, pd.DataFrame) or len(bars) < context + pred_len + 10:
        return {"ok": False, "error": f"K 线不足：需 > {context + pred_len + 10} 根",
                "weights": {}, "detail": {}}

    start = max(context, len(bars) - eval_bars)
    points = list(range(start, len(bars) - pred_len, max(1, step)))
    if len(points) < 5:
        return {"ok": False, "error": f"可评估点太少（{len(points)} 个），请提供更长历史",
                "weights": {}, "detail": {}}

    detail: dict[str, dict[str, Any]] = {}
    for key, inst in insts.items():
        if inst is None:
            continue
        preds: list[float] = []
        actuals: list[float] = []
        for i in points:
            window = bars.iloc[: i + 1]
            try:
                ret = inst.expected_return(window, pred_len)
            except Exception:  # noqa: BLE001
                ret = None
            if ret is None or not np.isfinite(ret):
                continue
            last = float(window["close"].iloc[-1])
            end = float(bars["close"].iloc[min(i + pred_len, len(bars) - 1)])
            if last <= 0:
                continue
            preds.append(float(ret))
            actuals.append(end / last - 1.0)

        if len(preds) < 5:
            detail[key] = {"ok": False, "error": f"有效样本仅 {len(preds)} 个"}
            continue
        p = np.asarray(preds, dtype=float)
        a = np.asarray(actuals, dtype=float)
        hit = float(np.mean(np.sign(p) == np.sign(a)))
        if np.std(p) < 1e-12 or np.std(a) < 1e-12:
            ic = 0.0
        else:
            ic = float(np.corrcoef(p, a)[0, 1])
            if not np.isfinite(ic):
                ic = 0.0
        score = max(0.0, ic) * max(0.0, 2.0 * hit - 1.0) * 2.0
        detail[key] = {"ok": True, "samples": len(p), "hit": hit,
                       "ic": ic, "score": score}

    scores = {k: v["score"] for k, v in detail.items() if v.get("ok")}
    total = sum(scores.values())
    if total <= 0:
        return {"ok": False, "error": "所有模型在回看区间都没有正贡献，建议保持手调权重",
                "weights": {}, "detail": detail}
    weights = {k: round(v / total, 4) for k, v in scores.items()}
    return {"ok": True, "weights": weights, "detail": detail,
            "eval_points": len(points), "eval_bars": eval_bars, "pred_len": pred_len}
