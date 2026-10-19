"""融合策略：把「大模型集成预测」和「因子策略」合成一个仓位决策。

## 为什么需要它

系统里有两类完全不同的信号来源：

- **大模型层**（``ensemble_forecast``）：Kronos / Chronos / 统计基线加权
  投票，给出「未来 N 根预期涨跌幅」和一套硬性门控；
- **因子层**（``dual_ma`` / ``macd`` / ``rsi``…）：经典技术指标状态机，
  给出离散的 ``target_weight``（-1 / 0 / +1）。

单独用任一层都有盲区：模型预测的是**方向与幅度**但不管入场时机；
因子给的是**明确的持仓状态**但很容易被震荡反复打脸。融合策略让用户
自由配置「哪些模型、权重多少、怎么和因子结合」，由这一层把两者缝起来。

## 四种融合方式

============  ==========================================================
方式          语义
============  ==========================================================
``blend``     加权平均：``w = model_share·w_model + factor_share·w_factor``，
               最直观，两侧都持续影响仓位
``model_gate`` 模型是闸门：模型给不出信号（门控不过 / 空仓）时因子
               一律作废，模型开多时才按因子权重决定持仓
``factor_gate`` 因子是闸门：因子空仓时模型说了也不算，因子持仓时
               按模型权重决定持仓
``vote``       投票制：两侧按各自权重投票（-1/0/+1 加权），谁的
               合计权重占优就听谁的，平票空仓
============  ==========================================================

四种方式的公共前提：**任一侧权重为 0 就自动退出融合，退回单侧策略**，
所以「只想用模型」和「只想用因子」都不需要额外开关。

## 为什么权重只影响方向决策、不影响硬门控

大模型侧的硬性门控（幅度下限、一致性、分歧度）是「模型们自己都
没搞懂就别下注」的保险，和用户想给因子多少权重无关。因此
``model_weights`` 只决定**模型之间**怎么加权，``factor_weights``
只决定**因子之间**怎么加权，而 ``model_share`` / ``factor_share``
才决定**两侧之间**怎么分话语权 —— 三者互不覆盖，改一个不会
悄悄改变另一个的含义。

## 成本

融合策略在每个预测点要跑 N 个模型 + M 个因子，比单侧都慢。
默认只启用 2 个统计基线（``ewma`` + ``momentum``）+ 2 个因子
（``dual_ma`` + ``macd``），毫秒级；把 kronos 权重调高才会变慢。
"""
from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.strategies.base import BaseStrategy, StrategyContext
from quant.strategies.registry import register

log = logging.getLogger(__name__)

#: 融合配置在 kv_store 里的键
CONFIG_KEY = "ml.fusion"

#: 融合方式 → 中文说明（UI 下拉与策略 reason 共用）
FUSION_MODES: dict[str, str] = {
    "blend": "加权混合（两侧同时影响仓位）",
    "model_gate": "模型闸门（模型不开门，因子不作数）",
    "factor_gate": "因子闸门（因子不持仓，模型不作数）",
    "vote": "投票制（两侧加权投票，占着者胜）",
}

#: 默认启用的大模型（权重 > 0）。Kronos 默认关：慢，且多数机器没装依赖。
DEFAULT_MODEL_WEIGHTS: dict[str, float] = {
    "kronos": 0.0,
    "random_walk": 0.0,
    "ewma": 1.0,
    "momentum": 0.8,
    "drift": 0.0,
    "reversion": 0.0,
    "chronos": 0.0,
    "timesfm": 0.0,
    "moirai": 0.0,
}

#: 默认启用的因子策略（权重 > 0）
DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "dual_ma": 1.0,
    "macd": 0.6,
    "rsi": 0.0,
    "bollinger": 0.0,
    "momentum": 0.0,
    "mean_reversion": 0.0,
    "turtle": 0.0,
    "grid": 0.0,
}


def _default_model_weights_json() -> str:
    return json.dumps(DEFAULT_MODEL_WEIGHTS, ensure_ascii=False)


def _default_factor_weights_json() -> str:
    return json.dumps(DEFAULT_FACTOR_WEIGHTS, ensure_ascii=False)


def _parse_weights(raw: str | None, fallback: dict[str, float]) -> dict[str, float]:
    """容错解析权重 JSON：解析失败 / 非 dict / 值非数字都退回默认。"""
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        log.warning("融合权重 JSON 解析失败，改用默认权重：%s", e)
        return dict(fallback)
    if not isinstance(data, dict):
        return dict(fallback)
    out: dict[str, float] = {}
    for k, v in data.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        fv = float(v)
        if np.isfinite(fv):
            out[str(k)] = fv
    return out


def _kv_config() -> dict[str, Any]:
    """读 kv_store 里的融合配置；任何失败都返回空 dict（走默认值）。"""
    try:
        from quant.core.config import AppConfig
        from quant.storage.repositories import KVRepo

        saved = KVRepo(AppConfig.load().effective_db_path).get(CONFIG_KEY)
        return saved if isinstance(saved, dict) else {}
    except Exception:  # noqa: BLE001 - 读配置失败不应让策略崩，退回默认
        log.debug("读取融合配置失败，使用默认值", exc_info=True)
        return {}


def _factor_signals(bars: pd.DataFrame, ctx: StrategyContext,
                    factor_keys: list[str]) -> dict[str, float]:
    """跑每个因子策略，取其在最后一根的 target_weight。

    用最后一根而不是整段信号：融合策略按预测点推进，只需要「当前
    这一下因子认为该持多少仓」。单个因子失败只记日志并跳过——
    一个坏因子不该让整个融合策略失效。
    """
    from quant.strategies import registry

    registry.ensure_loaded()
    out: dict[str, float] = {}
    for key in factor_keys:
        try:
            strat = registry.create(key)
            df = strat.generate_signals(bars, ctx)
            if df is None or df.empty or "target_weight" not in df.columns:
                continue
            val = df["target_weight"].iloc[-1]
            if val is None or pd.isna(val):
                continue
            out[key] = float(val)
        except Exception as e:  # noqa: BLE001
            log.warning("因子策略 %s 取信号失败：%s", key, e)
    return out


def _blend(model_w: float, factor_w: float,
           model_share: float, factor_share: float) -> float:
    """两侧各自归一后按份额加权。"""
    total = model_share + factor_share
    if total <= 0:
        return 0.0
    return (model_share * model_w + factor_share * factor_w) / total


def _vote(model_w: float, factor_w: float,
          model_share: float, factor_share: float) -> float:
    """投票制：把两侧的连续信号压成 -1/0/+1 三档，再按份额投票。"""
    def side(x: float) -> int:
        if x > 1e-9:
            return 1
        if x < -1e-9:
            return -1
        return 0

    up = model_share * (1 if side(model_w) > 0 else 0) \
        + factor_share * (1 if side(factor_w) > 0 else 0)
    down = model_share * (1 if side(model_w) < 0 else 0) \
        + factor_share * (1 if side(factor_w) < 0 else 0)
    if up > down + 1e-9:
        return 1.0
    if down > up + 1e-9:
        return -1.0
    return 0.0


def combine_signals(model_w: float, factor_w: float, *, mode: str,
                    model_share: float, factor_share: float,
                    allow_short: bool, position: float) -> tuple[float, str]:
    """按融合方式合成最终 target_weight。返回 (权重, 说明)。

    这是本策略唯一需要被单测覆盖的纯函数 —— 四种方式的语义差异
    全在这里，错了会让整个策略的语义反掉。
    """
    if not allow_short:
        factor_w = max(0.0, factor_w)
        model_w = max(0.0, model_w)

    if mode == "model_gate":
        # 模型是闸门：模型不开门 → 因子一律作废
        if abs(model_w) < 1e-9:
            return 0.0, "模型闸门未开（模型空仓/门控未过）→ 空仓"
        return position * float(np.sign(model_w)), \
            f"模型闸门开启（{model_w:+.2f}）→ 按模型方向{'做多' if model_w > 0 else '做空'}"
    if mode == "factor_gate":
        if abs(factor_w) < 1e-9:
            return 0.0, "因子闸门未开（因子空仓）→ 空仓"
        return position * float(np.sign(factor_w)), \
            f"因子闸门开启（{factor_w:+.2f}）→ 按因子方向{'做多' if factor_w > 0 else '做空'}"
    if mode == "vote":
        v = _vote(model_w, factor_w, model_share, factor_share)
        if abs(v) < 1e-9:
            return 0.0, f"两侧投票平手（模型 {model_w:+.2f} / 因子 {factor_w:+.2f}）→ 空仓"
        return position * v, f"投票结果 {'做多' if v > 0 else '做空'}（模型 {model_w:+.2f} / 因子 {factor_w:+.2f}）"

    # blend（默认）
    w = _blend(model_w, factor_w, model_share, factor_share)
    if abs(w) < 1e-9:
        return 0.0, f"两侧信号相互抵消（模型 {model_w:+.2f} / 因子 {factor_w:+.2f}）→ 空仓"
    return float(np.clip(w, -position, position)), \
        f"加权混合 {w:+.2f}（模型 {model_w:+.2f}×{model_share:.0%} + 因子 {factor_w:+.2f}×{factor_share:.0%}）"


@register
class FusionStrategy(BaseStrategy):
    key: ClassVar[str] = "fusion"
    name: ClassVar[str] = "模型因子融合策略"
    display_name: ClassVar[str] = "模型因子融合策略 (Model × Factor Fusion)"
    description: ClassVar[str] = ("把大模型集成预测与多个因子策略自由融合："
                                  "可选模型与权重、可选因子与权重、四种结合方式")

    class Params(BaseModel):
        # ---- 大模型侧 ----
        model_weights_json: str = Field(
            default_factory=_default_model_weights_json,
            description='大模型权重 JSON，如 {"ewma":1.0,"momentum":0.8,"kronos":1.0}；'
                        '权重 0 = 不启用该模型')
        # ---- 因子侧 ----
        factor_weights_json: str = Field(
            default_factory=_default_factor_weights_json,
            description='因子策略权重 JSON，如 {"dual_ma":1.0,"macd":0.6}；'
                        '权重 0 = 不启用该策略')
        # ---- 融合方式 ----
        fusion_mode: str = Field(
            "blend", description="两侧如何结合：加权混合 / 模型闸门 / 因子闸门 / 投票制")
        model_share: float = Field(
            0.5, ge=0.0, le=1.0, description="大模型侧的话语权占比（因子侧为 1 减此值）")
        confidence_weighted: bool = Field(
            True, description="用模型置信度缩放模型侧信号（模型自己都没把握时自动降权）")
        # ---- 模型侧硬性门控（与 ensemble_forecast 同一套） ----
        min_return_sigma: float = Field(
            0.3, ge=0.0, le=10.0,
            description="模型预测幅度至少是近 N 根波动率的几倍才采纳（按波动率归一，跨周期通用）")
        min_abs_return: float = Field(
            0.0, ge=0.0, le=0.5, description="模型预测幅度的绝对下限（0 = 不限）")
        min_agreement: float = Field(
            0.6, ge=0.0, le=1.0, description="同向模型的权重占比下限，低于此值视为分歧大 → 模型侧弃权")
        max_dispersion: float = Field(
            2.0, ge=0.0, le=20.0, description="模型间预测标准差不得超过波动率的几倍")
        min_models: int = Field(
            2, ge=1, le=9, description="至少几个模型给出有效预测才采纳模型侧结论")
        # ---- 仓位与运行 ----
        position: float = Field(1.0, ge=0.0, le=1.0, description="单侧满信号时的目标仓位")
        allow_short: bool = Field(False, description="允许做空（否则只做多）")
        context: int = Field(240, ge=32, le=2048, description="喂给模型的上下文 K 线根数")
        pred_len: int = Field(12, ge=1, le=120, description="预测未来多少根 K 线")
        rebalance_every: int = Field(
            20, ge=1, le=500, description="每多少根 K 线重新做一次「模型预测 + 因子取信号」")
        max_forecasts: int = Field(
            150, ge=1, le=5000, description="单次回测的预测次数上限（超出自动加大步长）")

    # ==================== 配置装载 ====================
    #: 参数 → 配置键的映射。**只有参数偏离默认值时才覆盖 kv_store**，
    #: 否则「用户没动策略参数」会被误判成「用户显式要了默认值」，
    #: 融合配置页里调好的全局设置就被悄悄盖掉了。
    _PARAM_TO_CONFIG = (
        ("fusion_mode", "fusion_mode"),
        ("model_share", "model_share"),
        ("confidence_weighted", "confidence_weighted"),
        ("min_abs_return", "min_abs_return"),
        ("min_return_sigma", "min_return_sigma"),
        ("min_agreement", "min_agreement"),
        ("max_dispersion", "max_dispersion"),
        ("min_models", "min_models"),
        ("position", "position"),
        ("allow_short", "allow_short"),
        ("pred_len", "pred_len"),
        ("context", "context"),
    )

    def _config(self) -> dict[str, Any]:
        """融合配置：kv_store 全局配置为底，**偏离默认值的策略参数**覆盖其上。

        分三类：

        - ``model_weights`` / ``factor_weights``：参数里有启用的模型/因子
          就用参数的，否则用全局配置的（都没配则用默认权重表）；
        - ``fusion_mode`` 等 12 个标量项：参数值 != 字段默认值才算「显式设置」；
        - ``factor_share``：由 ``model_share`` 推导，不单独存。

        这样「策略管理页改参数」和「融合配置页改全局」两条路径不会互相覆盖。
        """
        saved = _kv_config()
        merged = dict(saved) if isinstance(saved, dict) else {}
        p = self.params
        defaults = type(self).Params()

        try:
            mw = _parse_weights(p.model_weights_json, DEFAULT_MODEL_WEIGHTS)
        except Exception:  # noqa: BLE001
            mw = dict(DEFAULT_MODEL_WEIGHTS)
        try:
            fw = _parse_weights(p.factor_weights_json, DEFAULT_FACTOR_WEIGHTS)
        except Exception:  # noqa: BLE001
            fw = dict(DEFAULT_FACTOR_WEIGHTS)
        if any(v > 0 for v in mw.values()):
            merged["model_weights"] = mw
        if any(v > 0 for v in fw.values()):
            merged["factor_weights"] = fw
        merged.setdefault("model_weights", dict(DEFAULT_MODEL_WEIGHTS))
        merged.setdefault("factor_weights", dict(DEFAULT_FACTOR_WEIGHTS))

        for attr, key in self._PARAM_TO_CONFIG:
            val = getattr(p, attr)
            if val != getattr(defaults, attr):
                merged[key] = val
        merged.setdefault("fusion_mode", defaults.fusion_mode)
        merged.setdefault("model_share", defaults.model_share)
        merged.setdefault("confidence_weighted", defaults.confidence_weighted)
        merged.setdefault("min_abs_return", defaults.min_abs_return)
        merged.setdefault("min_return_sigma", defaults.min_return_sigma)
        merged.setdefault("min_agreement", defaults.min_agreement)
        merged.setdefault("max_dispersion", defaults.max_dispersion)
        merged.setdefault("min_models", defaults.min_models)
        merged.setdefault("position", defaults.position)
        merged.setdefault("allow_short", defaults.allow_short)
        merged.setdefault("pred_len", defaults.pred_len)
        merged.setdefault("context", defaults.context)

        merged["model_share"] = float(merged["model_share"])
        merged["factor_share"] = float(1.0 - merged["model_share"])
        return merged

    def _active_model_weights(self) -> dict[str, float]:
        mw = self._config().get("model_weights") or {}
        return {str(k): float(v) for k, v in mw.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) > 0}

    def _active_factor_weights(self) -> dict[str, float]:
        fw = self._config().get("factor_weights") or {}
        return {str(k): float(v) for k, v in fw.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) > 0}

    # ==================== 模型侧 ====================
    def _ensure_ensemble(self):
        from alpha.models.ensemble import EnsembleForecaster, EnsembleSpec

        cached = getattr(self, "_ens_cache", None)
        if cached is not None:
            return cached[0]

        cfg = self._config()
        spec = EnsembleSpec(
            weights=self._active_model_weights(),
            min_abs_return=cfg["min_abs_return"],
            min_return_sigma=cfg["min_return_sigma"],
            min_agreement=cfg["min_agreement"],
            max_dispersion=cfg["max_dispersion"],
            min_models=cfg["min_models"],
            position=cfg["position"],
            allow_short=cfg["allow_short"],
            pred_len=cfg["pred_len"],
            context=cfg["context"],
        )
        ens = EnsembleForecaster(spec)
        self._ens_cache = (ens, spec)
        return ens

    def _model_signal(self, window: pd.DataFrame) -> tuple[float, str]:
        """跑一次集成预测，返回 (归一到 [-1,1] 的模型侧信号, reason)。

        门控未过时 EnsembleResult.target_weight 已是 0，reason 说明原因，
        这里保留原文 —— 用户需要知道「模型为什么不说话」。
        """
        try:
            ens = self._ensure_ensemble()
            res = ens.predict(window, pred_len=self.params.pred_len)
        except Exception as e:  # noqa: BLE001 - 模型侧失败退回「模型不说话」
            log.warning("融合策略模型侧预测失败：%s", e)
            return 0.0, f"模型侧不可用（{type(e).__name__}: {e}）"
        if res is None:
            return 0.0, "模型侧不可用"
        sig = float(res.target_weight)
        reason = str(res.reason)
        if self.params.confidence_weighted and sig != 0.0:
            conf = float(np.clip(res.confidence, 0.0, 1.0))
            sig *= conf
            if conf < 0.34:
                reason += f"（置信度 {conf:.0%}，已按置信度降权）"
        return sig, reason

    # ==================== 因子侧 ====================
    def _factor_signal(self, bars: pd.DataFrame, ctx: StrategyContext) -> tuple[float, str]:
        """跑所有启用的因子策略，按权重合成归一到 [-1,1] 的因子侧信号。"""
        weights = self._active_factor_weights()
        if not weights:
            return 0.0, "未启用任何因子策略"
        sigs = _factor_signals(bars, ctx, list(weights.keys()))
        if not sigs:
            return 0.0, "因子策略均未返回有效信号"
        total = sum(weights[k] for k in sigs if weights.get(k, 0.0) > 0)
        if total <= 0:
            return 0.0, "因子策略权重异常"
        acc = sum(weights[k] * sigs[k] for k in sigs)
        sig = float(np.clip(acc / total, -1.0, 1.0))
        parts = "，".join(f"{k} {sigs[k]:+.2f}×{weights[k]:g}" for k in sorted(sigs))
        return sig, f"因子合成 {sig:+.2f}（{parts}）"

    # ==================== 融合 ====================
    def _decide(self, bars: pd.DataFrame, ctx: StrategyContext) -> tuple[float, str, float]:
        """一次完整融合决策。返回 (target_weight, reason, confidence)。"""
        cfg = self._config()
        model_share = float(cfg["model_share"])
        factor_share = float(cfg["factor_share"])
        if model_share + factor_share <= 0:
            return 0.0, "两侧话语权均为 0，无法决策", 0.0

        model_w, model_reason = self._model_signal(bars)
        factor_w, factor_reason = self._factor_signal(bars, ctx)

        weight, note = combine_signals(
            model_w, factor_w, mode=str(cfg["fusion_mode"]),
            model_share=model_share, factor_share=factor_share,
            allow_short=bool(cfg["allow_short"]),
            position=float(cfg["position"]),
        )
        # 置信度：两侧都有信号时取较小者（保守）；单侧时用该侧绝对值
        conf = 0.0
        if model_w != 0.0 and factor_w != 0.0:
            conf = min(abs(model_w), abs(factor_w))
        elif model_w != 0.0:
            conf = abs(model_w)
        elif factor_w != 0.0:
            conf = abs(factor_w)
        conf = float(np.clip(conf, 0.0, 1.0))
        reason = f"{note}｜模型：{model_reason}｜因子：{factor_reason}"
        return weight, reason, conf

    # ==================== BaseStrategy 契约 ====================
    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        if bars is None or bars.empty:
            return pd.DataFrame({"target_weight": [], "reason": [], "confidence": []})

        n = len(bars)
        p = self.params
        warmup = max(p.context, 2)
        if n <= warmup:
            return pd.DataFrame({
                "target_weight": pd.Series(0.0, index=bars.index),
                "reason": f"K 线不足（需 > {warmup} 根，实际 {n} 根）",
                "confidence": 0.0,
            })

        start = warmup - 1
        span = n - start
        step = max(1, int(p.rebalance_every))
        if span / step > p.max_forecasts:
            step = int(np.ceil(span / p.max_forecasts))
            log.info("融合策略预测次数超过上限 %d，步长自动调整为 %d 根", p.max_forecasts, step)
        points = list(range(start, n, step))

        target = pd.Series(np.nan, index=bars.index)
        reason = pd.Series("", index=bars.index, dtype=object)
        conf = pd.Series(np.nan, index=bars.index)

        for i in points:
            window = bars.iloc[: i + 1]
            w, r, c = self._decide(window, ctx)
            target.iloc[i] = w
            reason.iloc[i] = r
            conf.iloc[i] = c

        target = target.ffill().fillna(0.0).astype(float)
        conf = conf.ffill().fillna(0.0).astype(float)
        reason = reason.replace("", np.nan).ffill().fillna("等待融合决策")
        reason.iloc[:start] = "预热期（K 线不足）"

        return pd.DataFrame({"target_weight": target, "reason": reason,
                             "confidence": conf})

    # ---- 模拟盘：每根新 bar 只融合一次 ----
    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> tuple[float, str, float]:
        history = ctx.history
        if history is None or len(history) < self.params.context:
            return 0.0, f"预热中（需 {self.params.context} 根）", 0.0

        stamp = history.index[-1]
        cache = getattr(self, "_bar_cache", None)
        if cache is not None and cache[0] == stamp:
            return cache[1]

        window = history.iloc[-self.params.context:]
        out = self._decide(window, ctx)
        self._bar_cache = (stamp, out)
        return out
