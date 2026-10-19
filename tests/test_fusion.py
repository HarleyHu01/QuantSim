"""融合策略（大模型集成 × 因子策略）的测试。

同其它模型测试的原则：**不加载任何真实权重**。融合策略的价值全在
「两侧信号怎么缝起来」，所以测试重点在四种融合方式的语义边界、
权重解析容错、以及单侧失效时的降级行为。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alpha.models.strategies.fusion import (CONFIG_KEY, DEFAULT_FACTOR_WEIGHTS,
                                            DEFAULT_MODEL_WEIGHTS, FusionStrategy,
                                            _parse_weights, combine_signals)
from quant.strategies.base import StrategyContext
from quant.strategies.registry import ensure_loaded


# ---------------------------------------------------------------- 夹具

def _bars(n: int = 300, freq: str = "5min") -> pd.DataFrame:
    """带趋势 + 正弦扰动的行情，保证因子能给出非零信号。"""
    idx = pd.date_range("2026-10-12 01:30", periods=n, freq=freq, tz="UTC")
    base = 100.0 + np.arange(n, dtype=float) * 0.05
    base = base * (1.0 + 0.004 * np.sin(np.arange(n) / 5.0))
    close = pd.Series(base, index=idx)
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(n, 1000.0),
    }, index=idx)


def _ctx(bars: pd.DataFrame) -> StrategyContext:
    return StrategyContext(symbol="test.SZ", timeframe="5m", history=bars)


def _params(**kw) -> dict:
    """策略参数基座：只开便宜模型与两个因子，测试跑得快。"""
    p = {
        "model_weights_json": json.dumps({"ewma": 1.0, "momentum": 0.5}),
        "factor_weights_json": json.dumps({"dual_ma": 1.0, "macd": 0.5}),
        "fusion_mode": "blend",
        "model_share": 0.5,
        "context": 60,
        "pred_len": 6,
        "rebalance_every": 40,
        "min_models": 1,
        "min_agreement": 0.0,
    }
    p.update(kw)
    return p


# ---------------------------------------------------------------- 四种融合方式的语义

def test_blend_is_share_weighted_average():
    """blend：两侧各自归一后按份额加权。"""
    w, note = combine_signals(0.8, 0.4, mode="blend", model_share=0.5,
                              factor_share=0.5, allow_short=False, position=1.0)
    assert w == pytest.approx(0.6)
    assert "加权混合" in note


def test_blend_share_ratio_matters():
    """同一对信号，份额不同结果必须不同 —— 否则「自由配置权重」是假的。"""
    a, _ = combine_signals(1.0, 0.0, mode="blend", model_share=0.9,
                           factor_share=0.1, allow_short=False, position=1.0)
    b, _ = combine_signals(1.0, 0.0, mode="blend", model_share=0.1,
                           factor_share=0.9, allow_short=False, position=1.0)
    assert a == pytest.approx(0.9)
    assert b == pytest.approx(0.1)


def test_model_gate_blocks_factor_when_model_silent():
    """模型闸门：模型空仓时因子说了不算，哪怕因子满仓。"""
    w, note = combine_signals(0.0, 1.0, mode="model_gate", model_share=0.5,
                              factor_share=0.5, allow_short=False, position=1.0)
    assert w == 0.0
    assert "模型闸门未开" in note


def test_model_gate_passes_model_direction():
    """模型闸门开启时按模型方向满仓（不看因子，这是「闸门」的语义）。"""
    w, _ = combine_signals(0.3, -1.0, mode="model_gate", model_share=0.5,
                           factor_share=0.5, allow_short=False, position=1.0)
    assert w == pytest.approx(1.0)      # 模型 +0.3 → 做多满仓
    assert "做多" in _


def test_factor_gate_blocks_model_when_factor_flat():
    w, note = combine_signals(1.0, 0.0, mode="factor_gate", model_share=0.5,
                              factor_share=0.5, allow_short=False, position=1.0)
    assert w == 0.0
    assert "因子闸门未开" in note


def test_factor_gate_passes_factor_direction():
    w, _ = combine_signals(-1.0, 0.2, mode="factor_gate", model_share=0.5,
                           factor_share=0.5, allow_short=False, position=1.0)
    assert w == pytest.approx(1.0)      # 因子 +0.2 → 做多；模型的看空被截断
    assert "做多" in _


def test_vote_picks_heavier_side():
    """投票制：两侧信号压成三档后按份额投票。"""
    w, _ = combine_signals(1.0, -1.0, mode="vote", model_share=0.7,
                           factor_share=0.3, allow_short=False, position=1.0)
    assert w == pytest.approx(1.0)      # 模型侧份额大 → 做多（因子做空被截断）
    w2, note2 = combine_signals(-1.0, 1.0, mode="vote", model_share=0.3,
                                factor_share=0.7, allow_short=False, position=1.0)
    assert w2 == pytest.approx(1.0)     # 因子侧份额大 → 做多
    assert "做多" in note2


def test_vote_tie_is_flat():
    """平票必须空仓，不能靠 tie-break 随便选边。

    用 allow_short=True：只做多时负信号会被截断成 0，那样就永远不会有
    「两侧反向」的平票，测的就不是投票制本身了。
    """
    w, note = combine_signals(1.0, -1.0, mode="vote", model_share=0.5,
                              factor_share=0.5, allow_short=True, position=1.0)
    assert w == 0.0
    assert "平手" in note


def test_blend_cancellation_is_flat():
    """两侧反向等强时相互抵消 → 空仓，而不是四舍五入到某个方向。"""
    w, note = combine_signals(0.5, -0.5, mode="blend", model_share=0.5,
                              factor_share=0.5, allow_short=True, position=1.0)
    assert w == 0.0
    assert "抵消" in note


def test_only_long_clamps_both_sides():
    """只做多时两侧负信号都要被截断，否则权重总和会被负值悄悄改变。"""
    w, _ = combine_signals(-0.8, -0.4, mode="blend", model_share=0.5,
                           factor_share=0.5, allow_short=False, position=1.0)
    assert w == 0.0


def test_unknown_mode_falls_back_to_blend():
    """配置里出现没见过的融合方式时不能崩，退回默认的加权混合。"""
    w, note = combine_signals(0.8, 0.4, mode=" nonsense ", model_share=0.5,
                              factor_share=0.5, allow_short=False, position=1.0)
    assert w == pytest.approx(0.6)
    assert "加权混合" in note


# ---------------------------------------------------------------- 权重解析容错

def test_parse_weights_falls_back_on_garbage():
    assert _parse_weights("not json", {"ewma": 1.0}) == {"ewma": 1.0}
    assert _parse_weights("[1,2,3]", {"ewma": 1.0}) == {"ewma": 1.0}


def test_parse_weights_drops_non_numeric_and_non_finite():
    out = _parse_weights(json.dumps({"a": 1.0, "b": "x", "c": True,
                                     "d": float("nan"), "e": float("inf")}), {})
    assert out == {"a": 1.0}


def test_parse_weights_accepts_empty_object():
    assert _parse_weights("{}", {"ewma": 1.0}) == {}


# ---------------------------------------------------------------- 端到端

def test_generate_signals_contract():
    """契约：三列、与 bars 等长、target_weight 在 [-position, position]。"""
    ensure_loaded()
    bars = _bars(300)
    strat = FusionStrategy(_params())
    df = strat.generate_signals(bars, _ctx(bars))
    assert list(df.columns) == ["target_weight", "reason", "confidence"]
    assert len(df) == len(bars)
    assert df["target_weight"].abs().max() <= 1.0 + 1e-9
    assert df["reason"].notna().all()


def test_generate_signals_short_history_is_flat():
    """K 线不足时必须全空仓并说明原因，不能让因子在 NaN 上算。"""
    ensure_loaded()
    bars = _bars(30)
    strat = FusionStrategy(_params(context=240))
    df = strat.generate_signals(bars, _ctx(bars))
    assert (df["target_weight"] == 0.0).all()
    assert "K 线不足" in df["reason"].iloc[-1]


def test_empty_bars_returns_empty_frame():
    strat = FusionStrategy(_params())
    df = strat.generate_signals(pd.DataFrame(), _ctx(pd.DataFrame()))
    assert df.empty
    assert list(df.columns) == ["target_weight", "reason", "confidence"]


def test_factor_only_config_works():
    """模型权重全 0 = 只用因子：融合策略要能优雅退化成单侧策略。"""
    ensure_loaded()
    bars = _bars(300)
    strat = FusionStrategy(_params(model_weights_json=json.dumps({"ewma": 0.0})))
    df = strat.generate_signals(bars, _ctx(bars))
    assert df["target_weight"].abs().max() > 0.0
    assert "未启用任何因子" not in df["reason"].iloc[-1]


def test_model_only_config_works():
    """因子权重全 0 = 只用模型。"""
    ensure_loaded()
    bars = _bars(300)
    strat = FusionStrategy(_params(factor_weights_json=json.dumps({"dual_ma": 0.0})))
    df = strat.generate_signals(bars, _ctx(bars))
    assert df["target_weight"].abs().max() > 0.0


def test_on_bar_caches_per_bar():
    """模拟盘里每根 bar 只融合一次：第二次调用必须命中缓存（同一结果）。"""
    ensure_loaded()
    bars = _bars(200)
    ctx = _ctx(bars)
    strat = FusionStrategy(_params(context=60))
    w1, r1, c1 = strat.on_bar(bars.iloc[-1], ctx)
    w2, r2, c2 = strat.on_bar(bars.iloc[-1], ctx)
    assert (w1, c1) == (w2, c2)
    assert r1 == r2


def test_on_bar_warmup_is_flat():
    ensure_loaded()
    bars = _bars(50)
    strat = FusionStrategy(_params(context=240))
    w, reason, conf = strat.on_bar(bars.iloc[-1], _ctx(bars))
    assert w == 0.0 and conf == 0.0
    assert "预热中" in reason


def test_reason_names_both_sides():
    """reason 必须同时交代模型侧和因子侧的结论，否则用户无法判断该调哪边。"""
    ensure_loaded()
    bars = _bars(300)
    strat = FusionStrategy(_params())
    df = strat.generate_signals(bars, _ctx(bars))
    tail = df["reason"].iloc[-1]
    assert "模型：" in tail and "因子：" in tail


def test_defaults_only_enable_cheap_models():
    """默认不开 Kronos：慢且多数机器没装依赖，默认配置必须毫秒级。"""
    assert DEFAULT_MODEL_WEIGHTS["kronos"] == 0.0
    assert DEFAULT_MODEL_WEIGHTS["ewma"] > 0
    assert DEFAULT_FACTOR_WEIGHTS["dual_ma"] > 0
    assert all(v == 0.0 for k, v in DEFAULT_MODEL_WEIGHTS.items()
               if k in ("chronos", "timesfm", "moirai"))


def test_config_falls_back_to_kv_store(monkeypatch, tmp_path):
    """策略参数没显式给权重时，应读 kv_store 里的全局融合配置。

    注意：融合方式/份额/门控这些**参数里没有对应字段**的项（如
    ``fusion_mode``）由 kv_store 决定；而参数里有的字段（如
    ``model_share``）默认值就是策略默认值，所以只在参数显式偏离默认
    时才谈得上「覆盖」。这里验的是 kv_store 的值确实被读进来了。
    """
    ensure_loaded()
    from quant.storage.db import init_db
    from quant.storage.repositories import KVRepo

    db = tmp_path / "t.db"
    init_db(db)
    repo = KVRepo(db)
    repo.put(CONFIG_KEY, {"factor_weights": {"macd": 1.0, "rsi": 0.5},
                          "fusion_mode": "vote"})

    monkeypatch.setattr("alpha.models.strategies.fusion._kv_config",
                        lambda: repo.get(CONFIG_KEY))
    strat = FusionStrategy(_params(factor_weights_json=json.dumps({"dual_ma": 0.0})))
    cfg = strat._config()
    assert cfg["fusion_mode"] == "vote"          # kv_store 值生效
    assert cfg["factor_weights"]["macd"] == 1.0
    assert cfg["factor_weights"]["rsi"] == 0.5
    # 模型侧参数显式给了 → 参数优先，不被 kv_store 的空值盖掉
    assert cfg["model_weights"]["ewma"] == 1.0
