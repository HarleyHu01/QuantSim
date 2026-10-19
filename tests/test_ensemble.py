"""多模型集成的测试。

关键原则同 Kronos：**不加载任何真实权重**。
集成层价值全在「怎么把多个模型的票合成一个仓位」和「硬性门控怎么判」，
所以测试重点在合成数学与门控边界，而不是模型本身。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.models import baselines as bl
from alpha.models.ensemble import (DEFAULT_WEIGHTS, CONFIG_KEY, EnsembleForecaster,
                                   EnsembleSpec, ModelVote, auto_weights, list_models)


# ---------------------------------------------------------------- 夹具

def _bars(n: int = 300, freq: str = "5min", tz: str = "UTC",
          drift: float = 0.0, vol: float = 0.0) -> pd.DataFrame:
    """构造行情：可指定每根线性漂移与正弦扰动幅度。"""
    idx = pd.date_range("2026-10-12 01:30", periods=n, freq=freq, tz=tz)
    base = 100.0 + np.arange(n, dtype=float) * drift
    if vol:
        base = base * (1.0 + vol * np.sin(np.arange(n) / 7.0))
    close = pd.Series(base, index=idx)
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(n, 1000.0),
    }, index=idx)


class _StubForecaster:
    """固定预测值的假预测器（替代真实模型）。"""

    def __init__(self, key: str, ret: float | None, weight_ok: bool = True):
        self.key = key
        self.label = key.upper()
        self.kind = "stat"
        self._ret = ret
        self._ok = weight_ok

    def status(self):
        return {"key": self.key, "label": self.label, "kind": self.kind,
                "available": self._ok, "missing": [], "params": {}}

    def expected_return(self, bars, pred_len):
        return self._ret


def _patch_models(monkeypatch, mapping: dict[str, float | None]):
    """把 MODEL_FACTORIES 换成固定投票的假模型，返回可直接 predict 的集成器。

    默认 ``min_agreement=0``：这些用例考的是「加权怎么算」，
    一致性/分散度门控另有专门用例（各自显式设定阈值）。
    """
    from alpha.models import ensemble as ens

    factories = {}
    for key, ret in mapping.items():
        factories[key] = (lambda r: (lambda params=None: _StubForecaster(key, r)))(ret)
    monkeypatch.setattr(ens, "MODEL_FACTORIES", factories)
    spec = EnsembleSpec(weights={k: 1.0 for k in mapping},
                        pred_len=6, context=32, min_models=1,
                        min_agreement=0.0)
    return EnsembleForecaster(spec)


# ---------------------------------------------------------------- Spec 容错

def test_spec_defaults_match_documented_values():
    s = EnsembleSpec()
    assert s.min_models == 2
    assert s.min_return_sigma == 0.3
    assert s.min_agreement == 0.6
    assert s.max_dispersion == 2.0
    assert s.allow_short is False
    assert set(s.weights) == set(DEFAULT_WEIGHTS)


def test_spec_from_dict_tolerates_garbage():
    """配置存的是 JSON，手改坏了不能让策略直接崩。"""
    s = EnsembleSpec.from_dict({
        "weights": {"kronos": "not-a-number", "ewma": 2.0, "unknown": 1.0},
        "min_models": "3",          # 类型不对 → 保持默认
        "min_agreement": 0.9,
        "allow_short": "yes",       # 不是 bool → 保持默认
        "totally_unknown": 1,       # 未知键 → 忽略
    })
    assert s.weights == {"ewma": 2.0}
    assert s.min_models == 2        # 默认值
    assert s.min_agreement == 0.9
    assert s.allow_short is False


def test_spec_roundtrip_is_stable():
    s = EnsembleSpec(weights={"kronos": 2.0}, min_models=3, allow_short=True)
    again = EnsembleSpec.from_dict(s.as_dict())
    assert again.as_dict() == s.as_dict()


# ---------------------------------------------------------------- 合成数学

def test_combine_is_weighted_mean(monkeypatch):
    ens = _patch_models(monkeypatch, {"a": 0.02, "b": 0.00})
    r = ens.predict(_bars(), pred_len=6)
    # 等权 → (0.02 + 0.0) / 2 = 0.01
    assert r.expected_return == pytest.approx(0.01, rel=1e-9)
    assert r.target_weight > 0         # 方向为正，只做多通过
    assert len(r.votes) == 2


def test_combine_weight_ratio_matters(monkeypatch):
    from alpha.models import ensemble as ens_mod

    factories = {
        "bull": lambda params=None: _StubForecaster("bull", 0.02),
        "bear": lambda params=None: _StubForecaster("bear", -0.02),
    }
    monkeypatch.setattr(ens_mod, "MODEL_FACTORIES", factories)
    # 多头权重 3 : 空头 1 → (0.06 - 0.02)/4 = +0.01
    spec = EnsembleSpec(weights={"bull": 3.0, "bear": 1.0}, min_models=1,
                        min_agreement=0.0, context=32, pred_len=6)
    r = EnsembleForecaster(spec).predict(_bars(), pred_len=6)
    assert r.expected_return == pytest.approx(0.01, rel=1e-9)


def test_zero_weight_models_are_not_voted(monkeypatch):
    ens = _patch_models(monkeypatch, {"a": 0.01, "b": 0.01})
    ens.spec.weights["b"] = 0.0
    r = ens.predict(_bars(), pred_len=6)
    assert [v.key for v in r.votes] == ["a"]


# ---------------------------------------------------------------- 门控

def test_min_models_gate_blocks(monkeypatch):
    ens = _patch_models(monkeypatch, {"a": 0.05})
    ens.spec.min_models = 2
    r = ens.predict(_bars(), pred_len=6)
    assert r.target_weight == 0.0
    assert r.gates["min_models"]["pass"] is False
    assert "min_models" in r.reason


def test_min_return_sigma_gate_uses_volatility(monkeypatch):
    """同一预测幅度，在高波动行情下应被门控拦住。"""
    ens = _patch_models(monkeypatch, {"a": 0.001})
    ens.spec.min_return_sigma = 5.0
    ens.spec.min_models = 1

    calm = ens.predict(_bars(vol=0.0001), pred_len=6)
    wild = ens.predict(_bars(vol=0.02), pred_len=6)

    assert calm.gates["min_return_sigma"]["need"] < wild.gates["min_return_sigma"]["need"]
    assert calm.target_weight > 0
    assert wild.target_weight == 0.0
    assert "min_return_sigma" in wild.reason


def test_agreement_gate_blocks_on_split_votes(monkeypatch):
    """两个模型反向下注、权重相等 → 一致度 0.5，低于默认 0.6。"""
    from alpha.models import ensemble as ens_mod

    monkeypatch.setattr(ens_mod, "MODEL_FACTORIES", {
        "bull": lambda params=None: _StubForecaster("bull", 0.03),
        "bear": lambda params=None: _StubForecaster("bear", -0.01),
    })
    spec = EnsembleSpec(weights={"bull": 1.0, "bear": 1.0}, min_models=1,
                        min_agreement=0.6, context=32, pred_len=6)
    r = EnsembleForecaster(spec).predict(_bars(), pred_len=6)
    assert r.agreement == pytest.approx(0.5)
    assert r.target_weight == 0.0
    assert "min_agreement" in r.reason


def test_dispersion_gate_blocks_on_wild_disagreement(monkeypatch):
    from alpha.models import ensemble as ens_mod

    monkeypatch.setattr(ens_mod, "MODEL_FACTORIES", {
        "sane": lambda params=None: _StubForecaster("sane", 0.01),
        "wild": lambda params=None: _StubForecaster("wild", 0.30),
    })
    spec = EnsembleSpec(weights={"sane": 1.0, "wild": 1.0}, min_models=1,
                        min_agreement=0.0, max_dispersion=0.5,
                        context=32, pred_len=6)
    r = EnsembleForecaster(spec).predict(_bars(vol=0.001), pred_len=6)
    assert r.gates["max_dispersion"]["pass"] is False
    assert r.target_weight == 0.0


def test_direction_gate_has_readable_reason(monkeypatch):
    """看空 + 只做多：不该报「未通过硬性门控（direction）」这种黑话。"""
    ens = _patch_models(monkeypatch, {"a": -0.05})
    r = ens.predict(_bars(), pred_len=6)
    assert r.target_weight == 0.0
    assert "看空" in r.reason and "只做多" in r.reason
    assert "未通过硬性门控" not in r.reason


def test_short_allowed_produces_negative_weight(monkeypatch):
    ens = _patch_models(monkeypatch, {"a": -0.05})
    ens.spec.allow_short = True
    ens.spec.min_agreement = 0.0
    r = ens.predict(_bars(), pred_len=6)
    assert r.target_weight == pytest.approx(-ens.spec.position)
    assert "做空" in r.reason


def test_flat_when_no_model_returns_value(monkeypatch):
    ens = _patch_models(monkeypatch, {"a": None})
    r = ens.predict(_bars(), pred_len=6)
    assert r.target_weight == 0.0
    assert r.confidence == 0.0
    assert r.gates["override"] == "无有效模型预测"


def test_failing_model_does_not_break_ensemble(monkeypatch):
    """一个模型抛异常，其余模型仍要给出结论。"""
    from alpha.models import ensemble as ens_mod

    class _Boom:
        key, label, kind = "boom", "BOOM", "stat"

        def status(self):
            return {"key": "boom", "label": "BOOM", "kind": "stat", "available": True}

        def expected_return(self, bars, pred_len):
            raise RuntimeError("模型崩了")

    monkeypatch.setattr(ens_mod, "MODEL_FACTORIES", {
        "boom": lambda params=None: _Boom(),
        "ok": lambda params=None: _StubForecaster("ok", 0.02),
    })
    spec = EnsembleSpec(weights={"boom": 1.0, "ok": 1.0}, min_models=1,
                        min_agreement=0.0, context=32, pred_len=6)
    r = EnsembleForecaster(spec).predict(_bars(), pred_len=6)
    assert r.target_weight > 0
    boom_vote = [v for v in r.votes if v.key == "boom"][0]
    assert boom_vote.ok is False
    assert "模型崩了" in boom_vote.error


def test_confidence_is_zero_when_flat(monkeypatch):
    """空仓时置信度必须是 0，否则 UI 会出现「置信度 0.8 但仓位 0」。"""
    ens = _patch_models(monkeypatch, {"a": -0.05})
    r = ens.predict(_bars(), pred_len=6)
    assert r.target_weight == 0.0
    assert r.confidence == 0.0


# ---------------------------------------------------------------- 模型清单

def test_list_models_reports_without_loading():
    items = list_models()
    keys = [m["key"] for m in items]
    assert "kronos" in keys
    assert {"random_walk", "drift", "momentum", "reversion", "ewma"} <= set(keys)
    assert {"chronos", "timesfm", "moirai"} <= set(keys)
    # 每个都要有可用性字段，UI 才能画表
    for m in items:
        assert "available" in m
        assert "label" in m


def test_stat_baselines_are_zero_dependency_and_forecast():
    for cls in bl.STAT_FORECASTERS.values():
        fc = cls()
        st = fc.status()
        assert st["available"] is True
        out = fc.forecast(_bars(120), 5)
        assert len(out) == 5
        assert list(out.columns)[:4] == ["open", "high", "low", "close"]
        assert (out["high"] >= out["close"]).all()
        assert (out["low"] <= out["close"]).all()


def test_stat_baseline_direction_follows_hypothesis():
    """假设不同的基线在明确趋势上应当给出不同的符号。"""
    up = _bars(200, drift=0.4)
    assert bl.MomentumForecaster().expected_return(up, 5) > 0
    assert bl.DriftForecaster().expected_return(up, 5) > 0
    # 随机游走永远是 0
    assert bl.RandomWalkForecaster().expected_return(up, 5) == pytest.approx(0.0)


# ---------------------------------------------------------------- 自动权重

def test_auto_weights_returns_normalised_positive_weights():
    # 用带趋势的行情：全平序列会让所有模型预测恒 0 → 标准差为 0 → 无正贡献
    r = auto_weights(_bars(400, drift=0.15), pred_len=6, eval_bars=200, step=20,
                     context=60, keys=["drift", "momentum", "ewma"])
    assert r["ok"] is True
    assert set(r["weights"]) <= {"drift", "momentum", "ewma"}
    assert sum(r["weights"].values()) == pytest.approx(1.0, abs=1e-3)
    assert all(w >= 0 for w in r["weights"].values())
    for key, d in r["detail"].items():
        if d.get("ok"):
            assert d["score"] >= 0


def test_auto_weights_rejects_short_history():
    r = auto_weights(_bars(50), pred_len=6, context=60, keys=["drift"])
    assert r["ok"] is False
    assert "K 线不足" in r["error"]


def test_auto_weights_rejects_unknown_keys_gracefully():
    """传了不存在的模型 key → 不算崩，只是没有可用样本。"""
    r = auto_weights(_bars(400), pred_len=6, eval_bars=200, step=20, context=60,
                     keys=["does_not_exist"])
    assert r["ok"] is False


# ---------------------------------------------------------------- API

def _api_env(monkeypatch, bars: pd.DataFrame, kv: dict | None = None):
    """注入 data_manager + kv_repo，返回 ml 路由模块。"""
    from quant.api.routers import ml as ml_router
    from quant.api.state import app_state

    class FakeDM:
        def load_bars(self, symbol, timeframe):
            return bars

    class FakeKV:
        def __init__(self):
            self.data = dict(kv or {})

        def get(self, key, default=None):
            return self.data.get(key, default)

        def put(self, key, value):
            self.data[key] = value

    app_state["data_manager"] = FakeDM()
    app_state["kv_repo"] = FakeKV()
    return ml_router


def test_api_models_lists_all(monkeypatch):
    ml_router = _api_env(monkeypatch, _bars())
    from quant.api.state import app_state
    try:
        r = ml_router.models()
    finally:
        app_state.clear()
    assert "models" in r
    assert "kronos" in [m["key"] for m in r["models"]]
    assert set(r["available"]) | set(r["missing"]) == {m["key"] for m in r["models"]}


def test_api_ensemble_config_roundtrip(monkeypatch):
    ml_router = _api_env(monkeypatch, _bars())
    from quant.api.state import app_state
    try:
        default = ml_router.ensemble_config()["config"]
        assert set(default["weights"]) == set(DEFAULT_WEIGHTS)

        out = ml_router.ensemble_config_put(ml_router.EnsembleConfigIn(
            weights={"kronos": 2.0}, min_models=3, allow_short=True))
        assert out["ok"] is True
        assert out["config"]["weights"] == {"kronos": 2.0}
        assert out["config"]["min_models"] == 3
        assert out["config"]["allow_short"] is True

        # 再读一次，确认落库
        again = ml_router.ensemble_config()["config"]
        assert again["min_models"] == 3
        assert again["weights"] == {"kronos": 2.0}
    finally:
        app_state.clear()


def test_api_ensemble_config_clamps_negative_weights(monkeypatch):
    """负权重会变成「反向投票」，API 层直接截断到 0。"""
    ml_router = _api_env(monkeypatch, _bars())
    from quant.api.state import app_state
    try:
        out = ml_router.ensemble_config_put(ml_router.EnsembleConfigIn(
            weights={"kronos": -1.5, "ewma": 0.5}))
        assert out["config"]["weights"] == {"kronos": 0.0, "ewma": 0.5}
    finally:
        app_state.clear()


def test_api_ensemble_predict_requires_data(monkeypatch):
    from fastapi import HTTPException

    ml_router = _api_env(monkeypatch, pd.DataFrame())
    from quant.api.state import app_state
    try:
        with pytest.raises(HTTPException) as e:
            ml_router.ensemble_predict(ml_router.EnsemblePredictIn(symbol="600519.SH"))
        assert e.value.status_code == 404
        assert "无数据" in e.value.detail
    finally:
        app_state.clear()


def test_api_ensemble_predict_shrinks_context_when_short(monkeypatch):
    """历史不足以撑满 context 时应自动缩短，而不是直接 400。"""
    ml_router = _api_env(monkeypatch, _bars(60), kv={CONFIG_KEY: {
        "weights": {"random_walk": 1.0}, "context": 240, "pred_len": 6,
        "min_models": 1, "min_agreement": 0.0}})
    from quant.api.state import app_state
    try:
        r = ml_router.ensemble_predict(ml_router.EnsemblePredictIn(
            symbol="600519.SH", timeframe="5m", bars=60, pred_len=6))
    finally:
        app_state.clear()
    assert r["ok"] is True
    assert r["context_bars"] == 60
    assert r["config"]["context"] < 240


def test_api_ensemble_predict_rejects_tiny_history(monkeypatch):
    """窗口仍不足 40 根时明确 400（32 是请求下限，38 根减去 pred_len 后不够）。"""
    from fastapi import HTTPException

    ml_router = _api_env(monkeypatch, _bars(38))
    from quant.api.state import app_state
    try:
        with pytest.raises(HTTPException) as e:
            ml_router.ensemble_predict(ml_router.EnsemblePredictIn(
                symbol="600519.SH", bars=38, pred_len=6))
        assert e.value.status_code == 400
        assert "历史不足" in e.value.detail
    finally:
        app_state.clear()


def test_api_ensemble_predict_payload_shape(monkeypatch):
    """返回体必须含 votes / gates，UI 才能画「为什么空仓」。"""
    ml_router = _api_env(monkeypatch, _bars(300), kv={CONFIG_KEY: {
        "weights": {"random_walk": 1.0, "momentum": 1.0},
        "context": 120, "pred_len": 6, "min_models": 2, "min_agreement": 0.0}})
    from quant.api.state import app_state
    try:
        r = ml_router.ensemble_predict(ml_router.EnsemblePredictIn(
            symbol="600519.SH", timeframe="5m", bars=300, pred_len=6))
    finally:
        app_state.clear()
    for field in ("expected_return", "target_weight", "confidence", "reason",
                  "votes", "gates", "agreement", "dispersion", "config",
                  "last_close", "context_bars", "elapsed_seconds"):
        assert field in r, f"缺少字段 {field}"
    assert len(r["votes"]) == 2
    assert "min_models" in r["gates"]


def test_api_auto_weights_does_not_persist(monkeypatch):
    """建议权重只回传，不写库——用户要在 UI 里确认。"""
    ml_router = _api_env(monkeypatch, _bars(400))
    from quant.api.state import app_state
    try:
        kv = app_state["kv_repo"]
        r = ml_router.ensemble_auto_weights(ml_router.AutoWeightsIn(
            symbol="600519.SH", timeframe="5m", pred_len=6, eval_bars=200,
            step=20, context=60, keys=["drift", "momentum"]))
        assert kv.get(CONFIG_KEY) is None
    finally:
        app_state.clear()
    assert r["ok"] in (True, False)     # 趋势不明时可能全 0 → ok False，都算正常
    assert "elapsed_seconds" in r


def test_api_auto_weights_requires_data(monkeypatch):
    from fastapi import HTTPException

    ml_router = _api_env(monkeypatch, pd.DataFrame())
    from quant.api.state import app_state
    try:
        with pytest.raises(HTTPException) as e:
            ml_router.ensemble_auto_weights(ml_router.AutoWeightsIn(symbol="600519.SH"))
        assert e.value.status_code == 404
    finally:
        app_state.clear()
