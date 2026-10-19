"""Kronos 预测接入的测试。

关键原则：**测试不能真的加载模型**。
- 权重下载要几十秒且依赖网络，测试里必须打桩。
- `torch` 是可选依赖，测试要在「装了」和「没装」两种环境下都能跑：
  用 monkeypatch 打桩 `missing_deps` 来模拟缺依赖，而不是真的卸载 torch。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.models import kronos as kr


# ---------------------------------------------------------------- 基础工具

def _bars(n: int = 64, freq: str = "5min", tz: str = "UTC") -> pd.DataFrame:
    idx = pd.date_range("2026-10-12 01:30", periods=n, freq=freq, tz=tz)
    close = pd.Series(100.0 + np.arange(n, dtype=float) * 0.2, index=idx)
    return pd.DataFrame({
        "open": close - 0.05,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": np.full(n, 1000.0),
    }, index=idx)


# ---------------------------------------------------------------- spec 解析

def test_spec_resolves_short_model_name():
    """短名必须补上 HF 的 NeoQuasar/ 前缀。

    回归：早期直接把短名传给 from_pretrained，得到 401 Repository Not Found，
    而 huggingface_hub 把该异常吞掉后报的是
    「Kronos.__init__() missing 11 required positional arguments」，完全误导。
    """
    spec = kr.KronosSpec(model="Kronos-small")
    assert spec.resolved_model() == "NeoQuasar/Kronos-small"
    assert spec.resolved_tokenizer() == "NeoQuasar/Kronos-Tokenizer-base"


def test_spec_accepts_full_repo_id():
    spec = kr.KronosSpec(model="someone/my-finetuned-kronos")
    assert spec.resolved_model() == "someone/my-finetuned-kronos"


def test_spec_tokenizer_pairing_by_model():
    assert kr.KronosSpec(model="Kronos-mini").resolved_tokenizer() == "NeoQuasar/Kronos-Tokenizer-2k"
    assert kr.KronosSpec(model="Kronos-base").resolved_tokenizer() == "NeoQuasar/Kronos-Tokenizer-base"
    # 显式指定时优先用指定的
    assert kr.KronosSpec(model="Kronos-mini", tokenizer="x/y").resolved_tokenizer() == "x/y"


def test_spec_as_dict_is_serialisable():
    d = kr.KronosSpec().as_dict()
    assert d["enabled"] is True
    assert d["model"] == "NeoQuasar/Kronos-small"
    assert set(d) >= {"model", "tokenizer", "device", "max_context", "pred_len", "tz"}


def test_max_context_by_model():
    assert kr.MAX_CONTEXT["Kronos-mini"] == 2048
    assert kr.MAX_CONTEXT["Kronos-small"] == 512


# ---------------------------------------------------------------- 依赖探测

def test_available_false_when_dep_missing(monkeypatch):
    monkeypatch.setattr(kr, "missing_deps", lambda: ["torch"])
    assert kr.available() is False


def test_install_hint_uses_cpu_index_for_torch(monkeypatch):
    monkeypatch.setattr(kr, "missing_deps", lambda: ["torch", "einops"])
    hint = kr.install_hint()
    assert "download.pytorch.org/whl/cpu" in hint
    assert "einops" in hint
    assert "torch" not in hint.split("&&")[1]  # torch 不该出现在第二条 pip install 里


def test_install_hint_empty_when_all_present(monkeypatch):
    monkeypatch.setattr(kr, "missing_deps", lambda: [])
    assert kr.install_hint() == ""


def test_load_raises_when_disabled():
    fc = kr.KronosForecaster(kr.KronosSpec(enabled=False))
    with pytest.raises(kr.KronosUnavailable, match="关闭"):
        fc._load()


def test_load_raises_with_install_hint_when_deps_missing(monkeypatch):
    monkeypatch.setattr(kr, "missing_deps", lambda: ["torch"])
    fc = kr.KronosForecaster()
    with pytest.raises(kr.KronosUnavailable, match="torch"):
        fc._load()


def test_pick_device_respects_explicit_choice():
    assert kr._pick_device("cpu") == "cpu"
    assert kr._pick_device("cuda:1") == "cuda:1"


def test_pick_device_auto_never_raises():
    assert kr._pick_device("auto") in ("cpu", "cuda:0", "mps")


# ---------------------------------------------------------------- 状态查询

def test_status_does_not_load_model(monkeypatch):
    """status() 绝不能触发加载——UI 每次渲染都会调它。"""
    called = {"n": 0}
    monkeypatch.setattr(kr.KronosForecaster, "_load",
                        lambda self: called.__setitem__("n", called["n"] + 1))
    kr._CACHE.clear()
    s = kr.KronosForecaster().status()
    assert called["n"] == 0
    assert s["loaded"] is False
    assert s["enabled"] is True
    assert "model_choices" in s


# ---------------------------------------------------------------- forecast 契约

def test_forecast_rejects_empty_and_missing_columns():
    fc = kr.KronosForecaster()
    with pytest.raises(ValueError, match="为空"):
        fc.forecast(pd.DataFrame())
    with pytest.raises(ValueError, match="缺少列"):
        fc.forecast(pd.DataFrame({"close": [1.0]}))


def test_forecast_rejects_non_datetime_index():
    df = _bars(40)
    df.index = range(len(df))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        kr.KronosForecaster().forecast(df)


def test_forecast_converts_index_to_local_tz_and_calls_predictor(monkeypatch):
    """上游按「本地时间」提取 时/分/周 特征，喂 UTC 会算错特征，必须先转时区。"""
    captured: dict = {}

    class FakePredictor:
        def predict(self, df, x_timestamp, y_timestamp, pred_len, **kw):
            captured["x_tz"] = df.index.tz
            captured["first"] = df.index[0]
            captured["rows"] = len(df)
            captured["pred_len"] = pred_len
            out = pd.DataFrame(
                {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05,
                 "volume": 10.0, "amount": 10.5},
                index=pd.DatetimeIndex(list(y_timestamp)),
            )
            return out

    fake = kr._Loaded(model=object(), tokenizer=object(), predictor=FakePredictor(),
                      device="cpu", load_seconds=0.1, extra={"params_million": 24.7})
    monkeypatch.setattr(kr.KronosForecaster, "_load", lambda self: fake)

    bars = _bars(64)  # 索引是 UTC
    out = kr.KronosForecaster(kr.KronosSpec(tz="Asia/Shanghai")).forecast(bars, pred_len=6)
    assert captured["x_tz"] is not None
    assert captured["first"].utcoffset().total_seconds() == 8 * 3600
    assert captured["pred_len"] == 6
    assert len(out) == 6


def test_forecast_truncates_to_max_context(monkeypatch):
    captured: dict = {}

    class FakePredictor:
        def predict(self, df, x_timestamp, y_timestamp, pred_len, **kw):
            captured["rows"] = len(df)
            return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0],
                                 "close": [1.0], "volume": [1.0], "amount": [1.0]},
                                index=pd.DatetimeIndex(list(y_timestamp)))

    fake = kr._Loaded(object(), object(), FakePredictor(), "cpu", 0.1, {})
    monkeypatch.setattr(kr.KronosForecaster, "_load", lambda self: fake)
    kr.KronosForecaster(kr.KronosSpec(max_context=32)).forecast(_bars(200), pred_len=2)
    assert captured["rows"] == 32


def test_forecast_drops_nan_price_rows(monkeypatch):
    captured: dict = {}

    class FakePredictor:
        def predict(self, df, x_timestamp, y_timestamp, pred_len, **kw):
            captured["rows"] = len(df)
            return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0],
                                 "close": [1.0], "volume": [1.0], "amount": [1.0]},
                                index=pd.DatetimeIndex(list(y_timestamp)))

    fake = kr._Loaded(object(), object(), FakePredictor(), "cpu", 0.1, {})
    monkeypatch.setattr(kr.KronosForecaster, "_load", lambda self: fake)
    bars = _bars(20)
    bars.loc[bars.index[3], "close"] = np.nan  # 模拟停牌
    kr.KronosForecaster(kr.KronosSpec(max_context=512)).forecast(bars, pred_len=1)
    assert captured["rows"] == 19


def test_forecast_adds_amount_column(monkeypatch):
    """上游要求可选的 amount；缺失时要自己补，不能直接传给模型。"""
    captured: dict = {}

    class FakePredictor:
        def predict(self, df, x_timestamp, y_timestamp, pred_len, **kw):
            captured["cols"] = list(df.columns)
            return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0],
                                 "close": [1.0], "volume": [1.0], "amount": [1.0]},
                                index=pd.DatetimeIndex(list(y_timestamp)))

    fake = kr._Loaded(object(), object(), FakePredictor(), "cpu", 0.1, {})
    monkeypatch.setattr(kr.KronosForecaster, "_load", lambda self: fake)
    kr.KronosForecaster().forecast(_bars(40), pred_len=1)
    assert "amount" in captured["cols"]


def test_forecast_raises_on_bad_pred_len():
    with pytest.raises(ValueError, match="pred_len"):
        kr.KronosForecaster().forecast(_bars(40), pred_len=0)


# ---------------------------------------------------------------- 未来时间戳

def test_future_index_uses_median_step():
    idx = pd.date_range("2026-01-01", periods=10, freq="5min", tz="UTC")
    out = kr.KronosForecaster()._future_index(idx, 3)
    assert len(out) == 3
    assert out[0] - idx[-1] == pd.Timedelta(minutes=5)
    assert out[2] - idx[-1] == pd.Timedelta(minutes=15)


def test_future_index_falls_back_to_daily():
    idx = pd.DatetimeIndex(["2026-01-01"])
    out = kr.KronosForecaster()._future_index(idx, 2)
    assert out[0] - idx[-1] == pd.Timedelta(days=1)


def test_future_index_keeps_tz():
    idx = pd.date_range("2026-01-01", periods=5, freq="1h", tz="Asia/Shanghai")
    out = kr.KronosForecaster()._future_index(idx, 2)
    assert str(out.tz) == "Asia/Shanghai"


# ---------------------------------------------------------------- 策略层

class _StubForecaster:
    """替身：按预设序列返回预测收益，避免真的跑模型。"""

    def __init__(self, rets):
        self.rets = list(rets)
        self.i = 0

    def forecast(self, bars, pred_len=None, **kw):
        ret = self.rets[min(self.i, len(self.rets) - 1)]
        self.i += 1
        last = float(bars["close"].iloc[-1])
        idx = pd.date_range("2026-01-01", periods=int(pred_len or 1), freq="5min", tz="UTC")
        return pd.DataFrame({"open": last, "high": last, "low": last,
                             "close": [last * (1 + ret)] * len(idx),
                             "volume": [1.0] * len(idx), "amount": [1.0] * len(idx)},
                            index=idx)


def _make_strategy(rets, **params):
    from alpha.models.strategies.kronos_forecast import KronosForecastStrategy

    strat = KronosForecastStrategy(params or {"context": 40, "pred_len": 4,
                                              "rebalance_every": 20, "threshold_sigma": 0.5})
    stub = _StubForecaster(rets)
    strat._forecaster = lambda: stub  # 打桩，绕开模型
    return strat


def test_kronos_strategy_is_registered_and_has_ui_schema():
    from quant.strategies.registry import ensure_loaded, get

    ensure_loaded()
    cls = get("kronos_forecast")
    schema = cls.params_schema()
    props = schema["properties"]
    # UI 表单要求：数值字段必须带范围，否则渲染成无约束输入框
    for name in ("context", "pred_len", "rebalance_every", "threshold_sigma"):
        assert "minimum" in props[name] and "maximum" in props[name], name
    assert props["only_long"]["default"] is True


def test_kronos_strategy_contract_columns_and_index():
    bars = _bars(120)
    strat = _make_strategy([0.02])
    out = strat.generate_signals(bars, None)
    assert list(out.columns) == ["target_weight", "reason", "confidence"]
    assert len(out) == len(bars)
    assert out.index.equals(bars.index)
    assert out["target_weight"].between(0, 1).all()


def test_kronos_strategy_warmup_all_zero():
    bars = _bars(30)
    strat = _make_strategy([0.02], context=40)
    out = strat.generate_signals(bars, None)
    assert (out["target_weight"] == 0).all()
    assert "K 线不足" in str(out["reason"].iloc[0])


def test_kronos_strategy_threshold_is_volatility_scaled():
    """同样 +0.05% 的预测：低波动时开仓，高波动时不开仓。

    这验证了「阈值按波动率归一」这个设计——绝对百分比阈值在 5 分钟线和日线上矛盾。
    """
    from quant.strategies.registry import ensure_loaded

    ensure_loaded()
    flat = _bars(120)
    strat = _make_strategy([0.0005], context=40)  # +0.05%
    w_flat, why_flat = strat._weight_from_return(0.0005, strat._sigma(flat.head(60)))
    assert w_flat == 1.0, why_flat

    rng = np.random.default_rng(3)
    noisy = flat.copy()
    noisy["close"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(noisy))))  # 1% 波动
    sigma_noisy = strat._sigma(noisy)
    assert sigma_noisy > strat._sigma(flat)
    w_noisy, why_noisy = strat._weight_from_return(0.0005, sigma_noisy)
    assert w_noisy == 0.0, why_noisy


def test_kronos_strategy_only_long_blocks_short():
    strat = _make_strategy([-0.05], context=40, only_long=True)
    w, why = strat._weight_from_return(-0.05, 0.001)
    assert w == 0.0
    assert "空仓" in why

    strat2 = _make_strategy([-0.05], context=40, only_long=False)
    w2, why2 = strat2._weight_from_return(-0.05, 0.001)
    assert w2 == -1.0
    assert "做空" in why2


def test_kronos_strategy_min_move_floor():
    """波动极低时 min_move 兜底，避免阈值退化成 0 而频繁进出。"""
    strat = _make_strategy([0.0004], context=40, min_move=0.001)
    w, _ = strat._weight_from_return(0.0004, 0.0)
    assert w == 0.0
    w2, _ = strat._weight_from_return(0.002, 0.0)
    assert w2 == 1.0


def test_kronos_strategy_zero_confidence_when_flat():
    """空仓时置信度必须是 0，否则 UI 出现「置信度 0.8 但仓位 0」的误导组合。"""
    bars = _bars(120)
    strat = _make_strategy([-0.5], context=40, only_long=True)
    out = strat.generate_signals(bars, None)
    flat = out["target_weight"] == 0
    assert (out.loc[flat, "confidence"] == 0).all()


def test_kronos_strategy_forecast_failure_degrades_gracefully(monkeypatch):
    """单次预测抛错不能让整个回测崩，应降级为空仓。"""
    from alpha.models.strategies.kronos_forecast import KronosForecastStrategy

    strat = KronosForecastStrategy({"context": 40, "pred_len": 4, "rebalance_every": 20})

    class Boom:
        def forecast(self, *a, **kw):
            raise RuntimeError("模型挂了")

    strat._forecaster = lambda: Boom()
    out = strat.generate_signals(_bars(120), None)
    assert (out["target_weight"] == 0).all()


def test_kronos_strategy_on_bar_caches_per_bar():
    bars = _bars(120)
    strat = _make_strategy([0.02], context=40)

    class Ctx:
        history = bars

    r1 = strat.on_bar(bars.iloc[-1], Ctx())
    r2 = strat.on_bar(bars.iloc[-1], Ctx())  # 同一根 bar
    assert r1 == r2
    assert strat._bar_cache[0] == bars.index[-1]


def test_kronos_strategy_on_bar_warmup():
    bars = _bars(20)
    strat = _make_strategy([0.02], context=40)

    class Ctx:
        history = bars

    w, why, c = strat.on_bar(bars.iloc[-1], Ctx())
    assert (w, c) == (0.0, 0.0)
    assert "预热" in why


# ---------------------------------------------------------------- API

def test_api_kronos_status_and_deps_graceful(monkeypatch):
    """缺依赖时 /ml/kronos/status 必须 200 + available=false，不能 500。"""
    from quant.api.routers import ml as ml_router

    monkeypatch.setattr(kr, "missing_deps", lambda: ["torch"])
    s = ml_router.kronos_status()
    assert s["available"] is False
    assert s["missing"] == ["torch"]


def test_api_kronos_load_reports_missing_deps(monkeypatch):
    from quant.api.routers import ml as ml_router

    monkeypatch.setattr(kr, "missing_deps", lambda: ["torch"])
    r = ml_router.kronos_load()
    assert r["ok"] is False
    assert r["available"] is False
    assert "torch" in r["message"]


def test_api_kronos_predict_requires_data(monkeypatch):
    from fastapi import HTTPException

    from quant.api.routers import ml as ml_router
    from quant.api.state import app_state

    monkeypatch.setattr(kr, "missing_deps", lambda: [])

    class FakeDM:
        def load_bars(self, symbol, timeframe):
            return pd.DataFrame()

    app_state["data_manager"] = FakeDM()
    try:
        monkeypatch.setattr(kr, "missing_deps", lambda: [])
        with pytest.raises(HTTPException) as e:
            ml_router.kronos_predict(ml_router.KronosPredictIn(symbol="600519.SH"))
        assert e.value.status_code == 404
        assert "无数据" in e.value.detail
    finally:
        app_state.clear()


def test_api_kronos_predict_payload_shape(monkeypatch):
    from quant.api.routers import ml as ml_router
    from quant.api.state import app_state

    bars = _bars(60)
    monkeypatch.setattr(kr, "missing_deps", lambda: [])
    monkeypatch.setattr(kr.KronosForecaster, "_load",
                        lambda self: kr._Loaded(object(), object(), _FixedPredictor(),
                                                "cpu", 0.5, {"params_million": 24.7}))

    class FakeDM:
        def load_bars(self, symbol, timeframe):
            return bars

    app_state["data_manager"] = FakeDM()
    try:
        r = ml_router.kronos_predict(ml_router.KronosPredictIn(
            symbol="600519.SH", timeframe="5m", bars=48, pred_len=6))
    finally:
        app_state.clear()

    assert r["ok"] is True
    assert r["context_bars"] == 48
    assert len(r["forecast"]) == 6
    assert len(r["history"]) == 48
    assert r["last_close"] > 0
    assert isinstance(r["expected_return"], float)
    assert r["forecast"][0]["kind"] == "forecast"
    assert r["history"][0]["kind"] == "history"


class _FixedPredictor:
    """固定涨幅 1% 的假预测器。"""

    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kw):
        last = float(df["close"].iloc[-1])
        idx = pd.DatetimeIndex(list(y_timestamp))
        return pd.DataFrame({
            "open": [last] * len(idx),
            "high": [last * 1.02] * len(idx),
            "low": [last * 0.99] * len(idx),
            "close": [last * (1.01 ** (i + 1)) for i in range(len(idx))],
            "volume": [1.0] * len(idx),
            "amount": [1.0] * len(idx),
        }, index=idx)


# ---------------------------------------------------------------- 配置

def test_ml_config_defaults_present():
    from quant.core.config import AppConfig

    cfg = AppConfig.load().ml.kronos
    assert cfg.enabled is True
    assert cfg.device == "auto"
    assert cfg.tz == "Asia/Shanghai"
    assert cfg.max_context >= 32
