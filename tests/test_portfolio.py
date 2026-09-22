"""组合模块测试：三模式公式、confidence 加权、CombinedSignal 分解、仓储、API。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from quant.core.models import PortfolioConfig
from quant.portfolio.portfolio import StrategyPortfolio
from quant.storage.db import init_db
from quant.storage.repositories import PortfolioRepo, StrategiesRepo


def sig_df(targets, confidence=None, index=None) -> pd.DataFrame:
    if index is None:
        index = pd.date_range("2024-01-01", periods=len(targets), freq="D", tz="UTC")
    conf = confidence or [1.0] * len(targets)
    return pd.DataFrame({"target_weight": targets, "reason": [""] * len(targets),
                         "confidence": conf}, index=index)


# ---------- 三模式公式 ----------

def test_signal_merge_weights_cancel():
    st = StrategyPortfolio(PortfolioConfig(mode="signal_merge", weights={"a": 50, "b": 50}))
    out = st.combine_series({"a": sig_df([1.0] * 3), "b": sig_df([-1.0] * 3)})
    assert out["target_weight"].tolist() == [0.0] * 3  # 多空抵消


def test_signal_merge_single_weight():
    st = StrategyPortfolio(PortfolioConfig(mode="signal_merge", weights={"a": 100, "b": 0}))
    out = st.combine_series({"a": sig_df([1.0] * 3), "b": sig_df([-1.0] * 3)})
    assert out["target_weight"].tolist() == [1.0] * 3


def test_signal_merge_negative_weight_reverse():
    st = StrategyPortfolio(PortfolioConfig(mode="signal_merge", weights={"a": -100}))
    out = st.combine_series({"a": sig_df([1.0] * 3)})
    assert out["target_weight"].tolist() == [-1.0] * 3  # 负权重反向


def test_confidence_weighted_contribution():
    """confidence_weighted=True：置信度 0 的策略不贡献信号。"""
    a = sig_df([1.0, 1.0])
    b = sig_df([-1.0, -1.0], confidence=[0.0, 0.0])
    st = StrategyPortfolio(PortfolioConfig(mode="signal_merge", weights={"a": 50, "b": 50},
                                           confidence_weighted=True))
    out = st.combine_series({"a": a, "b": b})
    assert out["target_weight"].tolist() == [1.0, 1.0]  # b 置信 0 → 不参与
    st2 = StrategyPortfolio(PortfolioConfig(mode="signal_merge", weights={"a": 50, "b": 50},
                                            confidence_weighted=False))
    out2 = st2.combine_series({"a": a, "b": b})
    assert out2["target_weight"].tolist() == [0.0, 0.0]  # 不按置信 → 多空抵消


def test_equal_mode():
    st = StrategyPortfolio(PortfolioConfig(mode="equal"))
    out = st.combine_series({"a": sig_df([1.0] * 3), "b": sig_df([-1.0] * 3)})
    assert out["target_weight"].tolist() == [0.0] * 3


def test_combine_at_decomposition():
    st = StrategyPortfolio(PortfolioConfig(mode="signal_merge", weights={"a": 60, "b": 40}))
    idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    ts = idx[0]
    sigs = {"a": sig_df([1.0, 1.0], index=idx), "b": sig_df([-1.0, -1.0], index=idx)}
    c = st.combine_at(sigs, ts)
    assert c.target_weight == pytest.approx(0.2)  # (0.6×1 + 0.4×(-1))/1.0
    assert c.per_strategy_weights == {"a": 60.0, "b": 40.0}
    assert c.per_strategy_contribution["a"] == pytest.approx(0.6)
    assert c.per_strategy_contribution["b"] == pytest.approx(-0.4)
    assert c.confidence == pytest.approx(1.0)


def test_capital_split_pcts():
    st = StrategyPortfolio(PortfolioConfig(mode="capital_split", weights={"a": 60, "b": 40}))
    subs = st.split([SimpleNamespace(key="a"), SimpleNamespace(key="b")])
    assert len(subs) == 2
    assert subs[0].capital_pct == pytest.approx(0.6)
    assert subs[1].capital_pct == pytest.approx(0.4)
    assert subs[0].strategy_keys == ["a"]


def test_empty_weights_means_equal():
    st = StrategyPortfolio(PortfolioConfig(mode="signal_merge", weights={}))
    out = st.combine_series({"a": sig_df([1.0] * 2), "b": sig_df([-1.0] * 2)})
    assert out["target_weight"].tolist() == [0.0, 0.0]  # 空权重 → 等权 → 抵消


# ---------- 仓储 ----------

def test_strategies_repo_roundtrip(tmp_path):
    init_db(tmp_path / "quant.db")
    repo = StrategiesRepo(tmp_path / "quant.db")
    repo.upsert("dual_ma", "双均线", "desc", config_json={"fast": 10}, weight=80.0, enabled=1)
    row = repo.get("dual_ma")
    assert row is not None
    assert json.loads(row["config_json"]) == {"fast": 10}
    assert row["weight"] == 80.0
    repo.update(row["id"], weight=50.0, enabled=0)
    updated = repo.get_by_id(row["id"])
    assert updated["weight"] == 50.0 and updated["enabled"] == 0
    repo.delete(row["id"])
    assert repo.get("dual_ma") is None


def test_portfolio_repo_roundtrip(tmp_path):
    init_db(tmp_path / "quant.db")
    repo = PortfolioRepo(tmp_path / "quant.db")
    repo.put("capital_split", {"a": 60, "b": 40}, "weekly", 0.01)
    row = repo.get()
    assert row["mode"] == "capital_split"
    assert json.loads(row["weights_json"]) == {"a": 60, "b": 40}
    assert row["rebalance_freq"] == "weekly"
    assert row["signal_threshold"] == pytest.approx(0.01)


# ---------- API ----------

def test_api_strategies_and_portfolio():
    """策略列表/权重/启停 + 组合配置读写（真实 app，结束后恢复默认）。"""
    from fastapi.testclient import TestClient

    from quant.api.app import app

    with TestClient(app) as client:
        r = client.get("/api/strategies")
        assert r.status_code == 200
        strategies = r.json()["strategies"]
        keys = [s["key"] for s in strategies]
        assert "dual_ma" in keys and len(keys) >= 8
        dual = next(s for s in strategies if s["key"] == "dual_ma")
        assert "params_schema" in dual and "params_defaults" in dual.get("params_schema", {}) or "properties" in dual["params_schema"]

        # 权重/启停
        r2 = client.put(f"/api/strategies/{dual['id']}", json={"weight": 40, "enabled": False})
        assert r2.json()["weight"] == 40 and r2.json()["enabled"] == 0
        r3 = client.post(f"/api/strategies/{dual['id']}/toggle")
        assert r3.json()["enabled"] == 1
        client.put(f"/api/strategies/{dual['id']}", json={"weight": 100, "enabled": True})  # 恢复

        # 组合配置读写
        r4 = client.put("/api/portfolio", json={
            "mode": "signal_merge", "weights": {"dual_ma": 60, "macd": 40}})
        assert r4.json()["mode"] == "signal_merge"
        assert r4.json()["weights"] == {"dual_ma": 60, "macd": 40}
        r5 = client.get("/api/portfolio")
        assert r5.json()["weights"] == {"dual_ma": 60, "macd": 40}
        client.put("/api/portfolio", json={"mode": "signal_merge", "weights": {}})  # 恢复默认
