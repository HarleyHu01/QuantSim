"""指标计算测试。"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quant.core.models import Trade
from quant.engine.metrics import compute_metrics, periods_for


def _equity_df(prices: list[float], n: int | None = None) -> pd.DataFrame:
    n = n or len(prices)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    eq = pd.Series(prices, index=idx)
    return pd.DataFrame({
        "equity": eq,
        "cash": 0.0,
        "position_value": eq,
        "exposure": 0.5,
        "drawdown": 1.0 - eq / eq.cummax(),
        "benchmark": eq * 0.9,
        "buyhold": eq * 0.8,       # 买入持有权益（模拟：弱于策略）
        "symbol_price": eq / 100.0,  # 标的收盘价
    })


def _trade(ts, pnl, side="sell"):
    return Trade(timestamp=ts, symbol="X", side=side, price=100.0, qty=10.0,
                 amount=1000.0, fee=1.0, slippage_cost=0.0, pnl=pnl)


def test_metrics_basic():
    n = 100
    eq = _equity_df([1_000_000 * (1 + 0.001 * i) for i in range(n)])
    m = compute_metrics(eq, [], periods_per_year=252)
    assert m["total_return"] == pytest.approx((1 + 0.001 * (n - 1)) - 1, rel=1e-6)
    assert m["annual_return"] is not None and m["annual_return"] > 0
    assert m["max_drawdown"] >= 0
    assert m["trade_count"] == 0
    assert m["win_rate"] is None
    assert m["sharpe"] is not None


def test_metrics_win_rate_and_profit_factor():
    ts0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trades = [_trade(ts0, 10.0), _trade(ts0, 20.0), _trade(ts0, -5.0), _trade(ts0, -5.0)]
    m = compute_metrics(_equity_df([1e6] * 50), trades, periods_per_year=252)
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["profit_factor"] == pytest.approx(30.0 / 10.0)
    assert m["trade_count"] == 4


def test_metrics_no_losses():
    ts0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trades = [_trade(ts0, 5.0), _trade(ts0, 8.0)]
    m = compute_metrics(_equity_df([1e6] * 50), trades, periods_per_year=252)
    assert m["profit_factor"] is None  # 无亏损


def test_periods_for_timeframes():
    assert periods_for("1d") == 252
    assert periods_for("1w") == 52
    assert periods_for("1M") == 12
    assert periods_for("1h") > 252


def test_metrics_benchmark_return():
    eq = _equity_df([1e6] * 50)
    m = compute_metrics(eq, [], periods_per_year=252)
    assert m["benchmark_return"] == pytest.approx(0.0)


def test_metrics_buyhold_and_excess():
    """buyhold 列 → 买入持有收益/年化 + 超额收益（vs 买入持有）。"""
    n = 100
    # 策略涨 ~10%（0.1%/bar），买入持有只涨 ~5%（0.05%/bar）
    prices = [1_000_000 * (1 + 0.001 * i) for i in range(n)]
    eq = _equity_df(prices)
    eq["buyhold"] = pd.Series([1_000_000 * (1 + 0.0005 * i) for i in range(n)],
                              index=eq.index)
    m = compute_metrics(eq, [], periods_per_year=252)
    assert m["buyhold_return"] is not None and m["buyhold_return"] < m["total_return"]
    assert m["buyhold_annual"] is not None and m["buyhold_annual"] > 0
    assert m["excess_return_vs_buyhold"] == pytest.approx(
        m["total_return"] - m["buyhold_return"], rel=1e-6)
    # 无 buyhold 列 → 对比指标为 None（不报错）
    eq2 = eq.drop(columns=["buyhold"])
    m2 = compute_metrics(eq2, [], periods_per_year=252)
    assert m2["buyhold_return"] is None
    assert m2["excess_return_vs_buyhold"] is None


def test_metrics_beta_alpha_information_ratio():
    """benchmark 列 → Beta / Alpha / 信息比率 / 基准年化与基准最大回撤。"""
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    # 策略 = 基准 × 1.5（beta≈1.5）+ 独立噪声
    rng = np.random.default_rng(42)
    bench_ret = rng.normal(0.001, 0.01, n - 1)
    bench = pd.Series(1_000_000.0, index=idx)
    bench.iloc[1:] = bench.iloc[0] * np.cumprod(1 + bench_ret)
    strat_ret = 1.5 * bench_ret + rng.normal(0.0005, 0.005, n - 1)
    strat = pd.Series(1_000_000.0, index=idx)
    strat.iloc[1:] = strat.iloc[0] * np.cumprod(1 + strat_ret)
    eq = _equity_df(strat.tolist(), n)
    eq["benchmark"] = bench.values
    eq["buyhold"] = bench.values * 0.95
    m = compute_metrics(eq, [], periods_per_year=252)
    assert m["beta"] is not None and m["beta"] > 1.0  # 1.5 左右
    assert m["alpha"] is not None
    assert m["information_ratio"] is not None
    assert m["benchmark_annual"] is not None
    assert m["benchmark_max_drawdown"] is not None and m["benchmark_max_drawdown"] >= 0
