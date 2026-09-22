"""回测引擎测试：事件驱动撮合语义、双引擎信号一致性、风控、capital_split。"""
from __future__ import annotations

import pandas as pd
import pytest

from quant.core.models import PortfolioConfig, TradingSettings
from quant.engine.backtest import EventDrivenBacktestEngine, VectorizedBacktestEngine
from quant.portfolio.portfolio import StrategyPortfolio
from tests.strategies import StepStrategy, make_bars

ZERO = dict(commission_rate=0.0, commission_min=0.0, commission_fixed=0.0,
            stamp_tax_rate=0.0, transfer_fee_rate=0.0, slippage_bps=0.0,
            signal_threshold=0.0, max_position_pct=1.0, t_plus_enabled=False)


def ed_engine(**settings_over) -> EventDrivenBacktestEngine:
    s = TradingSettings(**{**ZERO, **settings_over})
    return EventDrivenBacktestEngine(s, StrategyPortfolio(PortfolioConfig(
        mode="signal_merge", weights={"test_step": 100.0})))


def vec_engine(**settings_over) -> VectorizedBacktestEngine:
    s = TradingSettings(**{**ZERO, **settings_over})
    return VectorizedBacktestEngine(s, StrategyPortfolio(PortfolioConfig(
        mode="signal_merge", weights={"test_step": 100.0})))


def test_next_open_fill_at_next_bar_open():
    """t 收盘信号 → t+1 开盘成交（无未来函数）。"""
    bars = make_bars([100.0] * 6)  # 平价数据避免现金缺口
    result = ed_engine(execution="next_open").run({"X": bars}, [StepStrategy({"hold_bars": 3})])
    assert result.trades, "应有成交"
    t0 = result.trades[0]
    assert t0.timestamp == bars.index[1]
    assert t0.side == "buy"
    assert t0.price == pytest.approx(bars["open"].iloc[1])


def test_same_close_fill_at_signal_bar():
    bars = make_bars([100, 102, 104, 103, 105])
    result = ed_engine(execution="same_close").run({"X": bars}, [StepStrategy({"hold_bars": 3})])
    assert result.trades[0].timestamp == bars.index[0]
    assert result.trades[0].price == pytest.approx(bars["close"].iloc[0])


def test_round_trip_flat_pnl():
    """常数价格完整往返：买入→卖出，费用 0，净值回到初始。"""
    bars = make_bars([100.0] * 10)
    result = ed_engine().run({"X": bars}, [StepStrategy({"hold_bars": 3})])
    sides = [t.side for t in result.trades]
    assert sides == ["buy", "sell"]
    assert result.trades[0].pnl == 0.0
    assert result.equity.iloc[-1]["equity"] == pytest.approx(1_000_000.0)


def test_dual_engine_signal_identical():
    """双引擎信号层完全一致：逐 bar 显式对比 target_weight / confidence / reason。"""
    bars = make_bars([100, 105, 98, 110, 102, 108, 95, 99, 103, 107])
    r_ed = ed_engine().run({"X": bars}, [StepStrategy({"hold_bars": 3})])
    r_vec = vec_engine().run({"X": bars}, [StepStrategy({"hold_bars": 3})])
    sig_ed = r_ed.combined_signal["X"]
    sig_vec = r_vec.combined_signal["X"]
    assert len(sig_ed) == len(sig_vec) == len(bars)
    assert sig_ed.index.equals(sig_vec.index)  # 时间轴逐点对齐
    # 逐 bar 显式断言（pd.testing，非最终收益近似；索引 freq 元数据差异忽略）
    pd.testing.assert_series_equal(
        sig_ed["target_weight"], sig_vec["target_weight"],
        check_names=False, check_freq=False, check_exact=False, rtol=1e-9)
    pd.testing.assert_series_equal(
        sig_ed["confidence"], sig_vec["confidence"],
        check_names=False, check_freq=False, check_exact=False, rtol=1e-9)
    assert list(sig_ed["reason"]) == list(sig_vec["reason"])  # reason 逐 bar 相同


def test_vectorized_rejects_multi_symbol():
    bars = make_bars([100.0] * 5)
    with pytest.raises(ValueError, match="单标的"):
        vec_engine().run({"X": bars, "Y": bars}, [StepStrategy()])


def test_min_trade_interval_enforced():
    """min_trade_interval_bars=3：两次成交至少间隔 3 根。"""
    bars = make_bars([100.0] * 10)
    result = ed_engine(min_trade_interval_bars=3).run({"X": bars}, [StepStrategy({"hold_bars": 1})])
    # hold_bars=1：bar0 目标 1 → bar1 成交；bar1 目标 0 → 间隔不足
    # bar3（i=3 ≥ 0+3）才允许平仓决策 → bar4 成交
    idx = list(bars.index)
    assert result.trades[0].timestamp == idx[1]
    assert result.trades[1].timestamp == idx[4]


def test_stop_loss_closes_position():
    """止损：浮亏超阈值 → 市价平仓（reason 含 stop_loss）。"""
    bars = make_bars([100, 100, 100, 80, 82])
    result = ed_engine(stop_loss_pct=0.1).run({"X": bars}, [StepStrategy({"hold_bars": 5})])
    closes = [t for t in result.trades if t.side == "sell" and "stop_loss" in t.reason]
    assert closes, "应有止损平仓"


def test_capital_split_sub_accounts():
    """capital_split：两策略独立子账户，成交带各自 strategy_key。"""
    bars = make_bars([100.0] * 10)
    s = TradingSettings(**ZERO)
    st = StrategyPortfolio(PortfolioConfig(mode="capital_split", weights={"test_step": 50.0}))
    engine = EventDrivenBacktestEngine(s, st)
    # 用两个不同参数的同一策略模拟双策略
    r = engine.run({"X": bars}, [StepStrategy({"hold_bars": 3}), StepStrategy({"hold_bars": 5})])
    assert len(r.trades) >= 2
    assert all(t.strategy_key in ("test_step",) for t in r.trades)
    assert r.trades[0].strategy_name == "测试阶梯策略"
    assert "test_step" in r.per_strategy_metrics


def test_per_strategy_metrics_present():
    bars = make_bars([100, 105, 98, 110])
    result = ed_engine().run({"X": bars}, [StepStrategy()])
    assert "test_step" in result.per_strategy_metrics
    assert "total_return" in result.per_strategy_metrics["test_step"]


def test_benchmark_bars_independent():
    """独立传入的基准数据（不与交易标的耦合）用于基准曲线与基准收益。"""
    bars = make_bars([100.0] * 10)
    bench = make_bars([100.0 * (1 + 0.01 * i) for i in range(10)])
    result = ed_engine().run({"X": bars}, [StepStrategy({"hold_bars": 3})], benchmark_bars=bench)
    assert result.equity["benchmark"].notna().all()  # 每根 bar 都有基准值
    assert result.metrics["benchmark_return"] is not None
    assert result.metrics["benchmark_return"] > 0
    # 未传基准 → 默认用交易标的本体（buy&hold）作为基准（不再为空）
    result2 = ed_engine().run({"X": bars}, [StepStrategy({"hold_bars": 3})])
    assert result2.equity["benchmark"].notna().all()
    assert result2.metrics["benchmark_return"] == pytest.approx(0.0)  # 标的本身零涨幅


def test_equity_has_compare_series():
    """权益曲线附带对比列：buyhold（等权买入持有）与 symbol_price（首标的收盘价）。"""
    bars = make_bars([100.0, 102.0, 98.0, 110.0, 108.0])
    for engine in (ed_engine(), vec_engine()):
        result = engine.run({"X": bars}, [StepStrategy({"hold_bars": 2})])
        eq = result.equity
        assert "buyhold" in eq.columns and "symbol_price" in eq.columns
        # 单标的：buyhold 归一化 = 标的 close 归一化 × 初始资金
        assert eq["symbol_price"].iloc[0] == pytest.approx(100.0)
        assert eq["buyhold"].iloc[-1] == pytest.approx(
            eq["buyhold"].iloc[0] * eq["symbol_price"].iloc[-1] / eq["symbol_price"].iloc[0])
        # 指标含对比项
        m = result.metrics
        assert "buyhold_return" in m and "excess_return_vs_buyhold" in m
        assert m["excess_return_vs_buyhold"] == pytest.approx(
            m["total_return"] - m["buyhold_return"], rel=1e-6)
