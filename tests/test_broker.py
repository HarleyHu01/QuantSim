"""Broker 撮合规则测试：费用/滑点/T+1/涨跌停/做空保证金/强平/lot。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant.core.models import Order, T1Ledger, TradingSettings
from quant.engine.broker import Broker, apply_slippage, calc_fee
from quant.engine.risk import RiskManager

UTC = timezone.utc


def ts(day: int, hour: int = 10) -> datetime:
    return datetime(2024, 1, day, hour, tzinfo=UTC)


ZERO_FEE = dict(commission_rate=0.0, commission_min=0.0, commission_fixed=0.0,
                stamp_tax_rate=0.0, transfer_fee_rate=0.0, slippage_bps=0.0)


def make_broker(**overrides) -> Broker:
    s = TradingSettings(**{**ZERO_FEE, **overrides})
    return Broker(s)


# ================= 费用（断言精确到小数） =================

def test_commission_proportional():
    s = TradingSettings(commission_rate=0.00025, commission_min=0.0,
                        commission_fixed=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0)
    assert calc_fee(10_000.0, "buy", s) == pytest.approx(2.5)  # 万2.5


def test_commission_minimum():
    s = TradingSettings(commission_rate=0.00025, commission_min=5.0,
                        commission_fixed=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0)
    assert calc_fee(1_000.0, "buy", s) == pytest.approx(5.0)  # 0.25 → 最低 5 元


def test_commission_fixed_takes_larger():
    s = TradingSettings(commission_rate=0.00025, commission_fixed=8.0, commission_min=0.0,
                        stamp_tax_rate=0.0, transfer_fee_rate=0.0)
    assert calc_fee(100_000.0, "buy", s) == pytest.approx(25.0)  # max(8, 25)


def test_stamp_tax_sell_only():
    s = TradingSettings(commission_rate=0.0, commission_min=0.0, commission_fixed=0.0,
                        stamp_tax_rate=0.0005, transfer_fee_rate=0.0)
    assert calc_fee(10_000.0, "buy", s) == pytest.approx(0.0)
    assert calc_fee(10_000.0, "sell", s) == pytest.approx(5.0)  # 卖出 0.05%


def test_transfer_fee_both_sides():
    s = TradingSettings(commission_rate=0.0, commission_min=0.0, commission_fixed=0.0,
                        stamp_tax_rate=0.0, transfer_fee_rate=0.00001)
    assert calc_fee(100_000.0, "buy", s) == pytest.approx(1.0)
    assert calc_fee(100_000.0, "sell", s) == pytest.approx(1.0)


def test_slippage_direction():
    s = TradingSettings(slippage_bps=1.0)
    assert apply_slippage(100.0, "buy", s) == pytest.approx(100.01)
    assert apply_slippage(100.0, "cover", s) == pytest.approx(100.01)
    assert apply_slippage(100.0, "sell", s) == pytest.approx(99.99)
    assert apply_slippage(100.0, "short", s) == pytest.approx(99.99)


def test_trade_fee_recorded():
    b = make_broker(commission_rate=0.00025, commission_min=0.0)
    b.register_symbol("X")
    t = b.place_order(Order(ts(1), "X", "buy", 100, 10.0))
    assert t is not None
    assert t.fee == pytest.approx(100 * 10.0 * 0.00025)  # 0.25


# ================= T+1 =================

def test_t1_bought_today_cannot_sell():
    b = make_broker()
    b.register_symbol("X", t_plus="T+1")
    assert b.place_order(Order(ts(1), "X", "buy", 100, 10.0)) is not None
    t2 = b.place_order(Order(ts(1), "X", "sell", 50, 11.0))
    assert t2 is None  # 当日买入不可卖
    assert any("T+1" in w for w in b.warnings)
    assert b.position("X").qty == 100


def test_t1_unlock_next_day():
    b = make_broker()
    b.register_symbol("X", t_plus="T+1")
    b.place_order(Order(ts(1), "X", "buy", 100, 10.0))
    b.settle_day(ts(1))  # 登记当日
    b.settle_day(ts(2))  # 次日解锁
    t2 = b.place_order(Order(ts(2), "X", "sell", 50, 11.0))
    assert t2 is not None
    assert t2.pnl == pytest.approx(50 * (11.0 - 10.0))


def test_t1_cross_weekend():
    b = make_broker()
    b.register_symbol("X", t_plus="T+1")
    b.place_order(Order(ts(1), "X", "buy", 100, 10.0))  # 周一
    b.settle_day(ts(2))
    b.settle_day(ts(3))
    b.settle_day(ts(4))  # 跨到周四
    assert b.place_order(Order(ts(4), "X", "sell", 100, 11.0)) is not None


def test_t0_no_restriction():
    b = make_broker()
    b.register_symbol("X", t_plus="T+0")
    assert b.place_order(Order(ts(1), "X", "buy", 100, 10.0)) is not None
    assert b.place_order(Order(ts(1), "X", "sell", 100, 11.0)) is not None


# ================= 涨跌停（按 symbol 元数据） =================

def test_limit_up_blocks_buy():
    b = make_broker()
    b.register_symbol("X", price_limit_pct=0.10)
    b.mark("X", 100.0, ts(1))  # 第一根：登记 close
    b.mark("X", 100.0, ts(2))  # 第二根：prev_close=100 → 涨停 110
    assert b.place_order(Order(ts(2), "X", "buy", 100, 110.0)) is None
    assert any("涨跌停" in w for w in b.warnings)
    assert b.place_order(Order(ts(2), "X", "buy", 100, 109.0)) is not None


def test_limit_down_blocks_sell():
    b = make_broker()
    b.register_symbol("X", price_limit_pct=0.10)
    b.place_order(Order(ts(1), "X", "buy", 100, 100.0))
    b.mark("X", 100.0, ts(1))
    b.settle_day(ts(1))  # 登记当日
    b.mark("X", 100.0, ts(2))  # prev_close=100 → 跌停 90
    b.settle_day(ts(2))
    assert b.place_order(Order(ts(2), "X", "sell", 100, 90.0)) is None   # 跌停
    assert b.place_order(Order(ts(2), "X", "sell", 100, 91.0)) is not None


def test_no_limit_when_pct_none():
    b = make_broker()
    b.register_symbol("X", price_limit_pct=None)
    b.mark("X", 100.0, ts(1))
    assert b.place_order(Order(ts(1), "X", "buy", 100, 999.0)) is not None


# ================= 做空保证金 / 强平 =================

def test_short_margin_freeze_and_release():
    b = make_broker(allow_short=True, margin_rate=0.5)
    b.register_symbol("X", t_plus="T+0")
    t = b.place_order(Order(ts(1), "X", "short", 100, 100.0))
    assert t is not None
    pos = b.position("X")
    assert pos.qty == -100
    assert pos.frozen_margin == pytest.approx(100 * 100.0 * 0.5)  # 冻结 5000
    # 现金：1,000,000 + 卖出所得 10,000 - 冻结 5,000
    assert b.cash == pytest.approx(1_000_000.0 + 10_000.0 - 5_000.0)
    # 平空释放冻结
    t2 = b.place_order(Order(ts(1), "X", "cover", 100, 100.0))
    assert t2 is not None
    assert b.position("X") is None or b.position("X").qty == 0
    assert b.cash == pytest.approx(1_000_000.0)  # 无价格变动 → 回到初始（费用为 0）


def test_margin_call_forced_cover():
    b = make_broker(allow_short=True, margin_rate=0.5, margin_call_threshold=0.8)
    b.register_symbol("X", t_plus="T+0")
    b.place_order(Order(ts(1), "X", "short", 100, 100.0))  # 冻结 5000，阈值 4000
    b.mark("X", 59.0, ts(2))  # 浮亏 (100-59)*100=4100 > 4000 → 强制回补
    assert b.position("X") is None or b.position("X").qty == 0
    covers = [t for t in b.trades if t.side == "cover"]
    assert covers and covers[-1].reason.startswith("强制回补")
    assert any(e.event_type == "margin_call" for e in b.risk.events)


def test_short_insufficient_margin_blocked():
    b = make_broker(allow_short=True, margin_rate=1.0, max_exposure_pct=10.0)  # 全额保证金
    b.register_symbol("X", t_plus="T+0")
    # 1,000,000 资金 → 只能做空约 1,000,000 面值
    assert b.place_order(Order(ts(1), "X", "short", 2_000_000, 1.0)) is None
    assert any("保证金不足" in w for w in b.warnings)


# ================= lot =================

def test_lot_size_floor_on_buy():
    b = make_broker()
    b.register_symbol("X", t_plus="T+0", lot_size=100)
    t = b.place_order(Order(ts(1), "X", "buy", 250, 10.0))
    assert t is not None
    assert t.qty == 200  # floor(250/100)*100


def test_sell_allows_odd_lot():
    b = make_broker()
    b.register_symbol("X", t_plus="T+0", lot_size=100)
    b.place_order(Order(ts(1), "X", "buy", 250, 10.0))  # 实际成交 200（lot 取整）
    t = b.place_order(Order(ts(1), "X", "sell", 250, 11.0))
    assert t is not None
    assert t.qty == 200  # 最多卖出全部持仓（含 0 股零头）
