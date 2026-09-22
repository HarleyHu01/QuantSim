"""核心数据模型与常量。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

# 统一 OHLCV 列约定（DataProvider 输出 + Parquet 存储列）
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
ALL_COLUMNS = ["open", "high", "low", "close", "volume", "adj_factor"]

# 系统支持的周期（统一命名）
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M"]

Side = Literal["buy", "sell", "short", "cover"]


@dataclass
class UpdateResult:
    """一次 (symbol, timeframe) 数据更新的结果。"""

    symbol: str
    timeframe: str
    ok: bool
    source: str | None = None
    rows: int = 0
    start: str | None = None
    end: str | None = None
    last_bar: str | None = None
    error: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "ok": self.ok,
            "source": self.source,
            "rows": self.rows,
            "start": self.start,
            "end": self.end,
            "last_bar": self.last_bar,
            "error": self.error,
            "message": self.message,
        }


@dataclass
class Signal:
    """策略输出信号（历史契约保留）。target_weight ∈ [-1, 1]。"""

    timestamp: datetime
    symbol: str
    target_weight: float
    reason: str = ""
    strength: float = 1.0
    confidence: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """订单（引擎/组合器产生，Broker 消费）。"""

    timestamp: datetime
    symbol: str
    side: Side
    qty: float  # 正数
    ref_price: float  # 执行参考价（Broker 在此价格上叠加滑点）
    reason: str = ""
    strategy_key: str = ""


@dataclass
class Trade:
    """成交记录（落库）。平仓成交由 Broker 回填 entry/exit/holding。"""

    timestamp: datetime
    symbol: str
    side: Side
    price: float
    qty: float
    amount: float
    fee: float
    slippage_cost: float
    pnl: float  # 平仓盈亏（开仓为 0）
    reason: str = ""
    strategy_key: str = ""
    strategy_name: str = ""
    signal_reason: str = ""
    entry_price: float | None = None
    exit_price: float | None = None
    holding_days: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": self.symbol,
            "side": self.side,
            "price": round(self.price, 6),
            "qty": round(self.qty, 6),
            "amount": round(self.amount, 4),
            "fee": round(self.fee, 4),
            "slippage_cost": round(self.slippage_cost, 4),
            "pnl": round(self.pnl, 4),
            "reason": self.reason,
            "strategy_key": self.strategy_key,
            "strategy_name": self.strategy_name,
            "signal_reason": self.signal_reason,
            "entry_price": round(self.entry_price, 6) if self.entry_price is not None else None,
            "exit_price": round(self.exit_price, 6) if self.exit_price is not None else None,
            "holding_days": self.holding_days,
        }


@dataclass
class Position:
    """运行期持仓（Broker 内部）。qty>0 多，qty<0 空。"""

    symbol: str
    qty: float
    avg_price: float
    last_price: float
    realized_pnl: float = 0.0
    opened_at: datetime | None = None
    frozen_margin: float = 0.0  # 做空冻结保证金

    def market_value(self) -> float:
        return self.qty * self.last_price

    def unrealized_pnl(self) -> float:
        if self.qty > 0:
            return (self.last_price - self.avg_price) * self.qty
        return (self.avg_price - self.last_price) * abs(self.qty)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "avg_price": round(self.avg_price, 6),
            "last_price": round(self.last_price, 6),
            "market_value": round(self.market_value(), 4),
            "unrealized_pnl": round(self.unrealized_pnl(), 4),
            "frozen_margin": round(self.frozen_margin, 4),
        }


class T1Ledger:
    """T+1 锁定账本：跨子账户/多策略共享同一持仓的锁定状态。

    today_bought / today_shorted 按日期记录；settle_day 在日期切换时解锁。
    """

    def __init__(self) -> None:
        self.today_bought: dict[str, float] = {}
        self.today_shorted: dict[str, float] = {}
        self._date: date | None = None

    def settle_day(self, ts: datetime) -> None:
        d = ts.date()
        if self._date is not None and d != self._date:
            self.today_bought.clear()
            self.today_shorted.clear()
        self._date = d


class TradingSettings(BaseModel):
    """交易设置（UI「交易设置页」持久化对象）。所有字段有默认值。"""

    initial_capital: float = 1_000_000.0

    # ---- 手续费（A股惯例默认值，可调） ----
    commission_rate: float = 0.00025  # 比例佣金（万2.5，双边）
    commission_fixed: float = 0.0  # 固定每笔（>0 时按笔）
    commission_min: float = 5.0  # 最低佣金（A股 5 元）
    stamp_tax_rate: float = 0.0005  # 卖出印花税（可置 0 关闭）
    transfer_fee_rate: float = 0.00001  # 过户费（双边，可置 0 关闭）
    slippage_bps: float = 1.0  # 滑点（bp）

    # ---- 做空（allow_short=True 时生效） ----
    margin_rate: float = 0.5  # 保证金率：冻结 = |仓位价值| × margin_rate
    margin_call_threshold: float = 0.8  # 浮亏 > 冻结×阈值 → 强制回补

    # ---- 交易频度 ----
    timeframe: str = "1d"
    min_trade_interval_bars: int = 1  # 每 N 根 bar 最多交易一次
    cooldown_bars: int = 0  # 成交后额外冷却 bar 数
    rebalance_freq: str = "none"  # none|daily|weekly|monthly
    signal_threshold: float = 0.0  # 目标仓位变化阈值 θ
    execution: str = "next_open"  # next_open（默认）| same_close

    # ---- 仓位与规则 ----
    max_position_pct: float = 0.3  # 单标的最大仓位（占权益比例）
    max_exposure_pct: float = 1.0  # 总敞口上限
    allow_short: bool = False
    t_plus_enabled: bool = True  # 按标的 t_plus 属性执行 T+0/T+1

    # ---- 风控（None=关闭） ----
    max_drawdown_halt: float | None = None  # 回撤熔断 %
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    daily_loss_limit_pct: float | None = None
    max_positions: int | None = None

    # ---- 基准与年化 ----
    benchmark_symbol: str = "SPY"
    periods_per_year: int = 252  # 按周期自动换算，可覆盖


class PortfolioConfig(BaseModel):
    """组合配置（UI「策略权重」持久化对象）。"""

    mode: str = "signal_merge"  # signal_merge | capital_split | equal
    weights: dict[str, float] = Field(default_factory=dict)  # strategy_key → weight(-100..100)
    signal_threshold: float = 0.0
    confidence_weighted: bool = True  # 合成时按 confidence 加权


@dataclass
class CombinedSignal:
    """单 bar 合成结果（UI 可展示信号分解）。"""

    target_weight: float
    per_strategy_weights: dict[str, float]  # 各策略权重（含符号）
    per_strategy_contribution: dict[str, float]  # wᵢ·sᵢ·confᵢ / Σ|wᵢ·confᵢ|
    confidence: float
    reason: str = ""


@dataclass
class BacktestResult:
    """回测结果（两种引擎统一输出结构）。"""

    engine: str
    timeframe: str
    symbols: list[str]
    start: datetime
    end: datetime
    metrics: dict[str, Any]
    per_strategy_metrics: dict[str, dict[str, Any]]
    equity: pd.DataFrame  # index=ts, cols=[equity, cash, position_value, exposure, drawdown, benchmark]
    trades: list[Trade]
    positions_timeline: pd.DataFrame  # index=ts, cols=symbol（持仓 qty）
    combined_signal: dict[str, pd.DataFrame]  # symbol → [target_weight, reason, confidence]
    settings_snapshot: dict[str, Any]
    warnings: list[str]
    run_id: int | None = None
