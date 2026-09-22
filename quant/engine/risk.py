"""风控模块：敞口限制、最大回撤熔断、止损止盈、每日最大亏损、最大持仓数。

事件写 risk_events 表（API 层落库）；Broker 在撮合前调用 check_order，
引擎/模拟盘在每 bar 权益更新后调用 on_equity_update。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from quant.core.models import Order, TradingSettings


@dataclass
class OrderCheck:
    allowed: bool
    reason: str = ""


@dataclass
class RiskEvent:
    ts: datetime
    event_type: str  # max_drawdown_halt | stop_loss | take_profit | daily_loss_limit | margin_call | resume
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ts": self.ts.strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": self.event_type, "detail": self.detail}


class RiskManager:
    def __init__(self, settings: TradingSettings):
        self.settings = settings
        self.events: list[RiskEvent] = []
        self._halted = False          # 回撤熔断状态
        self._halt_reason = ""
        self._peak_equity: float | None = None
        self._day: datetime | None = None
        self._day_start_equity: float | None = None
        self._day_loss_halted = False

    # ---- 事件 ----
    def add_event(self, ts: datetime, event_type: str, detail: dict | None = None) -> None:
        self.events.append(RiskEvent(ts, event_type, detail or {}))

    # ---- 状态 ----
    def halted(self) -> bool:
        return self._halted or self._day_loss_halted

    def resume(self) -> None:
        self._halted = False
        self._halt_reason = ""

    # ---- 订单前检查 ----
    def check_order(self, ts: datetime, symbol: str, side: str, qty: float,
                    price: float, broker) -> OrderCheck:
        if self._halted:
            return OrderCheck(False, f"回撤熔断中: {self._halt_reason}")
        if self._day_loss_halted:
            return OrderCheck(False, "每日最大亏损已达，当日停止交易")
        s = self.settings
        if s.max_positions and broker.active_symbols() >= s.max_positions \
                and symbol not in broker.positions:
            return OrderCheck(False, f"最大持仓数 {s.max_positions} 已达")
        if side in ("buy", "short"):
            eq = broker.equity()
            cur = abs(broker.position(symbol).market_value()) if broker.position(symbol) else 0.0
            add = qty * price
            total = sum(abs(p.market_value()) for p in broker.positions.values())
            if total + add > s.max_exposure_pct * eq:
                return OrderCheck(False, f"总敞口超限 {(total + add) / eq:.1%} > {s.max_exposure_pct:.0%}")
        return OrderCheck(True)

    # ---- 每 bar 权益更新 ----
    def on_equity_update(self, ts: datetime, equity: float, broker) -> list[RiskEvent]:
        s = self.settings
        events: list[RiskEvent] = []

        # 日切换
        if self._day is None or self._day.date() != ts.date():
            self._day = ts
            self._day_start_equity = equity
            self._day_loss_halted = False

        # 1. 最大回撤熔断
        if s.max_drawdown_halt:
            peak = self._peak_equity if self._peak_equity is not None else equity
            self._peak_equity = max(peak, equity)
            dd = 1.0 - equity / self._peak_equity if self._peak_equity else 0.0
            if dd > s.max_drawdown_halt and not self._halted:
                self._halted = True
                self._halt_reason = f"回撤 {dd:.1%} > {s.max_drawdown_halt:.1%}"
                events.append(RiskEvent(ts, "max_drawdown_halt", {"drawdown": round(dd, 4)}))
            elif self._halted and dd < s.max_drawdown_halt - 0.02:
                self._halted = False
                self._halt_reason = ""
                events.append(RiskEvent(ts, "resume", {"drawdown": round(dd, 4)}))

        # 2. 每日最大亏损
        if s.daily_loss_limit_pct and self._day_start_equity:
            loss = equity - self._day_start_equity
            if loss < -s.daily_loss_limit_pct * self._day_start_equity and not self._day_loss_halted:
                self._day_loss_halted = True
                events.append(RiskEvent(ts, "daily_loss_limit",
                                        {"day_loss": round(loss, 2),
                                         "limit": s.daily_loss_limit_pct}))

        # 3. 止损 / 止盈（市价平仓）
        for symbol, pos in list(broker.positions.items()):
            if pos.qty == 0:
                continue
            cost = abs(pos.qty) * pos.avg_price
            if cost <= 0:
                continue
            upnl_pct = pos.unrealized_pnl() / cost
            if s.stop_loss_pct and upnl_pct <= -s.stop_loss_pct:
                self._close_position(broker, symbol, ts, "stop_loss", upnl_pct)
                events.append(RiskEvent(ts, "stop_loss",
                                        {"symbol": symbol, "pnl_pct": round(upnl_pct, 4)}))
            elif s.take_profit_pct and upnl_pct >= s.take_profit_pct:
                self._close_position(broker, symbol, ts, "take_profit", upnl_pct)
                events.append(RiskEvent(ts, "take_profit",
                                        {"symbol": symbol, "pnl_pct": round(upnl_pct, 4)}))
        return events

    def _close_position(self, broker, symbol: str, ts: datetime, reason: str, upnl_pct: float) -> None:
        pos = broker.position(symbol)
        if pos is None or pos.qty == 0:
            return
        side = "sell" if pos.qty > 0 else "cover"
        order = Order(ts, symbol, side, abs(pos.qty), pos.last_price,
                      reason=f"{reason} ({upnl_pct:.1%})", strategy_key="risk")
        broker.place_order(order)
