"""Broker：撮合核心（事件驱动回测与模拟盘共用）。

规则：费用/滑点/涨跌停（按 symbol 元数据）/停牌/T+0-T+1（T1Ledger 统一账本）/
lot/资金/做空保证金与强制回补。全部参数来自 TradingSettings。
"""
from __future__ import annotations

import math
from datetime import datetime

from quant.core.models import Order, Position, T1Ledger, Trade, TradingSettings
from quant.engine.risk import RiskManager


def calc_fee(amount: float, side: str, settings: TradingSettings) -> float:
    """佣金 + 印花税 + 过户费。

    - 佣金：commission_fixed>0 按笔固定，否则按比例；两者同设取较大值；再 ≥ commission_min
    - 印花税：仅卖出侧（A股卖出征收）
    - 过户费：买卖双侧
    """
    s = settings
    if s.commission_fixed > 0:
        commission = s.commission_fixed
    else:
        commission = amount * s.commission_rate
    if s.commission_fixed > 0:
        commission = max(commission, amount * s.commission_rate)
    commission = max(commission, s.commission_min)
    stamp = amount * s.stamp_tax_rate if side == "sell" else 0.0
    transfer = amount * s.transfer_fee_rate
    return round(commission + stamp + transfer, 6)


def apply_slippage(price: float, side: str, settings: TradingSettings) -> float:
    """滑点：买/平空加价，卖/开空减价。"""
    bps = settings.slippage_bps / 1e4
    if side in ("buy", "cover"):
        return price * (1 + bps)
    return price * (1 - bps)


class Broker:
    def __init__(self, settings: TradingSettings, risk: RiskManager | None = None,
                 t1_ledger: T1Ledger | None = None, initial_capital: float | None = None):
        self.settings = settings
        self.risk = risk or RiskManager(settings)
        self.t1 = t1_ledger or T1Ledger()
        self.cash: float = initial_capital if initial_capital is not None else settings.initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.warnings: list[str] = []
        # symbol 元数据：price_limit_pct / t_plus / lot_size / prev_close
        self.symbol_meta: dict[str, dict] = {}

    # ---- 元数据 ----
    def register_symbol(self, symbol: str, price_limit_pct: float | None = None,
                        t_plus: str = "T+1", lot_size: int = 1) -> None:
        meta = self.symbol_meta.setdefault(symbol, {})
        if price_limit_pct is not None:
            meta["price_limit_pct"] = price_limit_pct
        meta.setdefault("t_plus", t_plus)
        meta.setdefault("lot_size", lot_size)

    # ---- 盯市与结算 ----
    def mark(self, symbol: str, price: float, ts: datetime) -> None:
        """每日盯市：记录 prev_close 用于涨跌停判定，更新持仓市值，检查强制回补。"""
        meta = self.symbol_meta.setdefault(symbol, {})
        if "close" in meta:
            meta["prev_close"] = meta["close"]
        meta["close"] = price
        pos = self.positions.get(symbol)
        if pos is not None:
            pos.last_price = price
            self._check_margin_call(symbol, price, ts)

    def settle_day(self, ts: datetime) -> None:
        self.t1.settle_day(ts)

    # ---- 权益 ----
    def equity(self) -> float:
        return self.cash + sum(p.market_value() for p in self.positions.values())

    def exposure(self) -> float:
        eq = self.equity()
        if eq <= 0:
            return 0.0
        return sum(abs(p.market_value()) for p in self.positions.values()) / eq

    def active_symbols(self) -> int:
        return len(self.positions)

    def position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    # ---- 撮合 ----
    def place_order(self, order: Order, force: bool = False) -> Trade | None:
        """尝试成交。force=True 用于强制回补（跳过 T+1/涨跌停）。返回 None 表示未成交。"""
        s = self.settings
        meta = self.symbol_meta.get(order.symbol, {})

        # 1. 风控
        if not force:
            check = self.risk.check_order(order.timestamp, order.symbol, order.side,
                                          order.qty, order.ref_price, self)
            if not check.allowed:
                self.warnings.append(f"{order.symbol}: 风控拦截 {check.reason}")
                return None

        if order.ref_price <= 0:
            return None

        # 2. 涨跌停
        if not force and self._limit_blocked(order):
            self.warnings.append(f"{order.symbol}: 涨跌停拦截 {order.side} @ {order.ref_price:.4f}")
            return None

        # 3. 成交价与滑点
        fill = self._fill_price(order)
        slippage_cost = order.qty * abs(fill - order.ref_price)
        amount = fill * order.qty

        # 4. T+1 / lot / 可用数量
        qty = order.qty
        if order.side == "buy":
            lot = int(meta.get("lot_size", 1) or 1)
            qty = math.floor(qty / lot) * lot
            if qty <= 0:
                return None
        elif order.side in ("sell", "cover"):
            if not force:
                avail = self._available_qty(order.symbol, order.side)
                if qty > avail:
                    self.warnings.append(
                        f"{order.symbol}: T+1 限制 请求 {qty:.0f} → 成交 {avail:.0f}")
                    qty = avail
                if qty <= 0:
                    return None
            else:
                pos = self.positions.get(order.symbol)
                qty = min(qty, abs(pos.qty) if pos else 0.0)
                if qty <= 0:
                    return None

        # 5. 资金 / 保证金
        fee = calc_fee(amount, order.side, s)
        if order.side in ("buy", "cover"):
            if self.cash < amount + fee:
                self.warnings.append(f"{order.symbol}: 资金不足 {self.cash:.2f} < {amount + fee:.2f}")
                return None
        elif order.side == "short":  # 做空：冻结保证金
            frozen = amount * s.margin_rate
            if self.cash < frozen + fee:
                self.warnings.append(f"{order.symbol}: 保证金不足 {self.cash:.2f} < {frozen + fee:.2f}")
                return None

        # 6. 执行
        return self._execute(order, qty, fill, amount, fee, slippage_cost)

    # ---- 内部：成交执行 ----
    def _execute(self, order: Order, qty: float, fill: float, amount: float,
                 fee: float, slippage_cost: float) -> Trade:
        s = self.settings
        symbol = order.symbol
        ts = order.timestamp
        pos = self.positions.get(symbol)
        side = order.side

        trade = Trade(
            timestamp=ts, symbol=symbol, side=side, price=fill, qty=qty,
            amount=amount, fee=fee, slippage_cost=slippage_cost, pnl=0.0,
            reason=order.reason, strategy_key=order.strategy_key,
            signal_reason=order.reason,
        )

        if side == "buy":
            if pos is None or pos.qty == 0:
                self.positions[symbol] = Position(symbol, qty, fill, fill, 0.0, ts, 0.0)
            elif pos.qty > 0:
                new_qty = pos.qty + qty
                pos.avg_price = (pos.avg_price * pos.qty + fill * qty) / new_qty
                pos.qty = new_qty
                pos.last_price = fill
            else:  # 空头持仓被 buy 覆盖（引擎一般发 cover，此处兜底）
                realized = (pos.avg_price - fill) * min(abs(pos.qty), qty)
                pos.realized_pnl += realized
                trade.pnl = realized
                trade.entry_price = pos.avg_price
                trade.exit_price = fill
                trade.holding_days = self._holding_days(pos, ts)
                pos.qty += qty
                if pos.qty >= 0:
                    pos.qty = 0.0
                    self.positions.pop(symbol, None)
            self.cash -= amount + fee
            self.t1.today_bought[symbol] = self.t1.today_bought.get(symbol, 0.0) + qty

        elif side == "sell":
            if pos is None or pos.qty <= 0:
                return trade  # 无仓可卖
            realized = (fill - pos.avg_price) * qty
            pos.realized_pnl += realized
            trade.pnl = realized
            trade.entry_price = pos.avg_price
            trade.exit_price = fill
            trade.holding_days = self._holding_days(pos, ts)
            pos.qty -= qty
            self.cash += amount - fee
            if pos.qty <= 0:
                self.positions.pop(symbol, None)

        elif side == "short":
            if pos is None or pos.qty == 0:
                self.positions[symbol] = Position(symbol, -qty, fill, fill, 0.0, ts, 0.0)
            elif pos.qty < 0:
                new_qty = pos.qty - qty
                pos.avg_price = (pos.avg_price * abs(pos.qty) + fill * qty) / abs(new_qty)
                pos.qty = new_qty
                pos.last_price = fill
            else:  # 多头被 short 覆盖（兜底）
                realized = (fill - pos.avg_price) * min(pos.qty, qty)
                pos.realized_pnl += realized
                trade.pnl = realized
                trade.entry_price = pos.avg_price
                trade.exit_price = fill
                trade.holding_days = self._holding_days(pos, ts)
                pos.qty -= qty
                if pos.qty <= 0:
                    self.positions.pop(symbol, None)
            frozen = amount * s.margin_rate
            pos = self.positions.get(symbol)
            if pos is not None:
                pos.frozen_margin += frozen
            self.cash += amount - frozen - fee  # 卖出所得入账，冻结保证金从现金扣
            self.t1.today_shorted[symbol] = self.t1.today_shorted.get(symbol, 0.0) + qty

        elif side == "cover":
            if pos is None or pos.qty >= 0:
                return trade
            realized = (pos.avg_price - fill) * qty
            pos.realized_pnl += realized
            trade.pnl = realized
            trade.entry_price = pos.avg_price
            trade.exit_price = fill
            trade.holding_days = self._holding_days(pos, ts)
            old_abs = abs(pos.qty)
            released = pos.frozen_margin * (qty / old_abs) if old_abs > 0 else 0.0
            pos.frozen_margin -= released
            pos.qty += qty
            self.cash += released - amount - fee  # 买回支出，释放的保证金退回现金
            if pos.qty >= 0:
                self.positions.pop(symbol, None)

        self.trades.append(trade)
        return trade

    @staticmethod
    def _holding_days(pos: Position, ts: datetime) -> float:
        if pos.opened_at is None:
            return 0.0
        return (ts.date() - pos.opened_at.date()).days

    # ---- 内部：规则 ----
    def _fill_price(self, order: Order) -> float:
        return apply_slippage(order.ref_price, order.side, self.settings)

    def _limit_blocked(self, order: Order) -> bool:
        """涨跌停拦截（按 symbol 元数据 price_limit_pct）。"""
        meta = self.symbol_meta.get(order.symbol, {})
        pct = meta.get("price_limit_pct")
        prev_close = meta.get("prev_close")
        if not pct or not prev_close:
            return False
        limit_up = prev_close * (1 + pct)
        limit_down = prev_close * (1 - pct)
        px = order.ref_price
        if order.side in ("buy", "short") and px >= limit_up * 0.999:
            return True  # 涨停：禁买/禁开空
        if order.side in ("sell", "cover") and px <= limit_down * 1.001:
            return True  # 跌停：禁卖/禁平空
        return False

    def _available_qty(self, symbol: str, side: str) -> float:
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        if not self.settings.t_plus_enabled:
            return abs(pos.qty)
        meta = self.symbol_meta.get(symbol, {})
        if meta.get("t_plus") != "T+1":
            return abs(pos.qty)
        if side == "sell":
            return max(pos.qty - self.t1.today_bought.get(symbol, 0.0), 0.0)
        return max(abs(pos.qty) - self.t1.today_shorted.get(symbol, 0.0), 0.0)

    def _check_margin_call(self, symbol: str, price: float, ts: datetime) -> None:
        """空头浮亏超过冻结保证金阈值 → 强制回补。"""
        pos = self.positions.get(symbol)
        if pos is None or pos.qty >= 0 or pos.frozen_margin <= 0:
            return
        loss = (pos.avg_price - price) * abs(pos.qty)
        if loss > pos.frozen_margin * self.settings.margin_call_threshold:
            order = Order(ts, symbol, "cover", abs(pos.qty), price,
                          reason=f"强制回补（保证金亏损 {loss:.2f} > 冻结×{self.settings.margin_call_threshold:.0%}）",
                          strategy_key="risk")
            self.place_order(order, force=True)
            self.risk.add_event(ts, "margin_call", {"symbol": symbol, "loss": round(loss, 2)})
