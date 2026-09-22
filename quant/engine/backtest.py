"""回测引擎：向量化（快）与事件驱动（准，与模拟盘共用 Broker）。

设计要点（docs/BACKTEST_ENGINE_DESIGN.md）：
- 信号层一致：两种引擎都用同一套 generate_signals 预计算信号序列；
  差异只在执行层（向量化近似 vs 事件驱动逐笔撮合）。
- 事件驱动：默认成交时点 next_open（t 收盘信号 → t+1 开盘成交，避免未来函数）。
- 向量化：仅支持单标的（共享现金的多标的向量化语义不清，文档注明）。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from quant.core.models import BacktestResult, Order, PortfolioConfig, T1Ledger, Trade
from quant.engine.broker import Broker, apply_slippage, calc_fee
from quant.engine.metrics import compute_metrics, periods_for
from quant.portfolio.portfolio import StrategyPortfolio
from quant.strategies.base import BaseStrategy, StrategyContext


def _is_suspended(row: pd.Series) -> bool:
    v = row.get("volume")
    return v is None or (isinstance(v, float) and np.isnan(v)) or v == 0


class BacktestEngine(ABC):
    engine_type: str = "base"

    def __init__(self, settings, portfolio: StrategyPortfolio):
        self.settings = settings
        self.portfolio = portfolio
        self._in_per_strategy = False

    @abstractmethod
    def run(self, symbol_bars: dict[str, pd.DataFrame],
            strategies: list[BaseStrategy],
            symbol_meta: dict[str, dict] | None = None,
            benchmark_bars: pd.DataFrame | None = None) -> BacktestResult: ...

    # ---- 共享工具 ----
    def _precompute_signals(self, symbol_bars, strategies) -> dict[tuple[str, str], pd.DataFrame]:
        """信号层预计算：dict[(strategy_key, symbol)] → 信号 DataFrame（与 bars 对齐）。"""
        out: dict[tuple[str, str], pd.DataFrame] = {}
        for symbol, bars in symbol_bars.items():
            for strat in strategies:
                ctx = StrategyContext(symbol, self.settings.timeframe, bars,
                                      extra={"initial_capital": self.settings.initial_capital})
                df = strat.generate_signals(bars, ctx)
                if df is None or df.empty:
                    continue
                out[(strat.key, symbol)] = df
        return out

    def _register_symbols(self, broker: Broker, symbol_meta: dict | None) -> None:
        for symbol, meta in (symbol_meta or {}).items():
            broker.register_symbol(
                symbol,
                price_limit_pct=meta.get("price_limit_pct"),
                t_plus=meta.get("t_plus", "T+1"),
                lot_size=meta.get("lot_size", 1),
            )

    def _benchmark_series(self, symbol_bars: dict[str, pd.DataFrame],
                          index: pd.DatetimeIndex,
                          benchmark_bars: pd.DataFrame | None = None) -> pd.Series:
        """基准（buy&hold）：benchmark_symbol 的 close 归一化 × 初始资金。

        优先使用独立传入的 benchmark_bars（与交易标解耦，避免基准被当作交易标的）；
        否则回退到 symbol_bars 中的同名字段；再否则**默认用第一个交易标的本身作为基准**。
        """
        bsym = self.settings.benchmark_symbol
        close = None
        if benchmark_bars is not None and not benchmark_bars.empty:
            close = benchmark_bars["close"]
        elif bsym in symbol_bars:
            close = symbol_bars[bsym]["close"]
        if close is None or len(close) == 0 or close.iloc[0] == 0:
            # 基准缺失/为空 → 默认用第一个交易标的（buy&hold）作为基准
            for bars in symbol_bars.values():
                c = bars["close"]
                if len(c) and c.iloc[0] != 0:
                    return (c / c.iloc[0] * self.settings.initial_capital).reindex(index).ffill()
            return pd.Series(np.nan, index=index)
        return (close / close.iloc[0] * self.settings.initial_capital).reindex(index).ffill()

    @staticmethod
    def _buyhold_series(symbol_bars: dict[str, pd.DataFrame],
                        index: pd.DatetimeIndex) -> pd.Series:
        """等权买入持有全部交易标的：各标的 close 归一化后等权平均 × 初始资金。"""
        norms = []
        for bars in symbol_bars.values():
            close = bars["close"].reindex(index).ffill()
            if len(close) and close.iloc[0] > 0:
                norms.append(close / close.iloc[0])
        if not norms:
            return pd.Series(np.nan, index=index)
        return pd.concat(norms, axis=1).mean(axis=1)

    @staticmethod
    def _symbol_price_series(symbol_bars: dict[str, pd.DataFrame],
                             index: pd.DatetimeIndex) -> pd.Series:
        """第一个交易标的的收盘价（用于"标的股价走势"对比）。"""
        for bars in symbol_bars.values():
            return bars["close"].reindex(index).ffill()
        return pd.Series(np.nan, index=index)

    def _attach_compare_series(self, equity_df: pd.DataFrame,
                               symbol_bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """在权益曲线上附加对比列：buyhold（等权买入持有权益）与 symbol_price（首标的收盘价）。

        equity_df 缺失时（空结果）返回原样；对比列以权益曲线时间为准。
        """
        if equity_df is None or equity_df.empty or not symbol_bars:
            return equity_df
        index = equity_df.index
        equity_df = equity_df.copy()
        equity_df["buyhold"] = (
            self._buyhold_series(symbol_bars, index)
            * self.settings.initial_capital
        ).reindex(index)
        equity_df["symbol_price"] = self._symbol_price_series(symbol_bars, index).reindex(index)
        return equity_df

    def _order_side(self, delta_qty: float, cur_qty: float) -> str:
        if delta_qty > 0:
            return "cover" if cur_qty < 0 else "buy"
        return "sell" if cur_qty > 0 else "short"

    def _per_strategy_metrics(self, symbol_bars, strategies, symbol_meta,
                              benchmark_bars: pd.DataFrame | None = None) -> dict[str, dict]:
        """每策略独立回测指标（同引擎、同设置、单策略），供策略对比。"""
        if self._in_per_strategy:
            return {}
        out: dict[str, dict] = {}
        saved_cfg = self.portfolio.config
        self._in_per_strategy = True
        try:
            for strat in strategies:
                try:
                    self.portfolio.config = PortfolioConfig(
                        mode="signal_merge", weights={strat.key: 100.0},
                        signal_threshold=self.settings.signal_threshold,
                        confidence_weighted=self.portfolio.config.confidence_weighted,
                    )
                    r = self.run(symbol_bars, [strat], symbol_meta, benchmark_bars)
                    out[strat.key] = r.metrics
                except Exception as e:  # noqa: BLE001
                    out[strat.key] = {"error": str(e)}
        finally:
            self.portfolio.config = saved_cfg
            self._in_per_strategy = False
        return out

    def _tag_strategy_names(self, trades: list[Trade], names: dict[str, str]) -> None:
        for t in trades:
            t.strategy_name = names.get(t.strategy_key, "")

    def _build_result(self, engine, symbols, strategies, symbol_bars, index,
                      equity_records, combined_records, position_records,
                      warnings, trades, symbol_meta, benchmark_bars=None) -> BacktestResult:
        settings = self.settings
        equity_df = pd.DataFrame(equity_records, columns=["ts", "equity", "cash",
                                                          "position_value", "exposure",
                                                          "drawdown", "benchmark"])
        if len(equity_df):
            equity_df = equity_df.set_index("ts")
        # 附加对比列（buyhold / symbol_price），供 UI 画归一化对比与指标计算
        equity_df = self._attach_compare_series(equity_df, symbol_bars)
        combined_signal = {}
        for sym in symbols:
            rows = combined_records.get(sym, [])
            if rows:
                combined_signal[sym] = pd.DataFrame(
                    rows, columns=["ts", "target_weight", "reason", "confidence"]).set_index("ts")
        pos_df = pd.DataFrame()
        if position_records:
            pos_df = pd.DataFrame(position_records).set_index("ts").fillna(0.0)

        metrics = compute_metrics(equity_df, trades, periods_for(settings.timeframe), rf=0.0)
        per_strategy = self._per_strategy_metrics(symbol_bars, strategies, symbol_meta, benchmark_bars)
        return BacktestResult(
            engine=engine,
            timeframe=settings.timeframe,
            symbols=symbols,
            start=index.min() if len(index) else None,
            end=index.max() if len(index) else None,
            metrics=metrics,
            per_strategy_metrics=per_strategy,
            equity=equity_df,
            trades=trades,
            positions_timeline=pos_df,
            combined_signal=combined_signal,
            settings_snapshot={
                "settings": settings.model_dump(),
                "portfolio": self.portfolio.config.model_dump(),
            },
            warnings=warnings,
        )

    def _rebalance_due(self, ts) -> bool:
        """再平衡日判断（daily/weekly/monthly；首日视为到期）。"""
        freq = self.settings.rebalance_freq
        if freq == "none":
            return False
        if freq == "daily":
            key = ts.date().isoformat()
        elif freq == "weekly":
            iso = ts.isocalendar()
            key = f"{iso.year}-W{iso.week}"
        else:  # monthly
            key = ts.strftime("%Y-%m")
        if self._last_rebalance_key != key:
            self._last_rebalance_key = key
            return True
        return False

    @staticmethod
    def _union_index(symbol_bars) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex([])
        for bars in symbol_bars.values():
            idx = idx.union(bars.index)
        return idx.sort_values()


class EventDrivenBacktestEngine(BacktestEngine):
    """事件驱动：逐 bar 完整撮合（费用/滑点/涨跌停/T+1/lot/保证金/风控）。"""

    engine_type = "event_driven"

    def run(self, symbol_bars, strategies, symbol_meta=None, benchmark_bars=None):
        t0 = time.perf_counter()
        settings = self.settings
        symbols = list(symbol_bars)
        signals = self._precompute_signals(symbol_bars, strategies)
        names = {s.key: s.name for s in strategies}

        subs = self.portfolio.split(strategies)
        if not subs:
            raise ValueError("无可交易策略（权重全为 0）")

        if self.portfolio.config.mode == "capital_split":
            result = self._run_capital_split(symbol_bars, strategies, signals, subs,
                                             symbol_meta, names, benchmark_bars)
        else:
            result = self._run_combined(symbol_bars, strategies, signals, subs[0],
                                        symbol_meta, names, benchmark_bars)
        result.metrics["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        return result

    # ---- signal_merge / equal ----
    def _run_combined(self, symbol_bars, strategies, signals, sub, symbol_meta, names,
                      benchmark_bars=None):
        settings = self.settings
        broker = Broker(settings)
        self._register_symbols(broker, symbol_meta)
        index = self._union_index(symbol_bars)
        benchmark = self._benchmark_series(symbol_bars, index, benchmark_bars)

        pending: list[Order] = []  # next_open 挂单
        next_allowed: dict[str, int] = {}
        last_target: dict[str, float] = {}  # 上次执行的目标仓位（信号未变不追涨杀跌）
        self._last_rebalance_key: str | None = None
        spacing = max(settings.min_trade_interval_bars, 1) + settings.cooldown_bars

        equity_records, position_records = [], []
        combined_records: dict[str, list] = {s: [] for s in symbol_bars}
        running_max = -1e18

        for i, ts in enumerate(index):
            # 1. 执行上一 bar 挂单（next_open → 本 bar 开盘价）
            remaining = []
            for order in pending:
                bars = symbol_bars.get(order.symbol)
                if bars is None or ts not in bars.index:
                    remaining.append(order)  # 该标的尚无 bar，继续等待
                    continue
                row = bars.loc[ts]
                if _is_suspended(row):
                    broker.warnings.append(f"{order.symbol}: 停牌，挂单作废 {order.side}")
                    continue
                order.timestamp = ts  # 成交时间 = 执行 bar
                order.ref_price = float(row["open"])
                broker.place_order(order)
            pending = remaining

            # 2. 盯市
            for symbol, bars in symbol_bars.items():
                if ts in bars.index:
                    broker.mark(symbol, float(bars.loc[ts, "close"]), ts)

            # 3. 风控权益更新
            eq_now = broker.equity()
            broker.risk.on_equity_update(ts, eq_now, broker)

            # 4. 信号与订单
            for symbol, bars in symbol_bars.items():
                if ts not in bars.index or broker.risk.halted():
                    continue
                row = bars.loc[ts]
                sigs = {sk: signals[(sk, symbol)] for sk in sub.strategy_keys if (sk, symbol) in signals}
                combined = self.portfolio.combine_at(sigs, ts, sub.strategy_keys)
                combined_records[symbol].append(
                    (ts, combined.target_weight, combined.reason, combined.confidence))

                pos = broker.position(symbol)
                cur_qty = pos.qty if pos else 0.0
                cur_weight = (cur_qty * float(row["close"])) / eq_now if eq_now > 0 else 0.0
                # 交易触发：信号目标变化（或到再平衡日），而非权重漂移
                intended = combined.target_weight
                prev_intended = last_target.get(symbol, 0.0)
                if not self._rebalance_due(ts) and abs(intended - prev_intended) <= settings.signal_threshold:
                    continue
                if i < next_allowed.get(symbol, -10**9):
                    continue
                if _is_suspended(row):
                    continue

                delta = intended - cur_weight
                target_qty = intended * eq_now / float(row["close"])
                cap_qty = settings.max_position_pct * eq_now / float(row["close"])
                target_qty = float(np.clip(target_qty, -cap_qty, cap_qty))
                if not settings.allow_short:
                    target_qty = max(target_qty, 0.0)
                delta_qty = target_qty - cur_qty
                if abs(delta_qty) < 1e-9:
                    continue
                side = self._order_side(delta_qty, cur_qty)
                order = Order(ts, symbol, side, abs(delta_qty), float(row["close"]),
                              reason=combined.reason or "组合信号", strategy_key="__combined__")
                if settings.execution == "next_open":
                    pending.append(order)  # 下一根 bar 开盘成交
                    next_allowed[symbol] = i + spacing
                    last_target[symbol] = intended
                else:
                    order.ref_price = float(row["close"])
                    if broker.place_order(order) is not None:
                        next_allowed[symbol] = i + spacing
                        last_target[symbol] = intended

            # 5. T+1 结算
            broker.settle_day(ts)

            # 6. 权益快照
            eq = broker.equity()
            running_max = max(running_max, eq)
            dd = 1.0 - eq / running_max if running_max > 0 else 0.0
            pv = sum(p.market_value() for p in broker.positions.values())
            equity_records.append((ts, eq, broker.cash, pv, broker.exposure(), dd,
                                   benchmark.loc[ts] if ts in benchmark.index else np.nan))
            position_records.append({"ts": ts, **{s: (broker.position(s).qty if broker.position(s) else 0.0)
                                                  for s in symbol_bars}})

        if pending:
            broker.warnings.append(f"尾单丢弃 {len(pending)} 笔（无下一根 bar 成交）")
        for ev in broker.risk.events:
            broker.warnings.append(f"[risk] {ev.event_type}: {ev.detail}")

        self._tag_strategy_names(broker.trades, names)
        return self._build_result(
            "event_driven", list(symbol_bars.keys()), strategies, symbol_bars, index,
            equity_records, combined_records, position_records,
            list(dict.fromkeys(broker.warnings)), broker.trades, symbol_meta)

    # ---- capital_split：每策略独立子账户（共享 T1Ledger） ----
    def _run_capital_split(self, symbol_bars, strategies, signals, subs, symbol_meta, names,
                           benchmark_bars=None):
        settings = self.settings
        t1 = T1Ledger()
        index = self._union_index(symbol_bars)
        benchmark = self._benchmark_series(symbol_bars, index, benchmark_bars)

        brokers = [Broker(settings, initial_capital=settings.initial_capital * sub.capital_pct,
                          t1_ledger=t1) for sub in subs]
        for b in brokers:
            self._register_symbols(b, symbol_meta)

        pending: list[list[Order]] = [[] for _ in subs]
        next_allowed: list[dict[str, int]] = [{} for _ in subs]
        last_target: list[dict[str, float]] = [{} for _ in subs]
        self._last_rebalance_key: str | None = None
        spacing = max(settings.min_trade_interval_bars, 1) + settings.cooldown_bars

        equity_records, position_records = [], []
        combined_records: dict[str, list] = {}
        running_max = -1e18

        for i, ts in enumerate(index):
            # 1. 执行挂单
            for bi, b in enumerate(brokers):
                remaining = []
                for order in pending[bi]:
                    bars = symbol_bars.get(order.symbol)
                    if bars is None or ts not in bars.index:
                        remaining.append(order)
                        continue
                    row = bars.loc[ts]
                    if _is_suspended(row):
                        b.warnings.append(f"{order.symbol}: 停牌，挂单作废")
                        continue
                    order.timestamp = ts  # 成交时间 = 执行 bar
                    order.ref_price = float(row["open"])
                    b.place_order(order)
                pending[bi] = remaining

            # 2. 盯市 + 3. 风控
            for b in brokers:
                for symbol, bars in symbol_bars.items():
                    if ts in bars.index:
                        b.mark(symbol, float(bars.loc[ts, "close"]), ts)
                b.risk.on_equity_update(ts, b.equity(), b)

            # 4. 信号与订单（每子账户只交易自己的策略）
            for bi, (b, sub) in enumerate(zip(brokers, subs)):
                sk = sub.strategy_keys[0]
                for symbol, bars in symbol_bars.items():
                    if ts not in bars.index or b.risk.halted():
                        continue
                    row = bars.loc[ts]
                    sigs = {sk: signals[(sk, symbol)]} if (sk, symbol) in signals else {}
                    combined = self.portfolio.combine_at(sigs, ts, sub.strategy_keys)
                    combined_records.setdefault(symbol, []).append(
                        (ts, combined.target_weight, combined.reason, combined.confidence))

                    pos = b.position(symbol)
                    cur_qty = pos.qty if pos else 0.0
                    eq_now = b.equity()
                    cur_weight = (cur_qty * float(row["close"])) / eq_now if eq_now > 0 else 0.0
                    intended = combined.target_weight
                    prev_intended = last_target[bi].get(symbol, 0.0)
                    if not self._rebalance_due(ts) and abs(intended - prev_intended) <= settings.signal_threshold:
                        continue
                    if i < next_allowed[bi].get(symbol, -10**9):
                        continue
                    if _is_suspended(row):
                        continue
                    delta = intended - cur_weight
                    target_qty = intended * eq_now / float(row["close"])
                    cap_qty = settings.max_position_pct * eq_now / float(row["close"])
                    target_qty = float(np.clip(target_qty, -cap_qty, cap_qty))
                    if not settings.allow_short:
                        target_qty = max(target_qty, 0.0)
                    delta_qty = target_qty - cur_qty
                    if abs(delta_qty) < 1e-9:
                        continue
                    side = self._order_side(delta_qty, cur_qty)
                    order = Order(ts, symbol, side, abs(delta_qty), float(row["close"]),
                                  reason=combined.reason or "子账户信号", strategy_key=sk)
                    if settings.execution == "next_open":
                        pending[bi].append(order)
                        next_allowed[bi][symbol] = i + spacing
                        last_target[bi][symbol] = intended
                    else:
                        order.ref_price = float(row["close"])
                        if b.place_order(order) is not None:
                            next_allowed[bi][symbol] = i + spacing
                            last_target[bi][symbol] = intended

            for b in brokers:
                b.settle_day(ts)

            eq = sum(b.equity() for b in brokers)
            running_max = max(running_max, eq)
            dd = 1.0 - eq / running_max if running_max > 0 else 0.0
            pv = sum(p.market_value() for b in brokers for p in b.positions.values())
            cash = sum(b.cash for b in brokers)
            exposure = (sum(abs(p.market_value()) for b in brokers for p in b.positions.values()) / eq
                        if eq > 0 else 0.0)
            equity_records.append((ts, eq, cash, pv, exposure, dd,
                                   benchmark.loc[ts] if ts in benchmark.index else np.nan))
            pos_map: dict[str, float] = {}
            for b in brokers:
                for s, p in b.positions.items():
                    pos_map[s] = pos_map.get(s, 0.0) + p.qty
            position_records.append({"ts": ts, **pos_map})

        trades = [t for b in brokers for t in b.trades]
        warnings = []
        for b in brokers:
            warnings.extend(b.warnings)
            warnings.extend(f"[risk] {e.event_type}: {e.detail}" for e in b.risk.events)
        self._tag_strategy_names(trades, names)
        return self._build_result(
            "event_driven", list(symbol_bars.keys()), strategies, symbol_bars, index,
            equity_records, combined_records, position_records,
            list(dict.fromkeys(warnings)), trades, symbol_meta)


class VectorizedBacktestEngine(BacktestEngine):
    """向量化：信号层与事件驱动一致；执行层近似（无 T+1/涨跌停/lot/保证金，费用按笔计算）。

    仅支持单标的（多标的请用事件驱动）。
    """

    engine_type = "vectorized"

    def run(self, symbol_bars, strategies, symbol_meta=None, benchmark_bars=None):
        t0 = time.perf_counter()
        if len(symbol_bars) != 1:
            raise ValueError("向量化引擎仅支持单标的；多标的请用事件驱动引擎")
        settings = self.settings
        symbol, bars = next(iter(symbol_bars.items()))
        signals = self._precompute_signals(symbol_bars, strategies)
        combined = self.portfolio.combine_series(
            {sk: signals[(sk, symbol)] for sk in [s.key for s in strategies] if (sk, symbol) in signals})
        index = bars.index
        n = len(bars)
        benchmark = self._benchmark_series(symbol_bars, index, benchmark_bars)

        # 阈值 + 间隔过滤（逐点，纯 Python 足够快）
        target_exec = np.zeros(n)
        last_trade = -10**9
        spacing = max(settings.min_trade_interval_bars, 1) + settings.cooldown_bars
        for i in range(n):
            t = float(combined["target_weight"].iloc[i])
            prev = target_exec[i - 1] if i > 0 else 0.0
            if abs(t - prev) > settings.signal_threshold and i - last_trade >= spacing:
                target_exec[i] = t
                last_trade = i
            else:
                target_exec[i] = prev

        # 资金/撮合近似（与事件驱动同语义：t 收盘决策 → t+1 开盘成交；
        # 目标数量只在信号变化时计算一次，避免权重漂移碎单）
        cash = float(settings.initial_capital)
        qty = 0.0
        avg = 0.0
        opened_ts = None
        trades: list[Trade] = []
        equity = np.zeros(n)
        exposure = np.zeros(n)
        drawdown = np.zeros(n)
        close = bars["close"].to_numpy(dtype=float)
        open_ = bars["open"].to_numpy(dtype=float)
        pending_qty = 0.0
        pending_side: str | None = None
        pending_ts = None
        last_exec_target: float | None = None
        exec_qty = 0.0

        for i in range(n):
            ts = index[i]
            # 执行上一决策的挂单（next_open → 本 bar 开盘价；成交时间戳 = 执行 bar）
            if pending_qty != 0 and i > 0 and pending_side:
                ref = float(open_[i])
                fill = apply_slippage(ref, pending_side, settings)
                fill_qty = abs(pending_qty)
                amount = fill * fill_qty
                fee = calc_fee(amount, pending_side, settings)
                if pending_side in ("buy", "cover"):
                    if qty >= 0:
                        new_qty = qty + fill_qty
                        avg = (avg * qty + fill * fill_qty) / new_qty if new_qty else fill
                        qty = new_qty
                        if opened_ts is None:
                            opened_ts = ts
                        trades.append(self._vtrade(ts, symbol, "buy", fill, fill_qty,
                                                   amount, fee, 0.0, None))
                    else:  # 平空
                        realized = (avg - fill) * min(abs(qty), fill_qty)
                        trades.append(self._vtrade(ts, symbol, "cover", fill,
                                                   fill_qty, amount, fee, realized, opened_ts))
                        qty += fill_qty
                        if abs(qty) < 1e-9:
                            qty = 0.0
                    cash -= amount + fee
                elif pending_side == "short":
                    if qty <= 0:
                        new_qty = qty - fill_qty
                        avg = (avg * abs(qty) + fill * fill_qty) / abs(new_qty) if new_qty else fill
                        qty = new_qty
                        if opened_ts is None:
                            opened_ts = ts
                        trades.append(self._vtrade(ts, symbol, "short", fill, fill_qty,
                                                   amount, fee, 0.0, None))
                    else:  # 开空时反而有多头（兜底：视为平多）
                        realized = (fill - avg) * min(qty, fill_qty)
                        trades.append(self._vtrade(ts, symbol, "sell", fill,
                                                   fill_qty, amount, fee, realized, opened_ts))
                        qty -= fill_qty
                    cash -= amount + fee
                else:  # sell（平多）
                    realized = (fill - avg) * fill_qty
                    trades.append(self._vtrade(ts, symbol, "sell", fill,
                                               fill_qty, amount, fee, realized, opened_ts))
                    qty -= fill_qty
                    cash += amount - fee
                    if abs(qty) < 1e-9:
                        qty = 0.0
                        avg = 0.0
                        opened_ts = None
                pending_qty = 0.0
                pending_side = None

            # 决策：信号目标变化时重算一次目标数量（t 收盘 → 下一 bar 开盘成交）
            if i == 0 or float(target_exec[i]) != last_exec_target:
                last_exec_target = float(target_exec[i])
                eq_now = equity[i - 1] if i > 0 else float(settings.initial_capital)
                if close[i] > 0:
                    exec_qty = float(np.clip(target_exec[i], -settings.max_position_pct,
                                             settings.max_position_pct)) * eq_now / close[i]
                else:
                    exec_qty = 0.0
                if not settings.allow_short:
                    exec_qty = max(exec_qty, 0.0)
            delta_qty = exec_qty - qty
            if abs(delta_qty) > 1e-9:
                pending_qty = delta_qty
                pending_side = self._order_side(delta_qty, qty)
                pending_ts = ts

            # 盯市与快照
            eq = cash + qty * close[i]
            equity[i] = eq
            exposure[i] = abs(qty * close[i]) / eq if eq > 0 else 0.0
            running = np.max(equity[: i + 1])
            drawdown[i] = 1.0 - eq / running if running > 0 else 0.0

        equity_df = pd.DataFrame({
            "equity": equity, "cash": cash, "position_value": qty * close[-1],
            "exposure": exposure, "drawdown": drawdown,
            "benchmark": benchmark.values,
        }, index=index)
        equity_df = self._attach_compare_series(equity_df, symbol_bars)
        metrics = compute_metrics(equity_df, trades, periods_for(settings.timeframe), rf=0.0)
        per_strategy = self._per_strategy_metrics(symbol_bars, strategies, symbol_meta, benchmark_bars)
        combined_df = pd.DataFrame({
            "target_weight": combined["target_weight"].values,
            "reason": combined["reason"].values,
            "confidence": combined["confidence"].values,
        }, index=index)
        result = BacktestResult(
            engine="vectorized",
            timeframe=settings.timeframe,
            symbols=[symbol],
            start=index.min(),
            end=index.max(),
            metrics=metrics,
            per_strategy_metrics=per_strategy,
            equity=equity_df,
            trades=trades,
            positions_timeline=pd.DataFrame({"qty": target_exec}, index=index),
            combined_signal={symbol: combined_df},
            settings_snapshot={
                "settings": settings.model_dump(),
                "portfolio": self.portfolio.config.model_dump(),
            },
            warnings=["向量化引擎未模拟 T+1/涨跌停/lot/保证金/风控（仅近似，见文档）"],
        )
        result.metrics["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        return result

    def _vtrade(self, ts, symbol, side, fill, qty, amount, fee, pnl, opened_ts) -> Trade:
        holding = None
        if opened_ts is not None:
            holding = float((ts.date() - opened_ts.date()).days)
        return Trade(
            timestamp=ts, symbol=symbol, side=side, price=fill, qty=qty,
            amount=amount, fee=fee, slippage_cost=0.0,  # 滑点已计入 fill 价
            pnl=pnl, reason="vectorized", strategy_key="",
            entry_price=None, exit_price=None, holding_days=holding,
        )
