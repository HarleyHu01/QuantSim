"""模拟盘引擎：APScheduler 调度 + 与回测共用 Broker。

调度策略（docs/BACKTEST_ENGINE_DESIGN.md §4）：
- 分钟级（1m/5m/15m/30m/1h/2h/4h）：interval 轮询（默认比 K 线周期多 5~60s）
- 日线/周线/月线（1d/1w/1M）：cron 按标的时区（A股 Asia/Shanghai 15:30、美股 America/New_York 16:30、crypto 00:05 UTC）
- 再平衡：按 rebalance_freq cron

组合模式：
- signal_merge / equal：单 Broker，多策略合成一次下单
- capital_split：每策略一个子 Broker（现金按 |wᵢ| 比例分配，负权重反向），共享 T1Ledger；
  持仓/成交按 sub_key 落库（paper_positions 唯一键 account_id+sub_key+symbol）

状态管理：账户/持仓/成交/净值/挂单全落库；recover() 恢复 running 账户时
**丢弃 PENDING 挂单**（避免重复成交），各子 Broker 从 paper_positions 按 sub_key 重建。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from quant.core.config import AppConfig
from quant.core.models import Order, PortfolioConfig, Position, T1Ledger, TradingSettings
from quant.data.manager import DataManager
from quant.engine.backtest import _is_suspended
from quant.engine.broker import Broker
from quant.portfolio.portfolio import StrategyPortfolio
from quant.storage.repositories import PaperRepo, SymbolsRepo
from quant.strategies.base import BaseStrategy, StrategyContext
from quant.strategies.registry import create, ensure_loaded

log = logging.getLogger(__name__)

# 分钟级轮询间隔（比 K 线周期多出余量）
DEFAULT_POLL_SECONDS = {
    "1m": 65, "5m": 315, "15m": 930, "30m": 1830,
    "1h": 3660, "2h": 7260, "4h": 14520,
}
# 日线级 cron（按标的时区）
DAILY_CRON = {
    "baostock": ("Asia/Shanghai", 15, 30),  # A股（baostock 主源）
    "akshare": ("Asia/Shanghai", 15, 30),   # A股 ETF/期货（akshare）
    "yfinance": ("America/New_York", 16, 30),
    "ccxt": ("UTC", 0, 5),
}
REBALANCE_CRON = {
    "daily": {"hour": 17, "minute": 0},
    "weekly": {"day_of_week": "mon", "hour": 17, "minute": 0},
    "monthly": {"day": 1, "hour": 17, "minute": 0},
}


@dataclass
class PaperAccountState:
    """运行中的模拟盘账户状态（内存态 + DB 持久化）。"""

    account_id: int
    brokers: list[Broker]  # signal_merge=1 个；capital_split=N 个（每策略一个）
    broker_keys: list[str]  # 每个 broker 对应的策略 key（signal_merge = ["__all__"]）
    strategies: list[BaseStrategy]
    portfolio: StrategyPortfolio
    symbols: list[str]
    timeframe: str
    settings: TradingSettings
    poll_seconds: int | None = None
    pending: list[tuple[int, Order]] = field(default_factory=list)  # (broker_idx, order)
    processed_ts: dict[str, pd.Timestamp] = field(default_factory=dict)
    last_target: dict[tuple[int, str], float] = field(default_factory=dict)

    @property
    def capital_split(self) -> bool:
        return len(self.brokers) > 1

    def equity(self) -> float:
        return sum(b.equity() for b in self.brokers)


class PaperTradingEngine:
    def __init__(self, config: AppConfig, data_manager: DataManager,
                 scheduler: AsyncIOScheduler, paper_repo: PaperRepo, symbols_repo: SymbolsRepo):
        self.config = config
        self.dm = data_manager
        self.scheduler = scheduler
        self.repo = paper_repo
        self.symbols_repo = symbols_repo
        self._state: dict[int, PaperAccountState] = {}

    # ================= 启动 / 停止 =================
    def start(self, account_id: int, symbols: list[str], timeframe: str,
              strategy_ids: list[str], params: dict | None = None,
              settings: dict | None = None, portfolio: dict | None = None,
              poll_seconds: int | None = None, name: str = "") -> dict:
        ensure_loaded()
        account = self.repo.get_account(account_id)
        if account is None:
            raise ValueError(f"账户不存在: {account_id}")
        if account["status"] == "running" and account_id in self._state:
            return self.status(account_id)

        cfg = TradingSettings(**settings) if settings else TradingSettings()
        cfg.timeframe = timeframe
        port_cfg = PortfolioConfig(**(portfolio or {}))
        strategies = [create(k, (params or {}).get(k)) for k in strategy_ids]
        st = StrategyPortfolio(port_cfg)

        # 构建 Broker（capital_split → 多子账户，共享 T1Ledger）
        t1 = T1Ledger()
        if port_cfg.mode == "capital_split" and strategies:
            weights = {s.key: port_cfg.weights.get(s.key, 0.0) for s in strategies}
            total_abs = sum(abs(w) for w in weights.values())
            if total_abs <= 0:
                raise ValueError("capital_split 需要非零权重")
            brokers = []
            broker_keys = []
            for s in strategies:
                w = weights[s.key]
                if w == 0:
                    continue
                broker = self._restore_broker(account_id, cfg, sub_key=s.key)
                # 子账户初始资金（无历史净值时）
                if not self.repo.list_equity(account_id, limit=1):
                    broker.cash = cfg.initial_capital * abs(w) / total_abs
                brokers.append(broker)
                broker_keys.append(s.key)
        else:
            brokers = [self._restore_broker(account_id, cfg, sub_key="main")]
            broker_keys = ["__all__"]

        state = PaperAccountState(
            account_id=account_id, brokers=brokers, broker_keys=broker_keys,
            strategies=strategies, portfolio=st, symbols=list(symbols),
            timeframe=timeframe, settings=cfg, poll_seconds=poll_seconds,
        )
        self._register_jobs(state)
        self._state[account_id] = state
        # 持久化运行配置（recover 依赖）
        self.repo.update_settings(account_id, {
            "symbols": symbols, "timeframe": timeframe,
            "strategy_ids": strategy_ids, "params": params or {},
            "settings": cfg.model_dump(), "portfolio": port_cfg.model_dump(),
            "poll_seconds": poll_seconds,
        })
        self.repo.set_status(account_id, "running")
        self.repo.clear_pending(account_id)  # 丢弃残留挂单（不重新撮合）
        log.info("模拟盘启动 account=%s symbols=%s tf=%s mode=%s",
                 account_id, symbols, timeframe, port_cfg.mode)
        return self.status(account_id)

    def stop(self, account_id: int) -> dict:
        if account_id in self._state:
            self._snapshot(self._state[account_id])  # 最终净值快照
            self._state.pop(account_id)
        for job in self.scheduler.get_jobs():
            if job.id.startswith(f"paper_{account_id}_"):
                job.remove()
        self.repo.set_status(account_id, "stopped")
        log.info("模拟盘停止 account=%s", account_id)
        return {"ok": True, "account_id": account_id, "status": "stopped"}

    def status(self, account_id: int) -> dict:
        account = self.repo.get_account(account_id)
        if account is None:
            raise ValueError(f"账户不存在: {account_id}")
        jobs = [j.id for j in self.scheduler.get_jobs() if j.id.startswith(f"paper_{account_id}_")]
        state = self._state.get(account_id)
        out: dict = {
            "account": account,
            "status": account["status"],
            "jobs": jobs,
            "positions": self.repo.list_positions(account_id),
            "trades_tail": self.repo.list_trades(account_id, limit=20),
            "equity_tail": self.repo.list_equity(account_id, limit=5),
        }
        if state is not None:
            out["processed_ts"] = {s: str(ts) for s, ts in state.processed_ts.items()}
            out["equity"] = round(state.equity(), 2)
            out["cash"] = round(sum(b.cash for b in state.brokers), 2)
            out["pending_orders"] = len(state.pending)
            out["mode"] = "capital_split" if state.capital_split else "signal_merge"
            out["sub_brokers"] = [round(b.equity(), 2) for b in state.brokers]
        return out

    # ================= 恢复 =================
    def recover(self) -> None:
        """启动时恢复 status='running' 的账户：重建 Broker、丢弃挂单、重注册任务。"""
        for account in self.repo.list_accounts():
            if account["status"] != "running":
                continue
            aid = account["id"]
            try:
                cfg_dict = self._parse_json(account["settings_json"]) or {}
                settings = TradingSettings(**cfg_dict.get("settings", {}))
                port_cfg = PortfolioConfig(**(cfg_dict.get("portfolio") or {}))
                strategies = [create(k, (cfg_dict.get("params") or {}).get(k))
                              for k in cfg_dict.get("strategy_ids", [])]
                t1 = T1Ledger()
                if port_cfg.mode == "capital_split" and strategies:
                    brokers, keys = [], []
                    for s in strategies:
                        if port_cfg.weights.get(s.key, 0.0) == 0:
                            continue
                        brokers.append(self._restore_broker(aid, settings, sub_key=s.key, t1_ledger=t1))
                        keys.append(s.key)
                else:
                    brokers = [self._restore_broker(aid, settings, sub_key="main", t1_ledger=t1)]
                    keys = ["__all__"]

                pend = self.repo.list_pending(aid)
                if pend:
                    log.warning("账户 %s 丢弃 %d 笔 PENDING 挂单（recover，避免重复成交）", aid, len(pend))
                self.repo.clear_pending(aid)

                state = PaperAccountState(
                    account_id=aid, brokers=brokers, broker_keys=keys,
                    strategies=strategies, portfolio=StrategyPortfolio(port_cfg),
                    symbols=list(cfg_dict.get("symbols", [])),
                    timeframe=cfg_dict.get("timeframe", "1d"),
                    settings=settings, poll_seconds=cfg_dict.get("poll_seconds"),
                )
                self._register_jobs(state)
                self._state[aid] = state
                log.info("模拟盘恢复 account=%s mode=%s", aid, port_cfg.mode)
            except Exception as e:  # noqa: BLE001
                log.error("恢复账户 %s 失败: %s", aid, e)

    # ================= 调度任务 =================
    def _register_jobs(self, state: PaperAccountState) -> None:
        aid = state.account_id
        for symbol in state.symbols:
            job_id = f"paper_{aid}_{symbol}_{state.timeframe}"
            if state.timeframe in DEFAULT_POLL_SECONDS:
                poll = state.poll_seconds or DEFAULT_POLL_SECONDS[state.timeframe]
                self.scheduler.add_job(self._tick, "interval", seconds=poll,
                                       id=job_id, args=[aid, symbol], replace_existing=True)
            else:  # 日线/周线/月线 → cron 按标的时区
                tz, hour, minute = self._cron_time(symbol)
                self.scheduler.add_job(self._tick, "cron", hour=hour, minute=minute, timezone=tz,
                                       id=job_id, args=[aid, symbol], replace_existing=True)
        if state.settings.rebalance_freq in REBALANCE_CRON:
            self.scheduler.add_job(self._rebalance, "cron",
                                   id=f"paper_{aid}_rebalance", replace_existing=True,
                                   **REBALANCE_CRON[state.settings.rebalance_freq])

    def _cron_time(self, symbol: str) -> tuple[str, int, int]:
        meta = self.symbols_repo.get(symbol)
        source = meta["source"] if meta else "yfinance"
        tz, hour, minute = DAILY_CRON.get(source, ("UTC", 0, 5))
        return tz, hour, minute

    # ================= 数据 tick =================
    def _tick(self, account_id: int, symbol: str) -> None:
        state = self._state.get(account_id)
        if state is None:
            return
        try:
            self.dm.update([symbol], [state.timeframe], incremental=True)
        except Exception as e:  # noqa: BLE001
            log.warning("[paper %s] 拉取 %s 失败: %s", account_id, symbol, e)
            return
        bars = self.dm.load_bars(symbol, state.timeframe)
        if bars.empty:
            return
        last_processed = state.processed_ts.get(symbol)
        new = bars if last_processed is None else bars[bars.index > last_processed]
        for ts, row in new.iterrows():
            self._process_bar(state, symbol, row, bars.loc[:ts])
        if len(new):
            state.processed_ts[symbol] = new.index[-1]

    def _process_bar(self, state: PaperAccountState, symbol: str, row: pd.Series,
                     history: pd.DataFrame) -> None:
        """新 bar → 与回测同口径：先执行挂单 → 盯市 → 信号 → 组合 → 撮合 → 快照。"""
        settings = state.settings
        ts = row.name
        aid = state.account_id

        # 1. 执行该 symbol 上一 bar 挂单（next_open → 本 bar 开盘）
        remaining: list[tuple[int, Order]] = []
        for bi, order in state.pending:
            if order.symbol != symbol:
                remaining.append((bi, order))
                continue
            if _is_suspended(row):
                state.brokers[bi].warnings.append(f"{symbol}: 停牌，挂单作废")
                continue
            broker = state.brokers[bi]
            order.timestamp = ts
            order.ref_price = float(row["open"])
            t = broker.place_order(order)
            if t is not None:
                self.repo.add_trade(aid, state.broker_keys[bi], t)
        state.pending = remaining
        self.repo.clear_pending(aid)
        for bi, order in state.pending:
            self.repo.add_pending(aid, order)

        # 2. 盯市 + 3. 风控（全部子账户）
        for bi, broker in enumerate(state.brokers):
            broker.mark(symbol, float(row["close"]), ts)
            broker.settle_day(ts)
            broker.risk.on_equity_update(ts, broker.equity(), broker)
            for ev in broker.risk.events:
                log.warning("[paper %s/%s] %s: %s", aid, state.broker_keys[bi], ev.event_type, ev.detail)

        # 4. 信号 → 组合 → 撮合
        ctx = StrategyContext(symbol, state.timeframe, history)
        strat_by_key = {s.key: s for s in state.strategies}

        if state.capital_split:
            for bi, key in enumerate(state.broker_keys):
                self._trade_for_signal(state, bi, key, symbol, row, ctx, strat_by_key.get(key))
        else:
            sig_dict: dict[str, pd.DataFrame] = {}
            for strat in state.strategies:
                tw, reason, conf = strat.on_bar(row, ctx)
                sig_dict[strat.key] = pd.DataFrame(
                    {"target_weight": [tw], "reason": [reason], "confidence": [conf]}, index=[ts])
            combined = state.portfolio.combine_at(sig_dict, ts)
            self._execute_combined(state, 0, symbol, row, combined, "__combined__")

        # 5. 快照
        self._snapshot(state)

    def _trade_for_signal(self, state, bi, key, symbol, row, ctx, strat) -> None:
        if strat is None:
            return
        tw, reason, conf = strat.on_bar(row, ctx)
        sig = pd.DataFrame({"target_weight": [tw], "reason": [reason], "confidence": [conf]},
                           index=[row.name])
        combined = state.portfolio.combine_at({key: sig}, row.name, [key])
        self._execute_combined(state, bi, symbol, row, combined, key)

    def _execute_combined(self, state: PaperAccountState, bi: int, symbol: str,
                          row: pd.Series, combined, strategy_key: str) -> None:
        """按组合目标撮合（与事件驱动引擎同口径：信号目标变化才交易）。"""
        settings = state.settings
        broker = state.brokers[bi]
        ts = row.name
        pos = broker.position(symbol)
        cur_qty = pos.qty if pos else 0.0
        eq = broker.equity()
        cur_weight = (cur_qty * float(row["close"])) / eq if eq > 0 else 0.0
        intended = combined.target_weight
        prev_intended = state.last_target.get((bi, symbol), 0.0)
        if abs(intended - prev_intended) <= settings.signal_threshold or broker.risk.halted():
            return
        target_qty = intended * eq / float(row["close"])
        cap = settings.max_position_pct * eq / float(row["close"])
        target_qty = float(max(-cap, min(cap, target_qty)))
        if not settings.allow_short:
            target_qty = max(target_qty, 0.0)
        delta_qty = target_qty - cur_qty
        if abs(delta_qty) < 1e-9:
            return
        if delta_qty > 0:
            side = "cover" if cur_qty < 0 else "buy"
        else:
            side = "sell" if cur_qty > 0 else "short"
        order = Order(ts, symbol, side, abs(delta_qty), float(row["close"]),
                      reason=combined.reason or "paper信号", strategy_key=strategy_key)
        if settings.execution == "next_open":
            state.pending.append((bi, order))
            state.last_target[(bi, symbol)] = intended
            self.repo.add_pending(state.account_id, order)
        else:
            t = broker.place_order(order)
            if t is not None:
                state.last_target[(bi, symbol)] = intended
                self.repo.add_trade(state.account_id, state.broker_keys[bi], t)

    def _rebalance(self, account_id: int) -> None:
        """再平衡：把持仓拉回组合目标（简化：按最新 bar 重新对齐）。"""
        state = self._state.get(account_id)
        if state is None:
            return
        for symbol in state.symbols:
            bars = self.dm.load_bars(symbol, state.timeframe)
            if bars.empty:
                continue
            self._process_bar(state, symbol, bars.iloc[-1], bars)

    # ================= 持久化 =================
    def _snapshot(self, state: PaperAccountState) -> None:
        """持仓（按 sub_key）+ 净值落库。"""
        aid = state.account_id
        for bi, broker in enumerate(state.brokers):
            sub_key = state.broker_keys[bi] if state.capital_split else "main"
            for symbol in list(broker.positions.keys()):
                pos = broker.positions[symbol]
                if abs(pos.qty) < 1e-9:
                    broker.positions.pop(symbol, None)
                    self.repo.delete_position(aid, sub_key, symbol)
                    continue
                self.repo.upsert_position(aid, sub_key, pos.to_dict())
        eq = state.equity()
        pv = sum(p.market_value() for b in state.brokers for p in b.positions.values())
        cash = sum(b.cash for b in state.brokers)
        self.repo.add_equity(aid, datetime.now(timezone.utc), eq, cash, pv)

    def _restore_broker(self, account_id: int, settings: TradingSettings,
                        sub_key: str = "main", t1_ledger: T1Ledger | None = None) -> Broker:
        """从 paper_positions 重建 Broker（按 sub_key；现金从最近净值快照恢复）。"""
        broker = Broker(settings, t1_ledger=t1_ledger or T1Ledger())
        equity_rows = self.repo.list_equity(account_id, limit=1)
        if equity_rows:
            broker.cash = float(equity_rows[0]["cash"])
        for p in self.repo.list_positions(account_id, sub_key=sub_key):
            qty = float(p["qty"])
            avg = float(p["avg_price"]) if p.get("avg_price") is not None else 0.0
            broker.positions[p["symbol"]] = Position(
                symbol=p["symbol"], qty=qty, avg_price=avg,
                last_price=float(p.get("last_price") or avg),
                realized_pnl=0.0, opened_at=None, frozen_margin=0.0)
        return broker

    @staticmethod
    def _parse_json(s: str | None) -> dict:
        if not s:
            return {}
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return {}
