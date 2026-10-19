"""轻量仓储层：封装 SQLite 访问。

- SymbolsRepo / MarketDataRepo：数据模块
- BacktestRepo：回测运行与结果持久化
- PaperRepo：模拟盘账户/持仓/成交/净值/挂单
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant.storage.db import get_conn


def _row_to_dict(row) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class SymbolsRepo:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def upsert(
        self,
        symbol: str,
        name: str = "",
        asset_type: str = "stock",
        source: str = "yfinance",
        exchange: str = "",
        lot_size: int = 1,
        price_limit_pct: float | None = None,
        t_plus: str = "T+1",
        enabled: int = 1,
    ) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO symbols(symbol, name, asset_type, source, exchange,
                                    lot_size, price_limit_pct, t_plus, enabled, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?, datetime('now'))
                ON CONFLICT(symbol) DO UPDATE SET
                  name=excluded.name, asset_type=excluded.asset_type, source=excluded.source,
                  exchange=excluded.exchange, lot_size=excluded.lot_size,
                  price_limit_pct=excluded.price_limit_pct, t_plus=excluded.t_plus,
                  enabled=excluded.enabled, updated_at=datetime('now')
                """,
                (symbol, name, asset_type, source, exchange, lot_size, price_limit_pct, t_plus, enabled),
            )

    def get(self, symbol: str) -> dict[str, Any] | None:
        with get_conn(self.db_path) as conn:
            row = conn.execute("SELECT * FROM symbols WHERE symbol = ?", (symbol,)).fetchone()
        return _row_to_dict(row)

    def list_all(self) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM symbols ORDER BY symbol").fetchall()
        return [dict(r) for r in rows]

    def list_enabled(self) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM symbols WHERE enabled = 1 ORDER BY symbol").fetchall()
        return [dict(r) for r in rows]

    def delete(self, symbol: str) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute("DELETE FROM symbols WHERE symbol = ?", (symbol,))


class MarketDataRepo:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def get_meta(self, symbol: str, timeframe: str) -> dict[str, Any] | None:
        with get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM market_data WHERE symbol = ? AND timeframe = ?", (symbol, timeframe)
            ).fetchone()
        return _row_to_dict(row)

    def upsert_meta(
        self,
        symbol: str,
        timeframe: str,
        source: str,
        rows: int,
        start_date: str | None,
        end_date: str | None,
        last_bar_time: str | None,
        file_path: str,
        adjust: str = "",
    ) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO market_data(symbol, timeframe, source, rows, start_date,
                                        end_date, last_bar_time, updated_at, file_path, adjust)
                VALUES(?,?,?,?,?,?,?, datetime('now'), ?, ?)
                ON CONFLICT(symbol, timeframe) DO UPDATE SET
                  source=excluded.source, rows=excluded.rows,
                  start_date=excluded.start_date, end_date=excluded.end_date,
                  last_bar_time=excluded.last_bar_time, updated_at=datetime('now'),
                  file_path=excluded.file_path, adjust=excluded.adjust
                """,
                (symbol, timeframe, source, rows, start_date, end_date, last_bar_time, file_path, adjust),
            )

    def list_meta(self) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM market_data ORDER BY symbol, timeframe").fetchall()
        return [dict(r) for r in rows]

    def delete_meta(self, symbol: str, timeframe: str) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute("DELETE FROM market_data WHERE symbol = ? AND timeframe = ?", (symbol, timeframe))


class BacktestRepo:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def create_run(self, engine: str, symbols: list[str], timeframe: str,
                   strategy_ids: list[str], portfolio_cfg: dict, settings: dict,
                   name: str = "") -> int:
        with get_conn(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO backtest_runs(name, engine, symbols_json, timeframe,
                                          strategy_ids_json, portfolio_json, settings_json,
                                          status, started_at)
                VALUES(?,?,?,?,?,?,?,'running', datetime('now'))
                """,
                (name, engine, json.dumps(symbols, ensure_ascii=False), timeframe,
                 json.dumps(strategy_ids), json.dumps(portfolio_cfg, ensure_ascii=False),
                 json.dumps(settings, ensure_ascii=False)),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, metrics: dict) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE backtest_runs SET status='done', metrics_json=?, finished_at=datetime('now') WHERE id=?",
                (json.dumps(metrics, ensure_ascii=False), run_id),
            )

    def fail_run(self, run_id: int, error: str) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE backtest_runs SET status='error', metrics_json=?, finished_at=datetime('now') WHERE id=?",
                (json.dumps({"error": error}, ensure_ascii=False), run_id),
            )

    def add_equity(self, run_id: int, equity_df) -> None:
        if equity_df is None or equity_df.empty:
            return
        rows = [
            (run_id, ts.strftime("%Y-%m-%d %H:%M:%S"),
             float(r.get("equity", 0)), float(r.get("cash", 0)),
             float(r.get("position_value", 0)), float(r.get("drawdown", 0)),
             (float(r.get("benchmark")) if pd_notna(r.get("benchmark")) else None),
             (float(r.get("buyhold")) if pd_notna(r.get("buyhold")) else None),
             (float(r.get("symbol_price")) if pd_notna(r.get("symbol_price")) else None))
            for ts, r in equity_df.iterrows()
        ]
        with get_conn(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO backtest_equity(run_id, ts, equity, cash, position_value,"
                " drawdown, benchmark, buyhold, symbol_price)"
                " VALUES(?,?,?,?,?,?,?,?,?)", rows)

    def add_trades(self, run_id: int, trades) -> None:
        if not trades:
            return
        rows = [
            (run_id, t.timestamp.strftime("%Y-%m-%d %H:%M:%S"), t.symbol, t.side,
             t.price, t.qty, t.amount, t.fee, t.slippage_cost, t.pnl,
             t.strategy_key, t.reason)
            for t in trades
        ]
        with get_conn(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO backtest_trades(run_id, ts, symbol, side, price, qty, amount,"
                " fee, slippage_cost, pnl, strategy_key, reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                rows)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM backtest_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with get_conn(self.db_path) as conn:
            row = conn.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,)).fetchone()
        return _row_to_dict(row)

    def get_equity(self, run_id: int, limit: int = 10000) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM backtest_equity WHERE run_id = ? ORDER BY ts LIMIT ?",
                (run_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_trades(self, run_id: int, limit: int = 1000) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY ts LIMIT ?",
                (run_id, limit)).fetchall()
        return [dict(r) for r in rows]


def pd_notna(v) -> bool:
    try:
        import math
        if v is None:
            return False
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


class StrategiesRepo:
    """策略配置持久化（启用/权重/参数 config_json）。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def upsert(self, key: str, name: str = "", description: str = "",
               strategy_type: str = "builtin", class_name: str = "",
               config_json: dict | None = None, enabled: int = 1,
               weight: float = 100.0, source_code: str | None = None) -> int:
        """插入或更新策略。source_code=None 表示不改动已存的源码。"""
        cfg = json.dumps(config_json or {}, ensure_ascii=False)
        with get_conn(self.db_path) as conn:
            if source_code is None:
                cur = conn.execute(
                    """
                    INSERT INTO strategies(key, name, description, strategy_type, class_name,
                                           config_json, enabled, weight, updated_at)
                    VALUES(?,?,?,?,?,?,?,?, datetime('now'))
                    ON CONFLICT(key) DO UPDATE SET
                      name=excluded.name, description=excluded.description,
                      strategy_type=excluded.strategy_type, class_name=excluded.class_name,
                      config_json=excluded.config_json, enabled=excluded.enabled,
                      weight=excluded.weight, updated_at=datetime('now')
                    """,
                    (key, name, description, strategy_type, class_name, cfg, enabled, weight),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO strategies(key, name, description, strategy_type, class_name,
                                           config_json, enabled, weight, source_code, updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?, datetime('now'))
                    ON CONFLICT(key) DO UPDATE SET
                      name=excluded.name, description=excluded.description,
                      strategy_type=excluded.strategy_type, class_name=excluded.class_name,
                      config_json=excluded.config_json, enabled=excluded.enabled,
                      weight=excluded.weight, source_code=excluded.source_code,
                      updated_at=datetime('now')
                    """,
                    (key, name, description, strategy_type, class_name, cfg, enabled,
                     weight, source_code),
                )
            return int(cur.lastrowid)

    def get(self, key: str) -> dict[str, Any] | None:
        with get_conn(self.db_path) as conn:
            row = conn.execute("SELECT * FROM strategies WHERE key = ?", (key,)).fetchone()
        return _row_to_dict(row)

    def get_by_id(self, strategy_id: int) -> dict[str, Any] | None:
        with get_conn(self.db_path) as conn:
            row = conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
        return _row_to_dict(row)

    def list_all(self) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM strategies ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def update(self, strategy_id: int, config_json: dict | None = None,
               enabled: int | None = None, weight: float | None = None,
               source_code: str | None = None, name: str | None = None,
               description: str | None = None) -> None:
        sets, vals = [], []
        if config_json is not None:
            sets.append("config_json = ?")
            vals.append(json.dumps(config_json, ensure_ascii=False))
        if source_code is not None:
            sets.append("source_code = ?")
            vals.append(source_code)
        if name is not None:
            sets.append("name = ?")
            vals.append(name)
        if description is not None:
            sets.append("description = ?")
            vals.append(description)
        if enabled is not None:
            sets.append("enabled = ?")
            vals.append(int(enabled))
        if weight is not None:
            sets.append("weight = ?")
            vals.append(float(weight))
        if not sets:
            return
        sets.append("updated_at = datetime('now')")
        vals.append(strategy_id)
        with get_conn(self.db_path) as conn:
            conn.execute(f"UPDATE strategies SET {', '.join(sets)} WHERE id = ?", vals)

    def delete(self, strategy_id: int) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute("DELETE FROM strategies WHERE id = ?", (strategy_id,))


class PortfolioRepo:
    """组合配置持久化（单行 id=1）。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def get(self) -> dict[str, Any] | None:
        with get_conn(self.db_path) as conn:
            row = conn.execute("SELECT * FROM portfolio WHERE id = 1").fetchone()
        return _row_to_dict(row)

    def put(self, mode: str, weights: dict, rebalance_freq: str,
            signal_threshold: float) -> dict[str, Any]:
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO portfolio(id, mode, weights_json, rebalance_freq,
                                      signal_threshold, updated_at)
                VALUES(1,?,?,?,?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                  mode=excluded.mode, weights_json=excluded.weights_json,
                  rebalance_freq=excluded.rebalance_freq,
                  signal_threshold=excluded.signal_threshold, updated_at=datetime('now')
                """,
                (mode, json.dumps(weights, ensure_ascii=False), rebalance_freq, signal_threshold),
            )
        return self.get()


class KVRepo:
    """通用键值配置（kv_store 表）。

    用于「新增的、还在迭代的」配置项（如模型集成权重、硬性风控阈值），
    不必每加一个功能就改 app_settings 的列结构。
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def get(self, key: str, default: Any = None) -> Any:
        with get_conn(self.db_path) as conn:
            row = conn.execute("SELECT value_json FROM kv_store WHERE key = ?", (key,)).fetchone()
        if row is None or row["value_json"] is None:
            return default
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:
            return default

    def put(self, key: str, value: Any) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO kv_store(key, value_json) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json
                """,
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def delete(self, key: str) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))


class SettingsRepo:
    """应用设置持久化（单行 id=1：trading / llm）。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def get_trading(self) -> dict[str, Any] | None:
        with get_conn(self.db_path) as conn:
            row = conn.execute("SELECT trading_json FROM app_settings WHERE id = 1").fetchone()
        if row is None:
            return None
        return json.loads(row["trading_json"] or "{}")

    def put_trading(self, trading: dict) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO app_settings(id, trading_json, updated_at) VALUES(1, ?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET trading_json=excluded.trading_json,
                                              updated_at=datetime('now')
                """,
                (json.dumps(trading, ensure_ascii=False),),
            )

    def get_llm(self) -> dict[str, Any] | None:
        with get_conn(self.db_path) as conn:
            row = conn.execute("SELECT llm_json FROM app_settings WHERE id = 1").fetchone()
        if row is None:
            return None
        return json.loads(row["llm_json"] or "{}")

    def put_llm(self, llm: dict) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO app_settings(id, llm_json, updated_at) VALUES(1, ?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET llm_json=excluded.llm_json,
                                              updated_at=datetime('now')
                """,
                (json.dumps(llm, ensure_ascii=False),),
            )


class PaperRepo:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    # ---- 账户 ----
    def create_account(self, name: str, initial_capital: float, settings_json: dict) -> int:
        with get_conn(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO paper_accounts(name, initial_capital, settings_json, status, created_at)"
                " VALUES(?,?,?,'stopped', datetime('now'))",
                (name, initial_capital, json.dumps(settings_json, ensure_ascii=False)),
            )
            return int(cur.lastrowid)

    def get_account(self, account_id: int) -> dict[str, Any] | None:
        with get_conn(self.db_path) as conn:
            row = conn.execute("SELECT * FROM paper_accounts WHERE id = ?", (account_id,)).fetchone()
        return _row_to_dict(row)

    def list_accounts(self) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM paper_accounts ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def set_status(self, account_id: int, status: str) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE paper_accounts SET status=?, updated_at=datetime('now') WHERE id=?",
                (status, account_id),
            )

    def delete_account(self, account_id: int, cleanup: bool = True) -> None:
        """删除账户；cleanup=True 时同时清理持仓/成交/净值/挂单记录。"""
        with get_conn(self.db_path) as conn:
            if cleanup:
                conn.execute("DELETE FROM paper_positions WHERE account_id = ?", (account_id,))
                conn.execute("DELETE FROM paper_trades WHERE account_id = ?", (account_id,))
                conn.execute("DELETE FROM paper_equity WHERE account_id = ?", (account_id,))
                conn.execute("DELETE FROM paper_pending_orders WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM paper_accounts WHERE id = ?", (account_id,))

    def update_settings(self, account_id: int, settings_json: dict) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE paper_accounts SET settings_json=?, updated_at=datetime('now') WHERE id=?",
                (json.dumps(settings_json, ensure_ascii=False), account_id),
            )

    # ---- 持仓（sub_key：capital_split 子账户，signal_merge = 'main'） ----
    def upsert_position(self, account_id: int, sub_key: str, pos: dict) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO paper_positions(account_id, sub_key, symbol, qty, avg_price,
                                            last_price, market_value, unrealized_pnl,
                                            opened_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?, datetime('now'))
                ON CONFLICT(account_id, sub_key, symbol) DO UPDATE SET
                  qty=excluded.qty, avg_price=excluded.avg_price, last_price=excluded.last_price,
                  market_value=excluded.market_value, unrealized_pnl=excluded.unrealized_pnl,
                  updated_at=datetime('now')
                """,
                (account_id, sub_key, pos["symbol"], pos["qty"], pos["avg_price"],
                 pos["last_price"], pos["market_value"], pos["unrealized_pnl"],
                 pos.get("opened_at") or None),
            )

    def delete_position(self, account_id: int, sub_key: str, symbol: str) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                "DELETE FROM paper_positions WHERE account_id = ? AND sub_key = ? AND symbol = ?",
                (account_id, sub_key, symbol))

    def list_positions(self, account_id: int, sub_key: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM paper_positions WHERE account_id = ?"
        args: list = [account_id]
        if sub_key is not None:
            sql += " AND sub_key = ?"
            args.append(sub_key)
        sql += " ORDER BY symbol"
        with get_conn(self.db_path) as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def clear_positions(self, account_id: int) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute("DELETE FROM paper_positions WHERE account_id = ?", (account_id,))

    # ---- 成交 ----
    def add_trade(self, account_id: int, sub_key: str, t) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                "INSERT INTO paper_trades(account_id, sub_key, ts, symbol, side, price, qty,"
                " fee, slippage_cost, pnl, strategy_key, reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (account_id, sub_key, t.timestamp.strftime("%Y-%m-%d %H:%M:%S"), t.symbol, t.side,
                 t.price, t.qty, t.fee, t.slippage_cost, t.pnl, t.strategy_key, t.reason),
            )

    def list_trades(self, account_id: int, limit: int = 500) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM paper_trades WHERE account_id = ? ORDER BY id DESC LIMIT ?",
                (account_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # ---- 净值 ----
    def add_equity(self, account_id: int, ts, equity: float, cash: float, pv: float) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                "INSERT INTO paper_equity(account_id, ts, equity, cash, position_value) VALUES(?,?,?,?,?)",
                (account_id, ts.strftime("%Y-%m-%d %H:%M:%S"), equity, cash, pv),
            )

    def list_equity(self, account_id: int, limit: int = 10000) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM paper_equity WHERE account_id = ? ORDER BY ts LIMIT ?",
                (account_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # ---- 挂单 ----
    def add_pending(self, account_id: int, order) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                "INSERT INTO paper_pending_orders(account_id, symbol, side, qty, ref_price,"
                " reason, strategy_key, created_ts) VALUES(?,?,?,?,?,?,?,?)",
                (account_id, order.symbol, order.side, order.qty, order.ref_price,
                 order.reason, order.strategy_key,
                 order.timestamp.strftime("%Y-%m-%d %H:%M:%S")),
            )

    def list_pending(self, account_id: int) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM paper_pending_orders WHERE account_id = ? ORDER BY id", (account_id,)).fetchall()
        return [dict(r) for r in rows]

    def clear_pending(self, account_id: int) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute("DELETE FROM paper_pending_orders WHERE account_id = ?", (account_id,))


class LLMRepo:
    """大模型对话与审计（llm_sessions / llm_messages / llm_prompts）。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    # ---- 会话 ----
    def create_session(self, title: str = "新对话", mode: str = "code",
                       strategy_id: int | None = None) -> dict[str, Any]:
        with get_conn(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO llm_sessions(title, mode, strategy_id, updated_at)"
                " VALUES(?,?,?, datetime('now'))",
                (title or "新对话", mode or "code", strategy_id))
            sid = int(cur.lastrowid)
        return self.get_session(sid) or {}

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        with get_conn(self.db_path) as conn:
            row = conn.execute("SELECT * FROM llm_sessions WHERE id = ?",
                               (session_id,)).fetchone()
        return _row_to_dict(row)

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM llm_messages m WHERE m.session_id = s.id)"
                " AS message_count FROM llm_sessions s"
                " ORDER BY COALESCE(s.updated_at, s.created_at) DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    def update_session(self, session_id: int, title: str | None = None,
                       mode: str | None = None,
                       strategy_id: int | None = None) -> None:
        sets, vals = [], []
        if title is not None:
            sets.append("title = ?")
            vals.append(title)
        if mode is not None:
            sets.append("mode = ?")
            vals.append(mode)
        if strategy_id is not None:
            sets.append("strategy_id = ?")
            vals.append(int(strategy_id))
        if not sets:
            return
        sets.append("updated_at = datetime('now')")
        vals.append(session_id)
        with get_conn(self.db_path) as conn:
            conn.execute(f"UPDATE llm_sessions SET {', '.join(sets)} WHERE id = ?", vals)

    def touch_session(self, session_id: int) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute("UPDATE llm_sessions SET updated_at = datetime('now') WHERE id = ?",
                         (session_id,))

    def delete_session(self, session_id: int) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute("DELETE FROM llm_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM llm_sessions WHERE id = ?", (session_id,))

    # ---- 消息 ----
    def add_message(self, session_id: int, role: str, content: str = "",
                    code: str = "", meta: dict | None = None) -> dict[str, Any]:
        with get_conn(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO llm_messages(session_id, role, content, code, meta_json)"
                " VALUES(?,?,?,?,?)",
                (session_id, role, content or "", code or "",
                 json.dumps(meta or {}, ensure_ascii=False)))
            mid = int(cur.lastrowid)
            conn.execute("UPDATE llm_sessions SET updated_at = datetime('now') WHERE id = ?",
                         (session_id,))
        return self.get_message(mid) or {}

    def get_message(self, message_id: int) -> dict[str, Any] | None:
        with get_conn(self.db_path) as conn:
            row = conn.execute("SELECT * FROM llm_messages WHERE id = ?",
                               (message_id,)).fetchone()
        return _msg_row(row)

    def list_messages(self, session_id: int) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM llm_messages WHERE session_id = ? ORDER BY id",
                (session_id,)).fetchall()
        return [_msg_row(r) for r in rows if r is not None]

    def update_message(self, message_id: int, content: str | None = None,
                       code: str | None = None, meta: dict | None = None) -> None:
        sets, vals = [], []
        if content is not None:
            sets.append("content = ?")
            vals.append(content)
        if code is not None:
            sets.append("code = ?")
            vals.append(code)
        if meta is not None:
            sets.append("meta_json = ?")
            vals.append(json.dumps(meta, ensure_ascii=False))
        if not sets:
            return
        vals.append(message_id)
        with get_conn(self.db_path) as conn:
            conn.execute(f"UPDATE llm_messages SET {', '.join(sets)} WHERE id = ?", vals)

    # ---- 审计 ----
    def log_prompt(self, prompt: str, model: str = "", strategy_id: int | None = None,
                   mode: str = "code", ok: int = 1, detail: str = "") -> int:
        with get_conn(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO llm_prompts(prompt, strategy_id, model, mode, ok, detail)"
                " VALUES(?,?,?,?,?,?)",
                (prompt, strategy_id, model, mode, int(ok), detail or ""))
            return int(cur.lastrowid)

    def list_prompts(self, limit: int = 50) -> list[dict[str, Any]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM llm_prompts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def _msg_row(row) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    try:
        d["meta"] = json.loads(d.get("meta_json") or "{}")
    except Exception:  # noqa: BLE001
        d["meta"] = {}
    d.pop("meta_json", None)
    return d
