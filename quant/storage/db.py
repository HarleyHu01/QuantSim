"""SQLite 初始化与连接管理。

表结构见 docs/ARCHITECTURE.md §12。本步先实现 symbols / market_data 的读写，
其余表随对应模块逐步接入。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS symbols (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL UNIQUE,
  name TEXT,
  asset_type TEXT NOT NULL DEFAULT 'stock',
  source TEXT NOT NULL DEFAULT 'yfinance',
  exchange TEXT,
  lot_size INTEGER DEFAULT 1,
  price_limit_pct REAL,            -- 涨跌停比例（0.10 主板 / 0.20 创业板·科创 / 0.05 ST；NULL=无）
  t_plus TEXT DEFAULT 'T+1',
  enabled INTEGER DEFAULT 1,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS market_data (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  source TEXT NOT NULL,
  rows INTEGER DEFAULT 0,
  start_date TEXT,
  end_date TEXT,
  last_bar_time TEXT,
  updated_at TEXT DEFAULT (datetime('now')),
  file_path TEXT,
  adjust TEXT DEFAULT '',
  UNIQUE(symbol, timeframe)
);

CREATE TABLE IF NOT EXISTS strategies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  description TEXT DEFAULT '',
  strategy_type TEXT NOT NULL DEFAULT 'builtin',
  class_name TEXT,
  config_json TEXT NOT NULL DEFAULT '{}',
  source_code TEXT DEFAULT '',
  enabled INTEGER DEFAULT 1,
  weight REAL DEFAULT 100.0,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS portfolio (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  mode TEXT DEFAULT 'signal_merge',
  weights_json TEXT DEFAULT '{}',
  rebalance_freq TEXT DEFAULT 'none',
  signal_threshold REAL DEFAULT 0.0,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS app_settings (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  trading_json TEXT DEFAULT '{}',
  llm_json TEXT DEFAULT '{}',
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS backtest_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  engine TEXT DEFAULT 'event_driven',
  symbols_json TEXT,
  timeframe TEXT,
  strategy_ids_json TEXT,
  portfolio_json TEXT,
  settings_json TEXT,
  status TEXT DEFAULT 'running',
  metrics_json TEXT,
  started_at TEXT,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS backtest_equity (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  ts TEXT,
  equity REAL,
  cash REAL,
  position_value REAL,
  exposure REAL,
  drawdown REAL,
  benchmark REAL,
  buyhold REAL,          -- 等权买入持有权益（对比曲线）
  symbol_price REAL      -- 首交易标的收盘价（标的股价走势）
);
CREATE INDEX IF NOT EXISTS idx_bt_equity_run ON backtest_equity(run_id, ts);

CREATE TABLE IF NOT EXISTS backtest_trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  ts TEXT,
  symbol TEXT,
  side TEXT,
  price REAL,
  qty REAL,
  amount REAL,
  fee REAL,
  slippage_cost REAL,
  pnl REAL,
  strategy_key TEXT,
  reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_bt_trades_run ON backtest_trades(run_id, ts);

CREATE TABLE IF NOT EXISTS paper_accounts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  initial_capital REAL,
  settings_json TEXT,
  status TEXT DEFAULT 'stopped',
  started_at TEXT,
  updated_at TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS paper_positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  account_id INTEGER,
  sub_key TEXT DEFAULT 'main',          -- capital_split 子账户标识（signal_merge = 'main'）
  symbol TEXT,
  qty REAL,
  avg_price REAL,
  last_price REAL,
  market_value REAL,
  unrealized_pnl REAL,
  opened_at TEXT,
  updated_at TEXT,
  UNIQUE(account_id, sub_key, symbol)
);

CREATE TABLE IF NOT EXISTS paper_trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  account_id INTEGER,
  sub_key TEXT DEFAULT '',
  ts TEXT,
  symbol TEXT,
  side TEXT,
  price REAL,
  qty REAL,
  fee REAL,
  slippage_cost REAL,
  pnl REAL,
  strategy_key TEXT,
  reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_paper_trades_acct ON paper_trades(account_id, ts);

CREATE TABLE IF NOT EXISTS paper_equity (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  account_id INTEGER,
  ts TEXT,
  equity REAL,
  cash REAL,
  position_value REAL
);
CREATE INDEX IF NOT EXISTS idx_paper_equity_acct ON paper_equity(account_id, ts);

-- 模拟盘挂单（next_open 语义：新 bar 到达前持久化；recover 时丢弃，避免重复成交）
CREATE TABLE IF NOT EXISTS paper_pending_orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  account_id INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  qty REAL NOT NULL,
  ref_price REAL NOT NULL,
  reason TEXT DEFAULT '',
  strategy_key TEXT DEFAULT '',
  created_ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_paper_pending_acct ON paper_pending_orders(account_id);

CREATE TABLE IF NOT EXISTS risk_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT DEFAULT (datetime('now')),
  scope TEXT,
  event_type TEXT,
  detail_json TEXT
);

CREATE TABLE IF NOT EXISTS llm_prompts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt TEXT,
  strategy_id INTEGER,
  model TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS kv_store (
  key TEXT PRIMARY KEY,
  value_json TEXT
);

CREATE TABLE IF NOT EXISTS llm_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT DEFAULT '新对话',
  mode TEXT DEFAULT 'code',
  strategy_id INTEGER,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS llm_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  role TEXT NOT NULL,
  content TEXT DEFAULT '',
  code TEXT DEFAULT '',
  meta_json TEXT DEFAULT '{}',
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_llm_messages_session ON llm_messages(session_id, id);
"""


def init_db(db_path: Path) -> None:
    """初始化数据库（幂等），含轻量迁移。"""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """对已存在的库做增量列迁移。"""
    # symbols.price_limit → price_limit_pct（第 3 步：涨跌停按 symbol 元数据）
    sym_cols = {r[1] for r in conn.execute("PRAGMA table_info(symbols)")}
    if "price_limit" in sym_cols and "price_limit_pct" not in sym_cols:
        conn.execute("ALTER TABLE symbols RENAME COLUMN price_limit TO price_limit_pct")
    # market_data.adjust（第 2 步遗留）
    md_cols = {r[1] for r in conn.execute("PRAGMA table_info(market_data)")}
    if "adjust" not in md_cols:
        conn.execute("ALTER TABLE market_data ADD COLUMN adjust TEXT DEFAULT ''")
    # paper_accounts.created_at（第 3 步遗留）
    pa_cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_accounts)")}
    if "created_at" not in pa_cols:
        conn.execute("ALTER TABLE paper_accounts ADD COLUMN created_at TEXT DEFAULT (datetime('now'))")
    # backtest_equity.position_value / exposure（第 3 步新增列）
    bte_cols = {r[1] for r in conn.execute("PRAGMA table_info(backtest_equity)")}
    if "position_value" not in bte_cols:
        conn.execute("ALTER TABLE backtest_equity ADD COLUMN position_value REAL DEFAULT 0")
    if "exposure" not in bte_cols:
        conn.execute("ALTER TABLE backtest_equity ADD COLUMN exposure REAL DEFAULT 0")
    # backtest_equity.buyhold / symbol_price（对比曲线列）
    if "buyhold" not in bte_cols:
        conn.execute("ALTER TABLE backtest_equity ADD COLUMN buyhold REAL")
    if "symbol_price" not in bte_cols:
        conn.execute("ALTER TABLE backtest_equity ADD COLUMN symbol_price REAL")
    # paper_positions.sub_key（第 5 步：capital_split 子账户支持，重建表以换唯一键）
    pp_cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_positions)")}
    if "sub_key" not in pp_cols:
        conn.execute("""
            CREATE TABLE paper_positions_new (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              account_id INTEGER, sub_key TEXT DEFAULT 'main',
              symbol TEXT, qty REAL, avg_price REAL, last_price REAL,
              market_value REAL, unrealized_pnl REAL,
              opened_at TEXT, updated_at TEXT,
              UNIQUE(account_id, sub_key, symbol)
            )
        """)
        conn.execute("""
            INSERT INTO paper_positions_new(account_id, sub_key, symbol, qty, avg_price,
                                            last_price, market_value, unrealized_pnl,
                                            opened_at, updated_at)
            SELECT account_id, 'main', symbol, qty, avg_price, last_price,
                   market_value, unrealized_pnl, opened_at, updated_at FROM paper_positions
        """)
        conn.execute("DROP TABLE paper_positions")
        conn.execute("ALTER TABLE paper_positions_new RENAME TO paper_positions")
    # paper_trades.sub_key（第 5 步）
    pt_cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)")}
    if "sub_key" not in pt_cols:
        conn.execute("ALTER TABLE paper_trades ADD COLUMN sub_key TEXT DEFAULT ''")
    # strategies.source_code（第 9 步：用户/AI 可直接编辑的 Python 代码）
    st_cols = {r[1] for r in conn.execute("PRAGMA table_info(strategies)")}
    if "source_code" not in st_cols:
        conn.execute("ALTER TABLE strategies ADD COLUMN source_code TEXT DEFAULT ''")
    # llm_prompts 审计表补列（第 9 步：记录模式/耗时/输出摘要）
    lp_cols = {r[1] for r in conn.execute("PRAGMA table_info(llm_prompts)")}
    for col, ddl in (
        ("mode", "ALTER TABLE llm_prompts ADD COLUMN mode TEXT DEFAULT 'code'"),
        ("ok", "ALTER TABLE llm_prompts ADD COLUMN ok INTEGER DEFAULT 1"),
        ("detail", "ALTER TABLE llm_prompts ADD COLUMN detail TEXT DEFAULT ''"),
    ):
        if col not in lp_cols:
            conn.execute(ddl)


@contextmanager
def get_conn(db_path: Path):
    """获取连接（自动提交/回滚）。"""
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
