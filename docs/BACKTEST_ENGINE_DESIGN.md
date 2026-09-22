# 回测引擎 + 模拟盘调度 — 设计文档（v2，已整合评审调整）

> 细化 docs/ARCHITECTURE.md §7/§8/§9。v2 整合：做空保证金、A股费用默认值、按 symbol 涨跌停、
> T+1 统一账本、Signal.confidence、CombinedSignal 分解、按周期轮询间隔、PENDING 订单恢复、
> Trade 字段扩展、per_strategy_metrics、exposure 序列、combo/single 回测 API、run_paper CLI。

---

## 1. 核心类设计（接口签名级）

### 1.1 运行期模型（quant/core/models.py 扩展）

```python
@dataclass
class Order:                      # 订单（引擎/组合器产生，Broker 消费）
    timestamp: datetime
    symbol: str
    side: Literal["buy", "sell", "short", "cover"]
    qty: float
    ref_price: float              # 参考价（信号 bar 收盘价）
    reason: str = ""
    strategy_key: str = ""

@dataclass
class Trade:                      # 成交（落库）
    timestamp: datetime
    symbol: str
    side: str
    price: float                  # 成交价（含滑点）
    qty: float
    amount: float
    fee: float
    slippage_cost: float
    pnl: float                    # 平仓盈亏（开仓为 0）
    reason: str
    strategy_key: str
    strategy_name: str = ""       # 策略显示名（UI 展示）
    signal_reason: str = ""       # 触发信号原因（组合/策略层）
    entry_price: float | None = None   # 平仓成交回填开仓均价
    exit_price: float | None = None    # 开仓成交=None；平仓成交=本次价
    holding_days: float | None = None  # 持仓天数（平仓成交回填）

@dataclass
class Position:                   # 运行期持仓（Broker 内部）
    symbol: str
    qty: float                    # 正=多，负=空
    avg_price: float
    last_price: float
    realized_pnl: float = 0.0
    opened_at: datetime | None = None
    frozen_margin: float = 0.0    # 做空冻结保证金

class T1Ledger:                   # T+1 锁定账本（跨子账户/多策略共享）
    today_bought: dict[str, float]
    def settle_day(self, ts) -> None   # 日期切换时解锁
```

### 1.2 交易设置与组合配置（Pydantic，全部有默认值）

```python
class TradingSettings(BaseModel):
    initial_capital: float = 1_000_000.0
    # 手续费（A股惯例默认值，可调）
    commission_rate: float = 0.00025     # 比例佣金（万2.5，双边）
    commission_fixed: float = 0.0        # 固定每笔（>0 时按笔）
    commission_min: float = 5.0          # 最低佣金（A股 5 元）
    stamp_tax_rate: float = 0.0005       # 卖出印花税（可置 0 关闭）
    transfer_fee_rate: float = 0.00001   # 过户费（双边，可置 0 关闭）
    slippage_bps: float = 1.0            # 滑点 bp
    # 做空（allow_short=True 时生效）
    margin_rate: float = 0.5             # 保证金率（冻结 = |仓位价值|×margin_rate）
    margin_call_threshold: float = 0.8   # 浮亏超过冻结保证金 80% → 强制回补
    # 交易频度
    timeframe: str = "1d"
    min_trade_interval_bars: int = 1
    cooldown_bars: int = 0
    rebalance_freq: str = "none"
    signal_threshold: float = 0.0
    execution: str = "next_open"         # next_open（默认）| same_close
    # 仓位与规则
    max_position_pct: float = 0.3
    max_exposure_pct: float = 1.0
    allow_short: bool = False
    t_plus_enabled: bool = True          # 按标的 t_plus 属性执行
    # 风控（NULL=关闭）
    max_drawdown_halt: float | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    daily_loss_limit_pct: float | None = None
    max_positions: int | None = None
    # 基准与年化
    benchmark_symbol: str = "SPY"
    periods_per_year: int = 252

class PortfolioConfig(BaseModel):
    mode: str = "signal_merge"           # signal_merge | capital_split | equal
    weights: dict[str, float] = {}       # strategy_key → weight（-100..100，负=反向）
    signal_threshold: float = 0.0
    confidence_weighted: bool = True     # 合成时按 confidence 加权
```

### 1.3 策略接口（Signal 增加 confidence）

```python
class StrategyContext:
    symbol: str
    timeframe: str
    history: pd.DataFrame

class BaseStrategy(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""
    params: BaseModel

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        """返回目标仓位 DataFrame：index=bars.index，
        列 = [target_weight(-1..1), reason, confidence(0~1，默认 1.0)]。"""

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> tuple[float, str, float]:
        """增量版（模拟盘用）；返回 (target_weight, reason, confidence)。"""
```

### 1.4 组合器（quant/portfolio/portfolio.py）

```python
@dataclass
class CombinedSignal:             # 单 bar 合成结果（UI 可展示信号分解）
    target_weight: float
    per_strategy_weights: dict[str, float]          # 各策略权重（含符号）
    per_strategy_contribution: dict[str, float]     # w_i·s_i·conf_i / Σ|w_i·conf_i|
    confidence: float
    reason: str                   # 合成说明

class StrategyPortfolio:
    def __init__(self, config: PortfolioConfig): ...
    def split(self, strategies) -> list[SubPortfolio]:
        """signal_merge/equal → 1 子组合；capital_split → 每策略独立子账户"""
    def combine_series(self, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """全序列合成（向量化引擎）：列 = [target_weight, reason, confidence]"""
    def combine_at(self, signals: dict[str, pd.DataFrame], ts) -> CombinedSignal:
        """单 bar 分解（事件驱动引擎/UI 用）"""
```

**合成公式（confidence 加权）**：
```
signal_merge:  combined[t] = Σᵢ wᵢ·sᵢ[t]·confᵢ[t] / Σᵢ |wᵢ·confᵢ[t]|   （Σ|...|=0 时不交易）
               target = clip(combined, −1, 1)；|target − position| ≤ θ 不撮合
capital_split: 资金池 Cᵢ = C₀×|wᵢ|/Σ|wⱼ|，子账户独立撮合，净值合并
equal:         wᵢ 全相等特例
```

### 1.5 Broker（quant/engine/broker.py，回测与模拟盘共用）

```python
class Broker:
    def __init__(self, settings: TradingSettings, risk: RiskManager | None = None,
                 t1_ledger: T1Ledger | None = None): ...   # t1_ledger 缺省自建
    def mark(self, symbol, price, ts) -> None              # 盯市（含强平检查）
    def settle_day(self, ts) -> None                       # T+1 解锁（T1Ledger）
    def place_order(self, order: Order) -> Trade | None    # 撮合（规则见 §3）
    def equity(self) -> float
    def exposure(self) -> float                            # Σ|qty×price| / equity
    def position(self, symbol) -> Position | None
    @property cash / positions / t1_ledger
```

### 1.6 回测引擎（quant/engine/backtest.py）

```python
class BacktestEngine(ABC):
    engine_type: ClassVar[str]                    # "vectorized" | "event_driven"
    def __init__(self, settings, portfolio): ...
    @abstractmethod
    def run(self, symbol_bars: dict[str, pd.DataFrame],
            strategies: list[BaseStrategy]) -> BacktestResult: ...

class VectorizedBacktestEngine(BacktestEngine): ...   # 快，执行近似
class EventDrivenBacktestEngine(BacktestEngine): ...  # 默认，完整撮合
```

### 1.7 风控（quant/engine/risk.py）

```python
@dataclass
class OrderCheck:
    allowed: bool
    reason: str = ""

class RiskManager:
    def __init__(self, settings): ...
    def check_order(self, ts, symbol, side, qty, price, broker) -> OrderCheck
    def on_equity_update(self, ts, equity, broker) -> list[RiskEvent]   # 熔断/日亏/止损止盈
    def halted(self) -> bool
    def resume(self) -> None
```

### 1.8 指标（quant/engine/metrics.py）

```python
def compute_metrics(equity: pd.Series, trades: list[Trade],
                    benchmark: pd.Series | None = None,
                    periods_per_year: int = 252, rf: float = 0.0) -> dict
```

### 1.9 模拟盘（quant/engine/paper.py）

```python
class PaperTradingEngine:
    def __init__(self, config, data_manager, scheduler: AsyncIOScheduler, account_repo): ...
    def start(self, account_id, symbols, timeframe, poll_seconds=None) -> dict
    def stop(self, account_id) -> None
    def status(self, account_id) -> dict
    def on_bar(self, account, symbol, bar) -> None
    def pending_order(self, account_id, order) -> None      # 挂单落库（paper_pending_orders）
    @classmethod
    def recover(cls, ...) -> None                           # 见 §4.3
```

---

## 2. 向量化 vs 事件驱动：差异与取舍

| 维度 | VectorizedBacktestEngine | EventDrivenBacktestEngine（默认） |
|---|---|---|
| 信号计算 | 全量序列 `combine_series` | 同样预计算信号序列；执行层逐 bar `combine_at` |
| 执行模型 | 仓位变化×收益近似 | 逐 bar 逐订单走 Broker（完整撮合） |
| 费用/滑点 | 按换手率近似 | 逐笔精确 |
| T+1 / 涨跌停 / 停牌 / lot | 不模拟（文档注明） | 完整模拟 |
| 做空保证金/强平 | 不模拟 | 完整（margin_rate + 强平阈值） |
| 风控 | 仅回撤熔断可近似 | 完整 |
| capital_split 子账户 | 不支持 | 支持（子 Broker 共享 T1Ledger） |
| 与模拟盘口径 | 不一致，快速验证 | **一致**（共用 Broker） |
| 性能（3k bar × 3 标的） | < 100ms | < 2s |

取舍：信号层一致、执行层分叉——同一策略两种模式信号相同，可直接对比"执行损耗"。UI 提供切换与差异提示，默认事件驱动。

---

## 3. Broker 撮合规则（详细）

### 3.1 订单流程

```
bar t 收盘 → 目标仓位（combine） vs 当前仓位 → |Δ|>θ → Order
  → 频度过滤（min_trade_interval / cooldown，按 symbol 记 last_trade_bar）
  → RiskManager.check_order
  → 成交（next_open：t+1 开盘；same_close：t 收盘）
  → 校验：停牌 → 涨跌停（按 symbol price_limit_pct）→ T+1（T1Ledger）→ lot → 资金/保证金 → 成交 → 计费
```

### 3.2 成交价与滑点

- `next_open`（默认）：参考价 = 信号 bar 收盘；成交价 = 下一根 bar 开盘（避免未来函数；尾单丢弃记 warning）。
- `same_close`：成交价 = 信号 bar 收盘。
- 滑点：买 `×(1+bps/1e4)`；卖/空 `×(1−bps/1e4)`；`slippage_cost = qty×|成交价−参考价|` 单独记录。

### 3.3 手续费（A股惯例默认，均可调）

| 项目 | 公式 | 默认 |
|---|---|---|
| 比例佣金 | `amount × commission_rate` | 0.00025（万2.5，双边） |
| 固定佣金 | `commission_fixed`（>0 时按笔） | 0 |
| 最低佣金 | `max(佣金, commission_min)` | 5.0 元 |
| 印花税 | 仅卖出/平空侧：`amount × stamp_tax_rate` | 0.0005 |
| 过户费 | 买卖双侧：`amount × transfer_fee_rate` | 0.00001 |

佣金结算：`commission_fixed>0` → 每笔固定；否则比例；两者同设取较大值；再 `≥ commission_min`。

### 3.4 涨跌停（按 symbol 元数据，非全局）

- `symbols.price_limit_pct`（如 0.10 主板 / 0.20 创业板·科创 / 0.05 ST；美股/加密 NULL=无）。
- 新增 symbol 时默认：`stock→0.10`（可改）、`etf→0.10`（可改）、`crypto/future→NULL`，以元数据为准。
- 判定：`limit_up = prev_close×(1+pct)`、`limit_down = prev_close×(1−pct)`；
  涨停（`open ≥ limit_up×0.999`）禁买/禁开空；跌停（`open ≤ limit_down×1.001`）禁卖/禁平空；命中记 warning。

### 3.5 停牌

- bar `volume==0` 或缺失 → 不撮合；持仓按最近有效价盯市。

### 3.6 T+0 / T+1（T1Ledger 统一管理）

- `t_plus_enabled=True` 时，`t_plus="T+1"` 标的：`today_bought` 当日买入不可卖；卖出 ≤ `qty−today_bought`（超出截断记 warning）。
- **T1Ledger 由所有子账户（capital_split）共享** → 多策略买卖同一 symbol 的锁定状态一致（用户调整 1.4）。
- `t_plus="T+0"` 不限制；`settle_day` 日期切换解锁。做空侧对称（当日开空不可平空）。

### 3.7 最小交易单位（lot）

- 买入：`floor(qty/lot)×lot`（A股 100）；卖出：允许零股，≤ 可用数量。

### 3.8 资金、仓位与做空保证金

- 目标数量：`target_qty = target_weight × equity / price`，单标的上限 `max_position_pct×equity`。
- 买入/开空需现金 ≥ `amount + fee`。
- **做空保证金**（allow_short 时）：
  - 开空：冻结 `frozen_margin = |qty×price| × margin_rate`（默认 50%），现金需 ≥ 冻结 + 费用；
  - 盯市：空头浮亏 `(avg_price − price)×qty > frozen_margin × margin_call_threshold(80%)` → **强制回补**（市价全平该空头，记 risk_event + warning）；
  - 平空：释放冻结。
- `equity = cash + Σ qty×price`（空头 qty 为负）；`exposure = Σ|qty×price| / equity`。

### 3.9 风控（RiskManager 与 Broker 联动）

| 风控项 | 触发 | 动作 |
|---|---|---|
| 最大回撤熔断 | 回撤 > max_drawdown_halt | 停新开仓，记 risk_event；回落 2% 或手动恢复 |
| 单笔止损/止盈 | 浮盈亏超阈值 | 市价平仓 |
| 单标的最大仓位 | 超 max_position_pct | 截断 |
| 总敞口 | exposure > max_exposure_pct | 拒新开仓 |
| 每日最大亏损 | 当日亏损超限 | 当日停交易 |
| 最大持仓数 | 超限 | 拒新开仓 |

---

## 4. 模拟盘调度与状态管理

### 4.1 按周期细分轮询间隔

| timeframe | 调度 | 说明 |
|---|---|---|
| 1m | interval 65s | 比 K 线周期多 5s 余量 |
| 5m | interval 315s | +15s |
| 15m | interval 930s | +30s |
| 30m | interval 1830s | +30s |
| 1h | interval 3660s | +60s |
| 2h / 4h | interval 周期+60s | |
| 1d / 1w / 1M | **cron 按标的时区** | A股 Asia/Shanghai 15:30；美股 America/New_York 16:30；crypto 00:05 UTC（可配覆盖） |

`poll_seconds` 参数可覆盖默认间隔。

### 4.2 任务注册与状态持久化

- job id：`paper_{account_id}_{symbol}_{tf}`、`paper_{account_id}_rebalance`；幂等注册。
- 持久化：`paper_accounts.status='running'`（含 symbols/timeframe/轮询配置/策略 ids/权重快照）、
  `paper_positions`、`paper_trades`、`paper_equity`、**`paper_pending_orders`**（挂单）。
- 同一标的 T+1 锁定跨策略共享（T1Ledger 重建后多策略共用）。

### 4.3 恢复（recover()）

- 扫描 `status='running'` 账户 → 重建 Broker（paper_positions 载入 + 重建 T1Ledger）→ 重注册 job → 从 parquet 元信息继续增量。
- **PENDING 订单处理**：`paper_pending_orders` 中未执行的挂单**全部丢弃**并记 warning（不重新撮合，避免重复成交）。

### 4.4 on_bar 流程（与回测同口径）

```
新 bar 到达（已追加 parquet）
  → Broker.mark（含强平检查）→ 执行上 bar 挂单（next_open）
  → 各策略 on_bar → StrategyPortfolio.combine_at → CombinedSignal
  → |Δ|>θ → Order → 挂单落库（下一 bar 执行）或 Broker.place_order（same_close）
  → RiskManager.on_equity_update → 落库 positions/trades/equity（含 exposure）
```

---

## 5. 回测结果结构与指标

### 5.1 BacktestResult

```python
@dataclass
class BacktestResult:
    engine: str                          # vectorized | event_driven
    timeframe: str
    symbols: list[str]
    start: datetime
    end: datetime
    metrics: dict[str, float | None]              # 组合整体指标（见 5.2）
    per_strategy_metrics: dict[str, dict]         # 每策略独立回测的指标（策略对比）
    equity: pd.DataFrame                # index=ts, cols=[equity, cash, position_value,
                                        #            exposure, drawdown, benchmark]
    trades: list[Trade]                 # 交易明细（含 strategy_name/signal_reason/
                                        #            entry_price/exit_price/holding_days）
    positions_timeline: pd.DataFrame    # index=ts, cols=symbol（持仓 qty）
    combined_signal: pd.DataFrame       # index=ts, cols=[target_weight, reason, confidence,
                                        #            contributions(JSON)]
    settings_snapshot: dict             # TradingSettings + PortfolioConfig 快照
    warnings: list[str]
    run_id: int | None
```

存储：`backtest_runs` → `backtest_equity`（逐点，含 exposure）→ `backtest_trades`（全字段）。

### 5.2 指标清单（compute_metrics + per_strategy_metrics 同套）

总收益 / 年化（periods 按周期换算）/ 最大回撤 / 夏普（rf 可配）/ 索提诺 / 卡玛 /
胜率（盈利平仓/平仓总笔）/ 盈亏比（Σ盈利/|Σ亏损|，无亏损=None）/ 交易次数与平均盈亏 /
暴露度（avg exposure）/ 基准收益（buy&hold）/ 年化波动率。

### 5.3 性能目标

- 事件驱动：信号预计算 + 执行逐 bar；3k bar × 3 标的 < 2s。
- 向量化：< 100ms。多标的 union index 对齐，缺失 bar 不撮合。

---

## 6. 本步交付物

```
quant/engine/          broker.py / backtest.py(双引擎) / risk.py / metrics.py / paper.py
quant/portfolio/       portfolio.py（三模式 + CombinedSignal + confidence 加权）
quant/strategies/      base.py（Signal 含 confidence 契约）
quant/core/models.py   扩展 Order/Trade/Position/T1Ledger/TradingSettings/PortfolioConfig
quant/storage/db.py    symbols.price_limit_pct 迁移 + paper_pending_orders 表
quant/api/routers/     backtest.py（POST /api/backtest/single、POST /api/backtest/combo）
                       paper.py（start/stop/status）
scripts/run_backtest.py（CLI 回测）
scripts/run_paper.py   （CLI：start 前台常驻 / stop / status）
tests/                 test_broker / test_metrics / test_engines / test_paper
```

模拟盘驱动方式：API 服务（常驻）或 `run_paper.py start`（前台常驻）二选一，共用 PaperTradingEngine。
（8 个内置策略下一步接入，本步用测试策略验证引擎。）
