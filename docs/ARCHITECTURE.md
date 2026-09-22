# 量化交易模拟系统 — 架构设计（v2，已整合评审调整）

> 目标：本地可运行的量化交易模拟系统。支持历史回测（向量化/事件驱动双模式）、实时模拟交易（paper trading）、
> 一键更新数据、内置经典策略库、策略组合与权重、LLM 自然语言创建策略、全部参数可在 UI 调节并持久化。
>
> v2 整合评审调整：① Streamlit 第一版前端 ② 数据源 fallback 与统一格式 ③ 双模式回测引擎 ④ 策略接口细化
> ⑤ 组合公式精确定义 ⑥ SQL schema 与索引 ⑦ 模拟盘调度机制 ⑧ LLM DSL 完整语法 ⑨ DeepSeek 集成 ⑩ 风控模块细节。

---

## 1. 总体架构

```
┌─────────────────────────── Web UI（第一版 Streamlit，API 优先，可整体替换 React）────────────────────┐
│  streamlit_app.py：数据管理页 │ 策略管理页 │ 策略创建页(LLM) │ 交易设置页 │ 回测页 │ 模拟交易页        │
│  K线图 = st.components + ECharts CDN；参数表单 = 按 Pydantic schema 动态渲染                           │
└──────────────────────────────────────┬──────────────────────────────────────────────────────────────┘
                                       │ REST API（HTTP + SSE/轮询；模拟盘用 WebSocket 或轮询刷新）
┌──────────────────────────────────────▼──────────────────────────────────────────────────────────────┐
│                                  FastAPI 应用层（quant/api）                                         │
│   routers: data / strategies / portfolio / settings / backtest / paper / llm / ws                    │
├───────────────────────────────┬──────────────────────────────┬───────────────────────────────────────┤
│ 数据模块 quant/data          │ 策略模块 quant/strategies    │ 组合模块 quant/portfolio              │
│  DataProvider 抽象 + fallback │  BaseStrategy 基类           │  StrategyPortfolio                   │
│  ├ yfinance（美股）             │  ├ 8+ 内置经典策略           │   signal_merge / capital_split /     │
│  ├ baostock（A股股票/指数/ETF） │  ├ RuleStrategy（LLM 声明式）│   equal（公式见 §6）                  │
│  ├ akshare（A股备用/期货）      │  └ 指标库 quant/core/indicators.py                                  │
│  └ ccxt（加密货币）             │                                                                    │
│  统一格式: datetime/open/high/low/close/volume/adj_factor，UTC 存储                                  │
├───────────────────────────────┴──────────────────────────────┴───────────────────────────────────────┤
│ 回测引擎：VectorizedBacktestEngine（快） ｜  EventDrivenBacktestEngine（准，与模拟盘共用 Broker）      │
│ Broker：撮合 / 手续费 / 滑点 / T+1 / 涨跌停 / 停牌 / lot         风控 risk.py（见 §11）               │
│ 模拟交易 PaperTradingEngine：APScheduler 调度（见 §9）                                                │
├──────────────────────────────────────────── 存储层 ──────────────────────────────────────────────────┤
│ SQLite（quant.db）：配置/策略/回测结果/模拟账户/风控事件      Parquet（data/market/*.parquet）：OHLCV  │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**数据流（5 条主线）**

1. **一键更新数据**：UI → `POST /api/data/update` → DataManager 按 symbol×timeframe 增量拉取 →
   归一化（统一列/UTC 时区/复权）→ 落 Parquet + 更新 `market_data` 元信息 → 返回行数/区间/更新时间。
2. **历史回测（向量化）**：读 Parquet → 策略目标仓位序列 → 组合合成 → 向量化收益计算（简化费用）→ 指标。
3. **历史回测（事件驱动）**：读 Parquet → 逐 bar 调用 Broker（费用/滑点/T+1/涨跌停/风控）→ 与模拟盘同口径。
4. **模拟交易**：APScheduler 按调度策略拉最新行情 → 增量信号 → Broker 撮合 → 更新账户 → 推送 UI；
   状态持久化，重启恢复。
5. **LLM 创建策略**：自然语言 → Prompt（JSON Schema + 示例）→ DeepSeek 返回声明式配置 → 校验链 → 草稿 →
   一键回测 → 激活。

---

## 2. 技术选型

| 层 | 选择 | 理由 |
|---|---|---|
| 语言 | Python 3.11+（本机 3.14 可用） | 生态成熟 |
| 后端 | FastAPI + uvicorn | 异步、OpenAPI、WebSocket、独立于 UI |
| 数据处理 | pandas / numpy | 标准 |
| 数据源 | baostock / yfinance / akshare / ccxt | 免费开源；多源 fallback（见 §3） |
| 行情存储 | Parquet（`data/market/`） | 列式高效、增量合并 |
| 元数据/配置/结果 | SQLite（`data/quant.db`） | 单文件、事务、WAL |
| 前端 | **第一版 Streamlit**（`streamlit_app.py`） | 零构建、迭代快；**API 独立设计**，后续可整体替换 React（Vite+Tailwind+ECharts 方案已评审通过，仅延后） |
| 大模型 | DeepSeek 官方 API（OpenAI 兼容），见 §10 | 国内直连、便宜；可换任何兼容接口 |
| 配置 | `config/default.yaml` + `.env` | 全部参数可持久化 |

---

## 3. 数据模块（quant/data）

### 3.1 统一数据格式与时区

- **统一列**：`datetime, open, high, low, close, volume, adj_factor`
  - `adj_factor`：复权因子（yfinance 由 Adj Close/Close 推导；baostock/akshare 已按 qfq/hfq 返回调整价时置 1）。
- **时区**：内部一律 **UTC** 存储（`DatetimeIndex`，tz-aware，bar 时间 = 开盘时间）；
  UI 按 `data.display_timezone`（默认本地时区）展示。
- **A股复权**：`adjust ∈ {qfq, hfq, none}`，默认 qfq（前复权），UI/API/CLI 均可切换。

### 3.2 DataProvider 抽象与 fallback

```python
class DataProvider(ABC):
    name: str
    asset_types: tuple[str, ...]
    def health_check(self) -> ProviderHealth: ...   # 每个数据源独立健康检查（网络探测+延迟）
    def fetch_history(self, symbol, start, end, timeframe, adjust) -> pd.DataFrame  # 统一格式
    def fetch_realtime(self, symbol) -> dict | None: ...

class DataManager:
    def health(self) -> dict[str, ProviderHealth]    # GET /api/data/status 展示
    def update(self, symbols, timeframes, incremental, adjust) -> list[UpdateResult]
    def load_bars(self, symbol, timeframe, start, end) -> pd.DataFrame
```

- **fallback 链**（`config/default.yaml` 配置，按资产类型路由）：

```yaml
data:
  providers:
    priority:
      stock:  [baostock, akshare, yfinance]   # A股股票：baostock 主源 → akshare（东财/腾讯/新浪）→ yfinance
      etf:    [baostock, akshare, yfinance]   # A股 ETF：baostock（2026-01-05 至今）→ akshare → yfinance
      index:  [baostock, akshare, yfinance]   # 指数：baostock 日/周/月（2006-01-01 至今）；分钟线走 akshare/yfinance
      future: [akshare, yfinance]             # 期货：仅 akshare（baostock 不支持期货）
      crypto: [ccxt]
```

- 每个 provider 有 `health_check()`（网络 ping + 延迟）；更新时按优先级逐个尝试，
  成功即停，结果记录**实际使用的数据源**；全部失败返回错误明细。
- 未安装的数据源（如 baostock/akshare 可选装）自动降级为 unhealthy 并跳过，不影响系统运行。
- **baostock（A股主源）**：官方客户端 TCP 直连证券宝服务器。覆盖（官方文档）：
  股票日/周/月自 1990-12-19、ETF 日/周/月+分钟自 2026-01-05、指数日/周/月自 2006-01-01
  （指数无分钟线，分钟请求快速失败并提示走 akshare/yfinance）。qfq/hfq/none 复权
  （分钟线/指数不支持复权，按不复权处理并告警）；成交量单位「股」；
  login/query/logout 全程加锁串行化。查询字段按周期区分（服务器严格校验，实测）：
  日线含 preclose 不含 time；周/月线不含 preclose；分钟线含 time 不含 preclose。
- **代码推断**（`infer_symbol_type`）：6 位 A股代码 → 51/56/58（沪）或 15/16（深）开头 → ETF；
  399xxx 或 000xxx（非 .SZ）→ 指数；其余 → 股票；均映射 baostock。
  注意：`000001.SZ` 是平安银行（股票），`000001.SH` 是上证指数。

### 3.3 增量更新与元信息

- 首次拉取 `lookback_days`（默认 730 天）；之后从 `last_bar_time - 重叠N根bar` 开始增量拉取，
  与本地 Parquet 按时间戳去重合并（keep=last）。
- 每次更新写 `market_data` 表：rows / start_date / end_date / last_bar_time / source / updated_at，
  UI 直接展示"数据更新时间、条数、覆盖区间"。

### 3.4 行情边界处理

- 缺失值：可配 ffill / drop（默认 ffill 且标记）。
- 停牌：零成交量 bar 标记停牌，引擎禁止交易。
- 涨跌停：按标的 `price_limit`（A股主板 10%、创业板/科创 20%、ST 5%；美股/加密/期货 NULL=无），
  引擎撮合校验（涨停禁买、跌停禁卖）。

---

## 4. 策略模块（quant/strategies）

### 4.1 统一接口（细化版）

```python
@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    target_weight: float      # -1(全空) ~ +1(全多)，0 = 空仓，可为小数
    reason: str = ""          # 信号原因（调试 + K线买卖点 tooltip）
    strength: float = 1.0
    meta: dict = {}

class StrategyContext:
    symbol: str
    timeframe: str
    history: pd.DataFrame          # 截至当前 bar 的完整行情
    extra: dict                    # 现金、持仓、bar_index 等运行期信息

class BaseStrategy(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""
    params: BaseModel              # Pydantic 参数模型，**每个参数必须有默认值**

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        """返回目标仓位 DataFrame：index=bar 时间，列 = [target_weight, reason]。
        策略只输出目标仓位，**不直接下单**；由组合器合成、引擎执行。"""

    def on_bar(self, bar, ctx) -> tuple[float, str]:
        """增量版本（模拟盘用）：默认取 generate_signals(ctx.history) 最新一行。"""
```

- **参数**：每个策略的 `params` 是 Pydantic 模型（含默认值）；UI 依据 schema 自动渲染参数表单；
  保存到 `strategies.config_json`，下次回测/模拟自动生效。
- **注册表**：`registry.py` 维护 key → 类映射；内置策略启动自动注册，RuleStrategy 保存时注册。

### 4.2 内置策略库（≥8 个，参数均有默认值）

| 策略 | key | 核心参数（默认值） |
|---|---|---|
| 双均线 | `dual_ma` | fast=20, slow=60, only_long=False |
| MACD | `macd` | fast=12, slow=26, signal=9, only_long=True |
| RSI 超买超卖 | `rsi` | period=14, oversold=30, overbought=70, only_long=True |
| 布林带 | `bollinger` | period=20, num_std=2.0, exit_at_mid=True |
| 动量 | `momentum` | lookback=60, hold_bars=10, only_long=True |
| 均值回归 | `mean_reversion` | lookback=20, entry_z=2.0, exit_z=0.0 |
| 网格交易 | `grid` | grid_pct=0.02, levels=10, base_position=0.5, only_long=True |
| 海龟交易 | `turtle` | entry_n=20, exit_n=10, atr_period=14, atr_mult=2.0, only_long=False |

全部基于 `quant/core/indicators.py`（SMA/EMA/MACD/RSI/BOLL/ATR/ROC/Donchian/Z-score…），
该指标库同时是 LLM 声明式策略的**指标白名单**。

---

## 5. LLM 声明式策略 DSL（完整语法）

### 5.1 结构

```json
{
  "name": "金叉动量策略",
  "description": "自然语言原文（审计用）",
  "symbols": ["AAPL"],
  "timeframe": "1d",
  "indicators": [{"name": "sma", "params": {"window": 20}, "as": "sma20"}],
  "entry_long":  {"all": [{"lhs": "sma20", "op": "crosses_above", "rhs": "sma60"},
                          {"lhs": "rsi14", "op": "<", "rhs": 70}]},
  "exit_long":   {"any": [{"lhs": "rsi14", "op": ">", "rhs": 80},
                          {"lhs": "close", "op": "<", "rhs": "sma20"}]},
  "entry_short": null,
  "exit_short":  null,
  "position": {"mode": "fixed_pct", "value": 1.0, "max_position": 1.0}
}
```

### 5.2 指标白名单（name → 参数 → 输出）

| 指标 | 参数 | 输出 |
|---|---|---|
| `sma` | window | sma |
| `ema` | window | ema |
| `macd` | fast, slow, signal | macd, macd_signal, macd_hist |
| `rsi` | period | rsi |
| `boll` | period, num_std | boll_up, boll_mid, boll_low |
| `atr` | period | atr |
| `donchian` | period | dc_up, dc_mid, dc_low |
| `roc` | window | roc |
| `zscore` | window | zscore |
| `vwap` | window | vwap |
| `cci` | period | cci |
| `stoch` | k, d | stoch_k, stoch_d |
| `volume_sma` | window | volume_sma |

价格引用：`open / high / low / close / volume / position`。

### 5.3 操作符白名单

`crosses_above, crosses_below, >, <, >=, <=, ==, !=, between, is_true, not`

### 5.4 逻辑组合

- `all: [...]` 全部满足；`any: [...]` 任一满足；可嵌套 `not`。
- 条件数量上限（默认 20），防失控。

### 5.5 仓位管理模板（position）

| mode | 说明 |
|---|---|
| `fixed_pct` | 固定比例仓位（value=0.5 即半仓） |
| `fixed_amount` | 固定金额（value=10000 即 1 万元/笔） |
| `atr_pct` | ATR 动态仓位：`position = clip(risk_per_trade × equity / (atr × price), 0, max_position)` |

### 5.6 安全校验链（validator.py）

JSON 解析 → Pydantic → 指标名白名单 → 操作符白名单 → 逻辑深度/条件数上限 → 字段类型。
**默认模式不生成、不执行任何代码**。"代码策略"模式默认关闭，需 AST 沙箱白名单
（仅 numpy/pandas 纯函数，禁 import/exec/eval/open/socket/网络/文件）。

---

## 6. 组合模块（quant/portfolio）— 精确公式

设第 i 个策略权重 `w_i ∈ [-100, 100]`（可负=反向），目标仓位序列 `s_i[t] ∈ [-1, 1]`，
实际持仓 `p[t]`（上期合成结果）。

**① 按权重合成信号（signal_merge）**（默认）

```
combined[t] = Σᵢ wᵢ·sᵢ[t] / Σᵢ |wᵢ|          # 归一化，负权重天然反向
交易触发阈值：|combined[t] − p[t−1]| > θ 才撮合   # θ = signal_threshold，默认 0，可配
目标仓位：p[t] = clip(combined[t], −1, 1)
```

- 全部权重为 0 → 不交易。
- `equal` 等权 = wᵢ 全相等的特例（θ 同样生效）。

**② 按权重分配资金（capital_split）**

```
资金池：Cᵢ = C₀ × |wᵢ| / Σⱼ |wⱼ|
每个策略独立子账户，独立撮合（负权重 = 该池做空），各自产生净值曲线，
组合净值 = Σᵢ 子账户净值。再平衡日按 Cᵢ 目标重新对齐。
```

**③ 再平衡**：`rebalance_freq ∈ {none, daily, weekly, monthly}`，到点把实际仓位拉回合成目标。

**④ 模式对比提示（UI 展示）**：signal_merge 简单快速、信号互相抵消自然；
capital_split 各策略独立、互不干扰，但资金利用更分散。

---

## 7. 回测引擎（双模式）

| 特性 | VectorizedBacktestEngine | EventDrivenBacktestEngine（默认） |
|---|---|---|
| 速度 | 毫秒级（纯 pandas） | 逐 bar 撮合，较慢但精确 |
| 费用/滑点 | 简化（按成交额近似） | 完整（佣金/最低/固定/滑点/印花税/过户费） |
| T+1 / 涨跌停 / 停牌 / lot | 不模拟（或简化） | 完整模拟 |
| 风控（熔断/止损等） | 不模拟 | 完整（risk.py） |
| 与模拟盘口径 | 不一致（仅快速验证） | **一致**（共用 Broker） |
| 适用 | 参数扫描、策略初筛 | 正式回测、验收 |

- 回测参数（全部 UI 可调，均有默认值）：
  - 费用：比例佣金（万 1）/最低佣金(0)/固定佣金(0)/滑点(1bp)/印花税(卖出 0.05%,可关)/过户费(0.001%,可关)；
  - 频度：timeframe（1m~1M）、最小交易间隔（N 根 bar，默认 1）、信号冷却（默认 0）、再平衡频率（默认 none）；
  - 资金：初始资金（默认 1,000,000）、单标的最大仓位（默认 30%）、总敞口上限（默认 100%）；
  - 规则：允许做空（默认关）、T+0/T+1（默认按标的：股票 T+1、加密/期货 T+0）、lot（A股 100）。
- **输出**：总收益/年化/最大回撤/夏普（rf 可配）/胜率/盈亏比/交易次数/暴露/基准对比，
  以及**买入持有对比**（超额收益 vs 买入持有 / 买入持有年化 / 基准最大回撤 / Beta / Alpha / 信息比率）；
  每日净值、回撤曲线、资金曲线、**归一化收益对比曲线（策略/买入持有/基准/标的股价）**、
  **回撤对比曲线（策略 vs 基准）**、**月度收益热力图**、交易明细（含 reason）、持仓变化。
  基准缺失时默认用交易标的本体（买入持有）作为基准。

---

## 8. Broker（撮合核心，quant/engine/broker.py）

- 输入：订单（来自组合目标仓位差）→ 输出：成交（价格=bar 收盘价×(1±滑点)）。
- 校验顺序：风控（见 §11）→ 停牌（零量禁）→ 涨跌停（涨停禁买/跌停禁卖）→ T+1（当日买入次日才能卖）→
  lot 取整 → 资金/持仓校验 → 成交 → 计费（佣金+印花税+过户费+滑点成本）→ 记录 reason。

---

## 9. 模拟盘调度（quant/engine/paper.py + APScheduler）

- **调度策略**（AsyncIOScheduler，APScheduler 3.x）：

| 场景 | 调度 |
|---|---|
| 日线/周线/月线策略 | cron：每日收盘后运行一次（如 17:00 本地；A股 15:30） |
| 分钟线策略（crypto） | interval：每 N 分钟轮询（默认 1m~5m 可配） |
| 分钟线策略（股票） | interval：每 60s 轮询延迟行情（免费源限制，文档注明） |
| 组合再平衡 | 按 rebalance_freq 定时触发 |

- 每个运行中的账户注册独立 job 组；`paper_accounts.status='running'` 持久化，服务重启后按状态恢复全部 job。
- 状态恢复：账户/持仓/成交/净值全部落库；重启不丢数据。
- 新 bar 到达 → `strategy.on_bar` 增量信号 → 组合合成 → Broker 撮合 → 账户更新 → WebSocket/轮询推送 UI。

### 9.x 模拟盘 capital_split 支持（第 5 步计划）

模拟盘 v1 仅支持 signal_merge / equal；第 5 步（组合与权重管理）补充 capital_split，方案概要：

1. `PaperAccountState` 持有 `sub_brokers: list[Broker]`（每策略一个子账户，现金 = 初始资金 × |wᵢ|/Σ|wⱼ|，负权重反向），共享同一 `T1Ledger`（T+1 锁定跨子账户一致）。
2. 存储改造：`paper_positions` 唯一键改为 `(account_id, sub_key, symbol)`（不同子账户可持同一标的）；`paper_trades` 增加 `sub_key` 列（成交已带 strategy_key 可区分）。
3. `_process_bar` 改为遍历子账户：每个子账户只用自己策略的信号独立撮合；净值 = Σ 子账户净值。
4. `recover()` 按 sub_key 分别重建各子 Broker；PENDING 挂单同样丢弃。
5. API/CLI 放行 `portfolio.mode="capital_split"` + weights；新增 `test_paper` 的 capital_split 场景（两策略同标的、独立持仓与净值合并）。

---

## 10. 大模型集成（DeepSeek，OpenAI 兼容）

- 配置（`settings/llm` + `.env`）：

```
LLM_API_KEY=sk-xxx          # 必填
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat     # deepseek-chat(V3) | deepseek-reasoner(R1)
```

- 调用：OpenAI 兼容 Chat Completions（`httpx` 直连，无需 SDK）；支持 `response_format: json_object`。
- **可靠性**：连接超时（默认 60s）、指数退避重试（默认 3 次）、错误分类处理
  （429 限流 → 退避重试；401 → 提示检查 key；超时 → 重试；网络错误 → 明确报错），
  全部错误返回给 UI 可读消息。
- 生成记录写入 `llm_prompts` 审计表（prompt/模型/时间/策略 id）。

---

## 11. 风控模块（quant/engine/risk.py）

| 风控项 | 参数（默认） | 行为 |
|---|---|---|
| 最大回撤熔断 | `max_drawdown_halt`（NULL=关） | 回撤 > X% → 暂停所有新开仓（可选强平），状态=halted，UI 通知；回撤恢复 < Y%（X−2%）或手动重置 → 恢复 |
| 单笔止损 | `stop_loss_pct`（NULL=关） | 持仓亏损 ≥ X% → 平仓 |
| 单笔止盈 | `take_profit_pct`（NULL=关） | 持仓盈利 ≥ X% → 平仓 |
| 最大持仓数 | `max_positions`（NULL=不限） | 超过则拒绝新开仓 |
| 每日最大亏损 | `daily_loss_limit_pct`（NULL=关） | 当日亏损 ≥ X% → 当日暂停交易 |
| 敞口限制 | `max_position_pct`(30%) / `max_exposure_pct`(100%) | 单标的最大仓位 / 总敞口上限 |

- **事件与恢复**：所有风控触发写入 `risk_events` 表（scope=backtest:<run_id>|paper:<account_id>，
  event_type，detail_json），UI 弹窗/横幅通知；恢复机制 = 手动重置 或 条件自愈（回撤恢复/次日零点）。

---

## 12. 数据库表结构（SQLite，`data/quant.db`，WAL 模式）

```sql
-- 标的池
CREATE TABLE IF NOT EXISTS symbols (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL UNIQUE,
  name TEXT, asset_type TEXT NOT NULL DEFAULT 'stock',   -- stock|etf|crypto|future
  source TEXT NOT NULL DEFAULT 'yfinance',
  exchange TEXT, lot_size INTEGER DEFAULT 1,
  price_limit REAL,                          -- 涨跌停比例(如0.10), NULL=无
  t_plus TEXT DEFAULT 'T+1',                 -- T+0|T+1
  enabled INTEGER DEFAULT 1,
  created_at TEXT DEFAULT (datetime('now')), updated_at TEXT
);

-- 行情元信息（一键更新后展示；datetime 明细存 parquet，SQLite 只存元数据）
CREATE TABLE IF NOT EXISTS market_data (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL, timeframe TEXT NOT NULL, source TEXT NOT NULL,
  rows INTEGER DEFAULT 0,
  start_date TEXT, end_date TEXT, last_bar_time TEXT,
  updated_at TEXT DEFAULT (datetime('now')), file_path TEXT,
  adjust TEXT DEFAULT '',                     -- 实际使用的复权方式 qfq|hfq|none
  UNIQUE(symbol, timeframe)
);
-- 注：唯一键为 (symbol, timeframe)；实际行情行按 datetime 去重存放在
--     data/market/{symbol}_{tf_token}.parquet（1M → 1mo，避免 Windows 大小写文件冲突）

-- 策略（参数与 LLM 声明式配置存 config_json）
CREATE TABLE IF NOT EXISTS strategies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL, description TEXT DEFAULT '',
  strategy_type TEXT NOT NULL DEFAULT 'builtin',   -- builtin | llm_generated
  class_name TEXT,
  config_json TEXT NOT NULL DEFAULT '{}',          -- params + LLM 声明式配置
  enabled INTEGER DEFAULT 1,
  weight REAL DEFAULT 100.0,                       -- -100..100
  created_at TEXT DEFAULT (datetime('now')), updated_at TEXT
);

-- 组合配置（单行）
CREATE TABLE IF NOT EXISTS portfolio (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  mode TEXT DEFAULT 'signal_merge',          -- signal_merge|capital_split|equal
  weights_json TEXT DEFAULT '{}',
  rebalance_freq TEXT DEFAULT 'none',        -- none|daily|weekly|monthly
  signal_threshold REAL DEFAULT 0.0,
  updated_at TEXT
);

-- 应用设置（JSON 存储，默认值在代码 Pydantic 模型，UI 可改）
CREATE TABLE IF NOT EXISTS app_settings (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  trading_json TEXT DEFAULT '{}',            -- 交易/频度/风控设置
  llm_json TEXT DEFAULT '{}',                -- base_url/api_key/model/超时/重试
  updated_at TEXT
);

-- 回测运行（关联策略组合与引擎类型）
CREATE TABLE IF NOT EXISTS backtest_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  engine TEXT DEFAULT 'event_driven',       -- vectorized|event_driven
  symbols_json TEXT, timeframe TEXT,
  strategy_ids_json TEXT,                   -- 参与的策略组合
  portfolio_json TEXT, settings_json TEXT,  -- 配置快照
  status TEXT DEFAULT 'running',            -- running|done|error
  metrics_json TEXT,
  started_at TEXT, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS backtest_equity (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER,
  ts TEXT, equity REAL, cash REAL, drawdown REAL, benchmark REAL,
  buyhold REAL, symbol_price REAL            -- 买入持有权益 / 首标的收盘价（对比曲线）
);
CREATE TABLE IF NOT EXISTS backtest_trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER,
  ts TEXT, symbol TEXT, side TEXT,          -- buy|sell|short|cover
  price REAL, qty REAL, amount REAL,
  fee REAL, slippage_cost REAL, pnl REAL,
  strategy_key TEXT, reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_bt_equity_run   ON backtest_equity(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_bt_trades_run   ON backtest_trades(run_id, ts);

-- 模拟账户
CREATE TABLE IF NOT EXISTS paper_accounts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
  initial_capital REAL, settings_json TEXT,
  status TEXT DEFAULT 'stopped', started_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS paper_positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, account_id INTEGER,
  symbol TEXT, qty REAL, avg_price REAL, last_price REAL,
  market_value REAL, unrealized_pnl REAL,
  opened_at TEXT, updated_at TEXT,
  UNIQUE(account_id, symbol)
);
CREATE TABLE IF NOT EXISTS paper_trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT, account_id INTEGER,
  ts TEXT, symbol TEXT, side TEXT, price REAL, qty REAL,
  fee REAL, slippage_cost REAL, pnl REAL, strategy_key TEXT, reason TEXT
);
CREATE TABLE IF NOT EXISTS paper_equity (
  id INTEGER PRIMARY KEY AUTOINCREMENT, account_id INTEGER,
  ts TEXT, equity REAL, cash REAL, position_value REAL
);
CREATE INDEX IF NOT EXISTS idx_paper_trades_acct ON paper_trades(account_id, ts);
CREATE INDEX IF NOT EXISTS idx_paper_equity_acct ON paper_equity(account_id, ts);

-- 风控事件（通知/恢复审计）
CREATE TABLE IF NOT EXISTS risk_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT DEFAULT (datetime('now')),
  scope TEXT,                                -- backtest:<run_id> | paper:<account_id> | manual
  event_type TEXT,                           -- max_drawdown_halt|stop_loss|take_profit|daily_loss_limit|max_positions|exposure_limit|resume
  detail_json TEXT
);

-- LLM 审计 + 通用键值
CREATE TABLE IF NOT EXISTS llm_prompts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt TEXT, strategy_id INTEGER, model TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value_json TEXT);
```

---

## 13. API 设计（FastAPI，前缀 /api）

```
GET    /api/health
-- 数据 --
GET    /api/data/symbols        POST /api/data/symbols      DELETE /api/data/symbols/{symbol}
GET    /api/data/status         # 每个 symbol×timeframe：覆盖区间/最新 bar/文件 mtime/stale 标记/adjust + 数据源健康
POST   /api/data/update         # 一键更新 {symbols?, timeframes?, incremental?, adjust?, full_reload?, start_date?}
GET    /api/data/bars/{symbol}  # ?timeframe=&start=&end= → OHLCV
-- 策略 --
GET    /api/strategies          POST /api/strategies
GET    /api/strategies/{id}     PUT  /api/strategies/{id}    # 参数/权重/启用
POST   /api/strategies/{id}/toggle     DELETE /api/strategies/{id}  # 内置→403；被组合引用→级联清理
GET    /api/strategies/builtins GET  /api/indicators         # 规则策略表单元信息（指标白名单/操作符/仓位模式）
POST   /api/strategies/rule     # 保存声明式规则策略（手动表单/LLM 共用校验链，草稿 enabled=0）
-- 组合 / 设置 --
GET/PUT /api/portfolio          GET/PUT /api/settings/trading
GET/PUT /api/settings/llm
-- 回测 --
POST   /api/backtest/run        {engine: vectorized|event_driven, ...}
GET    /api/backtest/runs       GET /api/backtest/runs/{id}
GET    /api/backtest/runs/{id}/equity    GET /api/backtest/runs/{id}/trades
-- 模拟盘 --
POST   /api/paper/start         POST /api/paper/stop
DELETE /api/paper/accounts/{id} # 仅 stopped 可删（运行中 409），?cleanup= 控制清理记录
GET    /api/paper/status        GET /api/paper/account
GET    /api/paper/positions     GET /api/paper/trades
GET    /api/paper/equity        WS  /ws/paper（实时推送）
-- 系统 --
POST   /api/system/shutdown     # 一键关闭：停模拟盘账户（落库）→ 优雅退出 uvicorn
-- LLM（预留）--
POST   /api/llm/generate-strategy       POST /api/llm/validate
```

---

## 14. Web UI 页面（Streamlit 第一版）

| 页面 | 功能 |
|---|---|
| 数据管理 | 标的池增删、资产类型/数据源/周期选择、**一键更新**、数据状态表（行数/区间/更新时间）+ 数据源健康（始终展示全部）、行情预览 |
| 策略创建 | **手动条件规则表单**：指标白名单（11 个）→ 买卖/做空/平空条件组（all/any、9 操作符）→ 仓位（固定比例/金额/ATR/信号强度）→ 实时 JSON 预览 → 保存草稿 / 保存并回测 → 激活；与 LLM 生成共用校验链 |
| 策略管理 | 内置 + 手动策略卡片：启用开关、schema 参数表单、**编辑跳转创建页**、**删除**（内置置灰/二次确认/组合引用级联清理） |
| 组合配置 | 三模式、权重滑杆（-100~100）、信号阈值/再平衡、组合信号分解预览 |
| 交易设置 | 手续费、频度（周期/最小间隔/冷却/再平衡）、资金与仓位、做空/T+0/T+1、风控参数、基准下拉预置 |
| 回测 | 引擎模式切换、运行、指标卡片（含**超额收益 vs 买入持有/Beta/Alpha/信息比率/基准最大回撤**）、**归一化收益对比图**（策略/买入持有/基准/标的股价）、**回撤对比图**（策略 vs 基准）、**月度收益热力图**、K线+买卖点+成交量（ECharts）、交易明细、多策略对比 |
| 模拟交易 | 开始/停止、**删除账户**（stopped 才可删）、调度信息、实时权益/持仓/成交、capital_split 子账户资金池 |
| 侧栏 | 环境标识（本地/沙箱）、**关闭系统**（红色按钮 + 二次确认 → `POST /api/system/shutdown` → 停模拟盘 → 优雅退出） |

---

## 15. 项目目录结构

```
D:\Insert\DeekSeek Harness\
├── main.py                     # 启动入口（uvicorn + 可选 streamlit 联动）
├── start.bat / start.sh        # 一条命令启动（自动建 venv、装依赖、起服务）
├── requirements.txt / requirements-optional.txt
├── pyproject.toml
├── .env.example                # LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
├── config/default.yaml         # 应用默认配置（数据源优先级/默认标的/默认周期）
├── docs/ARCHITECTURE.md        # 本文档
├── quant/
│   ├── core/                   # models / indicators / config / utils
│   ├── data/                   # base / yfinance_provider / baostock_provider /
│   │                           # akshare_provider / ccxt_provider / storage / manager
│   ├── strategies/             # base / registry / rule_strategy / builtin/ / llm/
│   ├── portfolio/portfolio.py
│   ├── engine/                 # backtest / paper / broker / risk / metrics
│   ├── api/                    # app / state / routers/ / ws
│   └── storage/                # db / repositories
├── streamlit_app.py            # 第一版 UI（Streamlit）
├── web/                        # 预留：React + Vite + Tailwind + ECharts（后续替换）
├── scripts/                    # update_data.py / run_backtest.py
├── tests/
└── data/                       # 运行时：market/*.parquet、quant.db
```

---

## 16. 配置与默认值

- 所有可调参数（策略参数、手续费、频度、仓位、权重、风控、LLM）**均带默认值**，空配置可直接运行。
- 优先级：代码默认值 < `config/default.yaml` < `.env`（LLM 密钥类）。
- 运行时修改全部持久化到 SQLite，重启自动加载。

---

## 17. 实施路线图（每步含测试与运行入口）

| 步 | 内容 | 验证方式 |
|---|---|---|
| 1 | 架构设计（本文档 v2） | 已确认 ✅ |
| 2 | 数据模块 + 一键更新 | 20 个测试 ✅（见 README） |
| 3 | **回测引擎（双模式）+ 模拟盘调度**（当前） | 37 个新增测试 ✅（合计 57）；`python scripts/run_backtest.py -s AAPL -tf 1d -st dual_ma` |
| 4 | **内置经典策略库（8 个）**（当前） | 25 个新增测试 ✅（合计 82）；`--combo` 组合回测实测通过 |
| 5 | **组合与权重管理**（当前） | 14 个新增测试 ✅（合计 96）；API/CLI 实测通过；模拟盘 capital_split 已实现 |
| 6 | **Web UI（Streamlit 第一版）**（当前） | 六页全通（AppTest 逐页无异常 + 回测交互实测）；`start.bat` 一键启动 |
| 6+ | **手动创建策略表单 + 策略删除 + 一键关闭系统**（当前） | 策略创建页（指标白名单表单→声明式 JSON→校验链→草稿→回测→激活）；策略删除（内置 403/组合级联）；`POST /api/system/shutdown` 优雅关闭；119 个测试全过 |
| 7 | LLM 策略生成（DeepSeek，**已预留未启用**） | `quant/strategies/llm/`（schema+validator）已就绪，`POST /api/strategies/rule` 接受同构配置；未来实现生成器即可 |
| 8 | 测试、文档、示例配置收尾 | `pytest -q`（119 全过）；README 全量验收 + `config/example.yaml` |

---

## 18. 验收标准对照

| 验收项 | 落地位置 |
|---|---|
| 一条命令启动 | `start.bat`/`start.sh` |
| 一键更新数据并看到结果 | 数据管理页 + `POST /api/data/update` |
| 内置策略选择运行回测 | 策略管理页 + 回测页 |
| 切换策略/调参/手续费/频度 | 策略管理页 + 交易设置页 |
| 策略权重与组合回测 | 策略管理页（权重滑杆）+ 组合模式 |
| 自然语言创建策略并回测 | 策略创建页（**手动表单已实现**；LLM 接口已预留 `quant/strategies/llm/`） |
| 完整指标/交易明细/图表 | 回测页 |
| 模拟交易持续运行并更新 | 模拟交易页（APScheduler + 推送） |
| 配置保存并下次自动加载 | SQLite 持久化 |
| 所有参数有默认值 | Pydantic 模型 + app_settings 默认值 |
