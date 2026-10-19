# 量化交易模拟系统 — 架构设计（v2，已整合评审调整）

> 目标：本地可运行的量化交易模拟系统。支持历史回测（向量化/事件驱动双模式）、实时模拟交易（paper trading）、
> 一键更新数据、内置经典策略库、策略组合与权重、LLM 自然语言创建策略、AI 策略工坊（对话生成/反复修改/思路建议）、
> 全部参数可在 UI 调节并持久化。
>
> v2 整合评审调整：① Streamlit 第一版前端 ② 数据源 fallback 与统一格式 ③ 双模式回测引擎 ④ 策略接口细化
> ⑤ 组合公式精确定义 ⑥ SQL schema 与索引 ⑦ 模拟盘调度机制 ⑧ LLM DSL 完整语法 ⑨ DeepSeek 集成 ⑩ 风控模块细节。
> 后续迭代（第 10~14 步）：⑪ 数据源精简与分钟级实时模拟 ⑫ 金融时序模型与 `alpha/` 分层
> ⑬ 大模型 × 因子自由融合 ⑭ AI 策略工坊三意图与版本实时预览。

---

## 1. 总体架构

```
┌─────────────────────────── Web UI（第一版 Streamlit，API 优先，可整体替换 React）────────────────────┐
│  streamlit_app.py：数据管理页 │ 策略管理页 │ 策略创建页(LLM) │ 交易设置页 │ 回测页 │ 模拟交易页        │
│  K线图 = st.components + ECharts CDN；参数表单 = 按 Pydantic schema 动态渲染                           │
└──────────────────────────────────────┬──────────────────────────────────────────────────────────────┘
                                       │ REST API（HTTP + 轮询；模拟盘页用 st.toggle 自动刷新 + st.rerun 轮询）
┌──────────────────────────────────────▼──────────────────────────────────────────────────────────────┐
│                                  FastAPI 应用层（quant/api）                                         │
│   routers: data / strategies / portfolio / settings / backtest / paper / llm / ws                    │
├───────────────────────────────┬──────────────────────────────┬───────────────────────────────────────┤
│ 数据模块 quant/data          │ 策略模块 quant/strategies    │ 组合模块 quant/portfolio              │
│  DataProvider 抽象 + fallback │  BaseStrategy 基类           │  StrategyPortfolio                   │
│  ├ yfinance（美股/ETF）         │  ├ 8+ 内置经典策略           │   signal_merge / capital_split /     │
│  ├ baostock（A股主源）          │  ├ RuleStrategy（LLM 声明式）│   equal（公式见 §6）                  │
│  ├ akshare（A股备选：          │  ├ CodeStrategy（AI 生成/手改）│ 指标库 quant/core/indicators.py      │
│  │   东财→腾讯→新浪 三腿）      │                                                                    │
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
| 数据源 | baostock / akshare（内置 东财·腾讯·新浪 三腿）/ yfinance | 免费开源；Provider 两级降级（见 §3） |
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

- **三渠道按序降级**（`config/default.yaml` 配置，按资产类型路由）：

```yaml
data:
  providers:
    stock:    [baostock, akshare]   # A股股票：baostock 主源 → akshare 备选（内部 东财→腾讯→新浪）
    etf:      [baostock, akshare]   # A股 ETF
    index:    [baostock, akshare]   # 指数：baostock 无分钟线，自动跳过交给下一渠道
    us_stock: [yfinance]            # 美股/海外
```

- **两级降级**：Provider 之间按序降级（baostock → akshare）；**Provider 内部的通道叫「腿」**，
  akshare 自己再按 东财 → 腾讯 → 新浪 降级。前一个拿不到、或该周期它根本不支持，才换下一个。
- **腾讯已并入 akshare 成为第三条腿**（第 18 步），不再是独立 provider：此前它既是独立 provider 又被
  akshare 内部调用，同一数据源在健康表出现两遍、代理开关要设两处、且「腾讯兜底」与「akshare 兜底」语义重叠。
  合并后语义清晰：**一个 provider = 一个网络技术栈**，Provider 之间才谈降级。
  `TencentProvider` 类仍保留（可单独使用/测试），但不在默认链里。
- 每个 provider 声明两项能力，`DataManager` 调度前先问能力：
  - `supports(symbol, timeframe) -> bool` —— 不支持的周期**直接跳过**（不计失败、不发无效请求）
  - `max_lookback_days(symbol, timeframe) -> int | None` —— 请求起点会被裁剪进可得窗口
    （例如 Yahoo 5m 只到近 60 天，请求 2000 年起会自动裁到 60 天前，而不是撞 422 报错）
- 每个 provider 仍有 `health_check()`（网络 ping + 延迟）；多通道 provider 额外填 `ProviderHealth.channels`
  （每项 `{name, healthy, latency_ms, error, breaker, preferred}`），`latency_ms` 只报**主用通道自己**的耗时。
  全部失败返回错误明细（含每个源被跳过/失败的原因）。
- 未安装的数据源（baostock/akshare 可选装）自动降级为 unhealthy 并跳过，不影响系统运行。
- **baostock（A股主源）**：官方客户端 TCP 直连证券宝服务器。
  - 股票日/周/月自上市起（实测 2001 年起 6000+ 根）、5/15/30/60 分钟实测 2020 年仍可取（≥6 年）
  - ETF 支持日/周/月 + 分钟
  - 指数日/周/月自 2006-01-01；**指数无分钟线**（`supports()` 返回 False，直接跳过）
  - **无 1 分钟粒度**（最小 5 分钟）
  - qfq/hfq/none 复权（分钟线/指数不支持复权，按不复权处理并告警）；成交量单位「股」
  - **进程内会话保持**（`_logged_in` / `_ensure_login()` / `_force_relogin()` / `close_session()`）：
    首次用到时 `login()` 一次，之后全部查询复用，**不再每条查询 login→logout**。
    根因：实测 `login()` 单次 **11~13 秒**（13450ms / 13151ms / 11107ms），而一次 query 只要 2~3 秒，
    TCP 建连仅 49ms——「每条查询都重登」等于每次拉数据白付十几秒，叠加 socket 超时就会判死报
    `10002007 网络接收错误`，**这才是历史上「baostock 总超时」的真正原因**，与网络抖动无关。
  - 服务器中途掉线（长查询跑到一半）时 `_query()` 自动 `_force_relogin()` 并重试 `_QUERY_RETRIES = 2` 次。
  - socket 超时 `_SOCKET_TIMEOUT = 30.0`（健康检查探测传 `timeout=6.0`）：官方 `socketutil` 不设超时，
    monkey-patch `bs_sock.context.default_socket` 注入；**30s 而非 15s** 是因为 login 正常也要 11~13s。
  - `DataManager.warm_up()` 在 app 启动时后台预热一次登录（login 要 11~13s，而 `health()` 总预算只有 8s，
    不预热必然误报「健康检查超时」）。实测健康检查因此从「超时」降到约 2 秒。
  - login/query/logout 全程加锁串行化。查询字段按周期区分（服务器严格校验，实测）：
    日线含 preclose 不含 time；周/月线不含 preclose；分钟线含 time 不含 preclose。
- **akshare（A股备选）**：**三条腿**——东方财富 → 腾讯 → 新浪（同 provider 内自动降级）。
  - **通道顺序随周期动态决定**（`_channels(timeframe)`，这是「数据广度」的直接体现，不能一刀切）：
    - `1m` → `[腾讯]`（东财/新浪都没有 1m，腾讯独占）
    - `5m/15m/30m/1h` → `[东方财富, 新浪, 腾讯]`（新浪恒 1970 根 ≫ 腾讯恒 640 根，
      **优先拿更深的历史**；若腾讯排前面，用户会莫名其妙只能拿到 640 根、比只有新浪时反而缩水）
    - `1d/1w/1M` → `[东方财富, 腾讯, 新浪]`（深度相当但腾讯约 0.55s、新浪约 1.6s，**腾讯更快**）
  - 东财：`push2his.eastmoney.com`（带 Referer + 浏览器 UA）；`_channel_eastmoney` 开头有周期守卫
    （`timeframe not in _KLT or timeframe == "1m"` 直接 `ValueError`）
  - 腾讯：**复用 `tencent_provider` 的取数原语**（`_get_json` / `_fetch_minute_rows` / `_fetch_daily_rows` /
    `_rows_to_df` / `_to_tx_code`），不重复实现一遍解析逻辑——字段序与成交量单位的坑只在一处。
  - 新浪：`stock_zh_a_daily`（个股日线，**只认 qfq/hfq，`adjust="none"` 返回空表**）、
    `stock_zh_a_minute`（分钟线）、`fund_etf_hist_sina`（ETF 日线）、`stock_zh_index_daily`（指数日线）；
    周/月线由日线 `resample("W-FRI" / "ME")` 聚合（open=first / high=max / low=min / close=last / volume=sum）
  - 标的分派 `_sina_classify(code, pref)`：sh 前缀 `51/56/58/50/52` → ETF、`000` → 指数、其余股票；
    sz 前缀 `15/16/18` → ETF、`399` → 指数、其余股票
  - `_clip_to_window(df, start, end)`：腾讯端点**没有区间参数**、总是返回「最近 640 根」，
    不裁剪会让 `UpdateResult.start/end` 报出远超请求区间的范围；裁完为空则 `raise ValueError("该区间内无数据")`。
    **必须显式对齐时区**：通道索引是 tz-naive 北京时间、请求边界常是 tz-aware(UTC)，
    直接比较抛 `Invalid comparison between dtype=datetime64[us] and Timestamp`，
    会被 fallback 链误吞成「腾讯通道故障」→ 曾表现为 1m 全部下载失败
    （回归测试 `test_akshare_clip_to_window_handles_tz_aware_bounds`）。
  - **通道熔断**（`_channel_down_until` / `_CHANNEL_COOLDOWN = 300.0` / `_CHANNEL_LABELS`）：传输层失败
    （`ConnectionError` / `Timeout` / `ProxyError` / `SSLError` / 文本含 `remotedisconnected`、
    `connection aborted`、`max retries exceeded` 等）→ **立即熔断该通道 5 分钟并换下一条**；
    限流（429）与 5xx 才原地指数退避 2s/4s/8s、连续失败 3 次暂停 30s。批量间隔 ≥1s。
  - 健康检查 `_ping()` **不受熔断影响**（用户点「重新检测」必须真探一次才能自愈），
    且**并发逐腿独立计时**（`ThreadPoolExecutor` + `time.perf_counter()`）：旧版串联探测会把
    「东财失败等满超时」的等待累计算到新浪头上，UI 显示「新浪 2010ms 延迟异常」这种假象（实际约 0.7s）。
    产出 `self.channels` 明细与 `self._primary_latency_ms`（覆盖 `h.latency_ms`）。
    `channel_status()` 供 `GET /api/data/status` 的 `akshare_channels` 字段。
  - 非 6 位 A股代码立即抛 `ValueError` 快速失败，不拖慢 fallback 链。
  - `max_lookback_days("...","1m")` 返回 7（1m 只能靠腾讯腿，覆盖约 4 个交易日），其余 None。
  - **为什么必须保留新浪**：东财在部分网络环境下会拒绝请求（真实路由返回 0 字节、不存在路由正常返回 404），
    一旦触发则 akshare 彻底失效；移除新浪会让整个 provider 变单点。
    完整取证见 README「东方财富为什么不通」——注意**不是**链路按 URL 过滤
    （跨主机对照 + 无效路由 404 两组硬证据才是否定它的依据；「HTTP/HTTPS 表现一致」是错的推理，
    WFP 的 stream 层在 TLS 加密前就能看到出站明文 URL）。退出安全软件后复测依旧全挂，
    也排除了本地安全软件。是否出现随网络环境而变，因此 UI 只显示「可用 / 暂不可用」。
- **腾讯（akshare 内部通道 / 独立 `TencentProvider` 保留）**：日/周/月走 `fqkline`（640 根/页，**按结束日期回退分页**，
  实测可回溯到 2010 年日线 / 2001 年周月线）；分钟线走**另一个端点
  `ifzq.gtimg.cn/appstock/app/kline/mkline`**（**不能带 `web.` 前缀**——`web.ifzq.gtimg.cn` 对 mkline
  会 301 重定向到不可达的 `web3.ifzq.gtimg.cn`，全部 `ConnectionError`；`fqkline` 只认 day/week/month，
  传 `m5` 一律 `bad params`）。分钟线端点无历史参数、**恒返回最近 640 根**。
  返回字段序为 `date, open, close, high, low, volume`（**open/close 在 high/low 之前**，
  按 OHLC 顺序解析会把高低价读反），成交量单位「手」需 ×100。
  **唯一支持 1m 的通道**（实测约 399 根 ≈ 1.5 天）。
- **代码推断**（`infer_symbol_type`）：6 位 A股代码 → 51/56/58（沪）或 15/16（深）开头 → ETF；
  399xxx 或 000xxx（非 .SZ）→ 指数；其余 → 股票；均映射 baostock。
  其余（如 `AAPL`）→ 股票 / yfinance。
  注意：`000001.SZ` 是平安银行（股票），`000001.SH` 是上证指数。
  已删除加密对识别（旧实现把含 `/` 的符号判为 crypto/ccxt）。
- **周期**（`quant/core/models.py` 的 `TIMEFRAMES`）：`1m / 5m / 15m / 30m / 1h / 1d / 1w / 1M`。
  已删除 `2h`/`4h`——Yahoo 对 2h 返 400、4h 返 422，baostock/akshare/腾讯均无此粒度，本就取不到数据。
  `GET /api/data/channels` 返回每个渠道对每个周期的支持情况与可回溯深度，UI 数据管理页直接展示成能力表。

### 3.3 增量更新与元信息

- 首次拉取 `lookback_days`（默认 730 天）；之后从 `last_bar_time - 重叠N根bar` 开始增量拉取，
  与本地 Parquet 按时间戳去重合并（keep=last）。
- 每次更新写 `market_data` 表：rows / start_date / end_date / last_bar_time / source / updated_at，
  UI 直接展示"数据更新时间、条数、覆盖区间"。

### 3.4 行情边界处理

- 缺失值：可配 ffill / drop（默认 ffill 且标记）。
- 停牌：零成交量 bar 标记停牌，引擎禁止交易。
- 涨跌停：按标的 `price_limit`（A股主板 10%、创业板/科创 20%、ST 5%；美股 NULL=无），
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

第 12 步新增 2 个**模型化**策略（代码在 `alpha/models/strategies/`，见 §10.2）：

| 策略 | key | 核心参数（默认值） |
|---|---|---|
| Kronos 预测 | `kronos_forecast` | context=240, pred_len=12, rebalance_every=12, threshold_sigma=0.5, min_move=0.0, position=1.0, only_long=True |
| 多模型集成 | `ensemble_forecast` | weights_json, min_return_sigma=0.3, min_abs_return=0.0, min_agreement=0.6, max_dispersion=2.0, min_models=2, context=240, pred_len=12 |

两者共同点：**阈值按波动率归一**（`threshold_sigma × σ`，σ = 近 60 根收益标准差）。
绝对百分比阈值跨周期不成立——5m 的 0.1% 已是大波动，日线要 0.5% 才算；早期用绝对值时
在 5m 上频繁开仓、日线上几乎不动。空仓时置信度强制为 0（避免 UI 出现「置信度 0.8 但仓位 0」）。

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
  - 规则：允许做空（默认关）、T+0/T+1（默认按标的：A股 T+1、美股 T+0）、lot（A股 100）。
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
| 日线/周线/月线策略 | cron：每日收盘后运行一次（A股 15:30 Asia/Shanghai / 美股 16:30 America/New_York） |
| 分钟线策略 | **cron 对齐 K 线边界**：`BAR_CRON = {"1m": ("*", 20), "5m": ("*/5", 30), "15m": ("*/15", 30), "30m": ("*/30", 30), "1h": ("1", 10)}`，时区取自标的所属市场 |
| 非标准周期 | interval 回退（`DEFAULT_POLL_SECONDS`：1m→65s、5m→315s…） |
| 组合再平衡 | 按 rebalance_freq 定时触发 |

  **为什么分钟级从 interval 改成 cron**：interval 的相位随服务运行时长漂移，可能刚好在 K 线收盘前几秒触发、拿到尚未更新的旧 bar，然后要等一整个周期才重试——表现为「明明有行情却不动」。cron 固定在每根 bar 收盘后若干秒触发，相位稳定。

- **交易时段门控**（`quant/core/market_hours.py`）：分钟级 tick 先查 `session_state(market)`，闭市直接跳过并记 `state.last_tick_skip[symbol]`，不发任何网络请求。`market_of(source, symbol)` 由数据源/代码推断市场（baostock/akshare/tencent→cn，yfinance→us）。A股含午休（11:30–13:00 判为「午间休市」）。**日线/周线/月线不门控**：cron 排在收盘后，若门控就永远拿不到当日日线。
- **启动水位线（防历史重放）**：`start(..., warmup_bars=1)` 把 `processed_ts` 设为倒数第 2 根 bar，首次 tick 只补算最近 1 根。`warmup_bars=0` 表示只等新 bar；`None` 表示全量重放（仅测试/补跑用）。水位线在每次 tick 后与 `stop()` 时写回 `paper_accounts.settings_json.processed_ts`，`recover()` 优先据此续跑——这样重启既不复读历史，也不丢停机期间的增量。
- 每个运行中的账户注册独立 job 组；`paper_accounts.status='running'` 持久化，服务重启后按状态恢复全部 job。
- 状态恢复：账户/持仓/成交/净值全部落库；重启不丢数据。
- 新 bar 到达 → `strategy.on_bar` 增量信号 → 组合合成 → Broker 撮合 → 账户更新 → 轮询推送 UI（`GET /api/paper/status` 的 `realtime` 字段给出时段与延迟）。
- **实时语义的边界**：`DataProvider.fetch_realtime()` 仍是空实现（默认 `return None`），四个渠道都没有 tick/盘口接口。撮合价是已收盘 K 线的收盘价（`execution=next_open` 取下一根开盘价），滑点为固定 bps，不按 bar 成交量限制成交量。因此这是按 K 线粒度推进的准实时模拟，不适合评估冲击成本与成交概率。

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

- **配置优先级**（第 17 步）：代码默认值 < `config/default.yaml` < `config/user.yaml` < 环境变量 / `.env`。
  `user.yaml` 是**本机覆盖层**（`quant/core/config.py:USER_CONFIG`，已 gitignore），
  用 `_deep_merge()` 深度合并，只写想覆盖的字段；升级/重装不会覆盖它。
  所有路径从 `ROOT`（软件目录，由 `__file__` 推导）出发生成，**无任何本机绝对路径硬编码**，
  因此整个目录可直接拷给其他用户使用（见 README「分发给其他用户」）。

- 调用：OpenAI 兼容 Chat Completions（`httpx` 直连，无需 SDK）；支持 `response_format: json_object`。
- **可靠性**：连接超时（默认 60s）、指数退避重试（默认 3 次）、错误分类处理
  （429 限流 → 退避重试；401 → 提示检查 key；超时 → 重试；网络错误 → 明确报错），
  全部错误返回给 UI 可读消息。
- 生成记录写入 `llm_prompts` 审计表（prompt/模型/时间/策略 id/mode/ok/detail）。

### 10.1 实现结构（第 9 步已落地）

```
quant/llm/client.py      LLMClient（chat/ping，_url 兼容各家 base_url 写法，指数退避）
                         resolve_llm_config（.env 为底 + DB 覆盖；空 key 不覆盖）
                         client_from_settings（每次调用重读设置，UI 改完即时生效）
quant/llm/prompts.py     CODE_SYSTEM_PROMPT / RULE_SYSTEM_PROMPT（含 DF 返回契约与指标签名）
                         build_messages（多轮，带 history + 上一版代码）
                         extract_code / extract_json / explanation_of
quant/strategies/code_strategy.py
                         check_source（AST：导入白名单/禁用调用/禁用属性/必须有 generate_signals）
                         exec_source（受限 builtins + pd/np/math/indicators/BaseStrategy）
                         CodeStrategy（实现覆写 .key，支持同批次多份自定义策略）
                         inspect_source / smoke_test / validate_signal_output
quant/api/routers/llm.py /llm/status|ping|chat|generate-strategy|validate|save|code|sessions
                          intent=write|revise|ideas（第 14 步：三意图正交于 mode）
                          IDEA_SYSTEM_PROMPT（策略研究顾问）→ parse_ideas / ideas_text
                          _version_of（当前版本 = 会话里最近一版 + 版本号）
```

**两种产物**：
1. `mode="code"` → Python 源码存 `strategies.source_code`，`strategy_type="code"`，
   由 `registry._default_rule_loader` 按 key 动态 `CodeStrategy(...)` 加载；
2. `mode="rule"` → 声明式 JSON 存 `config_json`，`strategy_type="llm_generated"`，
   仍走 `RuleStrategy` + `validate_config`（与手动表单完全同一条链）。

**三意图（第 14 步）**：`intent` 与 `mode` 正交，决定系统提示与输出契约——
`write`（从零写 / 改需求重写）、`revise`（请求带上一版完整 code 或 config，LLM 整体重写而非给片段）、
`ideas`（换 IDEA_SYSTEM_PROMPT，只返回 JSON 想法数组 title/logic/scene/risk，不产出策略；
未显式带 code 时后端自动沿用会话里最近一次生成的代码/配置，见 `llm.py` `llm_chat`）。

**为什么保存前必须冒烟回测**：静态检查只能发现语法/导入/命名问题，抓不到「列名错、行数不一致、
权重越界、运行期异常」——这些只有真跑一次 `generate_signals` 才暴露。合成 240 根K线（趋势+波动+噪声）
成本极低，却拦住了绝大多数坏代码，也保证手改后同样不会把策略改坏。

**会话上下文**：`llm_sessions` / `llm_messages` 持久化；下一条消息自动带上会话里最近一条带 code 的
assistant 消息，因此"加个止损""换成放量条件"这类增量指令无需重述完整需求。
第 14 步起「当前版本」由 `llm_chat` 的 `version` 字段（`_version_of`）提供：每成功生成一版 +1，
UI 右侧「🧾 当前版本」面板实时预览该版代码/配置、校验状态与冒烟指标，刷新页面后从会话消息还原。

**安全边界**：受限命名空间 + AST 白名单是防误操作的软约束，不是对抗恶意代码的沙箱。
UI 与文档均明确提示：只运行自己或可信 AI 生成的策略。

---

### 10.2 金融时序模型与多模型集成（`alpha/models/`，第 12 步）

**分层动机**：`quant/` 是基础设施（数据、引擎、撮合、存储），`alpha/` 是业务能力。
后者按「要不要训练/加载模型」二分：

- `alpha/factors/` — **因子化**：只用历史数据算指标，无状态、零依赖、可解释。
  8 个经典策略（双均线/MACD/RSI/布林/动量/均值回归/网格/海龟）在此。
- `alpha/models/` — **模型化**：需要权重或训练的预测模型，输出「未来价格路径」而非直接给信号。
  `base.py` 统一接口，`baselines.py` 零依赖统计基线，`kronos.py` / `external.py` 深度学习模型，
  `ensemble.py` 把多个模型的预测加权整合，`strategies/` 把预测转成持仓。

**统一预测接口（`alpha/models/base.py`）**

```python
class BaseForecaster:
    key: str; label: str; kind: str        # kind ∈ stat | dl | external
    def status(self) -> dict               # 不触发加载
    def forecast(self, bars, pred_len) -> DataFrame[open,high,low,close,volume,amount]
    def expected_return(self, bars, pred_len) -> float | None   # 末根预测收盘/当前收盘-1
```

公共工具：`prepare_bars`（校验 DatetimeIndex、补 `amount = volume × 均价`、价格 NaN 丢行而非填 0、
按 `max_context` 只留尾部）、`future_index`（用已有索引间隔的**中位数**外推，一套代码服务 1m/5m/日/月）、
`return_sigma`（近 60 根收益标准差）、`path_to_ohlcv`（由收盘路径补 OHLCV，**open 用上一根收盘**）。

**可用模型**

| key | kind | 说明 | 依赖 |
|---|---|---|---|
| `kronos` | dl | Kronos-small/base/mini，K 线语言模型，AAAI 2026（arXiv:2508.02739，MIT） | torch/einops/safetensors/huggingface_hub/tqdm |
| `random_walk` | stat | 零假设锚点：未来收盘 = 当前收盘（μ≡0） | 无 |
| `drift` | stat | 250 根平均对数收益外推 | 无 |
| `momentum` | stat | 20 根平均对数收益，逐根衰减（damping=0.03） | 无 |
| `reversion` | stat | 向 20 日均线回归（speed=0.15，horizon=10） | 无 |
| `ewma` | stat | 对数收益的 EWMA（halflife=12） | 无 |
| `chronos` | external | Amazon Chronos-T5，取 20 条采样路径的**中位数** | torch + `chronos-forecasting` |
| `timesfm` | external | Google TimesFM 1.0-200m | torch + `timesfm` |
| `moirai` | external | Salesforce Moirai-1.0-R-small | torch + `uni2ts` |

统计基线**零依赖**且解释性强，是「模型到底有没有用」的对照基线——如果 Kronos 打不过
`random_walk`，说明该标的/周期上根本没有可预测性，不该往下调门控去凑信号。

**加权集成（`alpha/models/ensemble.py`）**

```
combined   = Σ(w_i × ret_i) / Σw_i                       # 只统计 weight>0 且成功的模型
agreement  = 与 combined 同向的权重和 / Σw_i
dispersion = sqrt(Σ(w_i × (ret_i − combined)²) / Σw_i)
```

五道**硬性门控**（任一不过即目标仓位 0，并在 `reason` 里写明原因与实测值）：

| 门控 | 默认 | 判据 |
|---|---|---|
| `min_models` | 2 | 有效投票数 ≥ N（防单模型独断） |
| `min_return_sigma` | 0.3 | `|combined| ≥ 0.3 × σ`（幅度要够大） |
| `min_abs_return` | 0.0 | `|combined| ≥` 绝对下限（补足 σ 极小时的缝隙） |
| `min_agreement` | 0.6 | 同向权重占比 ≥ 60%（防分歧） |
| `max_dispersion` | 2.0 | `dispersion ≤ 2.0 × σ`（防个别模型极端） |
| `direction` | — | 只做多时 `combined > 0` 才开仓 |

`combined` 的符号决定方向，`position` 决定幅度，`confidence = clip(|combined| / max(阈值, min_abs_return) / 3, 0, 1)`。

**`auto_weights` 的口径**：在历史上滚动取评估点，算每个模型的方向命中率 `hit` 与预测收益/实际收益的
Pearson `ic`，`score = max(0, ic) × max(0, 2·hit − 1) × 2` 后归一化。**负相关或命中率低于 50% 直接给 0 分**
——反向指标不该靠负权重融合（负权重在不同市场状态下翻脸，稳定性差），直接弃用更稳。
文件头明确写着「这不是严格样本外评估，只能当初始猜测」；真正的 walk-forward 在后续步骤里。

**策略化**：`alpha/models/strategies/kronos_forecast.py` 与 `ensemble_forecast.py` 都把「预期收益」
按波动率归一后映射成仓位（见 §4.2）；`on_bar` 按 `history.index[-1]` 缓存，同一根 K 线不重复预测。

**性能实测（本机 RTX 4060 Laptop，实跑 CPU）**：Kronos-small 首次加载 4.3~5.5s（24.7M 参数，
含下载约 50s），单次预测 0.6~0.7s；`POST /api/ml/kronos/predict`（240 根 5m 上下文 → 预测 12 根）1.5s；
`POST /api/backtest/combo`（000001.SH 5m + `kronos_forecast`）2.5s。**延迟瓶颈在首次加载，不在推理。**

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
  strategy_type TEXT NOT NULL DEFAULT 'builtin',   -- builtin | llm_generated | code
  class_name TEXT,
  config_json TEXT NOT NULL DEFAULT '{}',          -- params + LLM 声明式配置
  source_code TEXT DEFAULT '',                     -- 仅 strategy_type='code'：Python 源码
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

-- LLM 会话 + 消息（多轮增量修改的上下文靠它们持久化）
CREATE TABLE IF NOT EXISTS llm_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT DEFAULT '新对话', mode TEXT DEFAULT 'code',
  strategy_id INTEGER, created_at TEXT DEFAULT (datetime('now')), updated_at TEXT
);
CREATE TABLE IF NOT EXISTS llm_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL, role TEXT NOT NULL,
  content TEXT DEFAULT '', code TEXT DEFAULT '', meta_json TEXT DEFAULT '{}',
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_llm_messages_session ON llm_messages(session_id, id);

-- LLM 审计 + 通用键值
CREATE TABLE IF NOT EXISTS llm_prompts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt TEXT, strategy_id INTEGER, model TEXT, created_at TEXT,
  mode TEXT DEFAULT 'code', ok INTEGER DEFAULT 1, detail TEXT DEFAULT ''
);
-- kv_store：新功能配置不必每加一项就改 app_settings 列结构
-- 目前只用于 ml.ensemble（多模型集成配置）
CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value_json TEXT);
```

> 第 14 步说明：**没有新增表**。「当前版本」快照不落库——它由 `llm_messages` 的 `code` 列与
> `meta_json`（intent/mode/valid/config/smoke）现场算出：`llm_chat` 的 `_version_of()` 统计
> 会话里非 ideas、非 error 的 assistant 消息条数得版本号 n，取最近一条的 code/config 即当前版本。
> 因此反复修改的每一版都完整保留在会话消息里，换会话/刷新页面即可还原。

---

## 13. API 设计（FastAPI，前缀 /api）

```
GET    /api/health
-- 数据 --
GET    /api/data/symbols        POST /api/data/symbols      DELETE /api/data/symbols/{symbol}
GET    /api/data/status         # 每个 symbol×timeframe：覆盖区间/最新 bar/文件 mtime/stale 标记/adjust + 数据源健康（含 akshare 逐通道明细）+ akshare_channels（三腿熔断状态）
POST   /api/data/update         # 一键更新 {symbols?, timeframes?, incremental?, adjust?, full_reload?, start_date?}
GET    /api/data/bars/{symbol}  # ?timeframe=&start=&end= → OHLCV
GET    /api/data/channels       # 每个渠道对每个周期的 supports() + max_lookback_days()（纯静态，不发网络请求）；
                                # akshare 额外返回 sub_channels（东财/腾讯/新浪三腿各自的周期能力与实测深度）
-- 策略 --
GET    /api/strategies          POST /api/strategies
GET    /api/strategies/{id}     PUT  /api/strategies/{id}    # 参数/权重/启用
POST   /api/strategies/{id}/toggle     DELETE /api/strategies/{id}  # 内置→403；被组合引用→级联清理
GET    /api/strategies/builtins GET  /api/indicators         # 规则策略表单元信息（指标白名单/操作符/仓位模式）
POST   /api/strategies/rule     # 保存声明式规则策略（手动表单/LLM 共用校验链，草稿 enabled=0）
-- 组合 / 设置 --
GET/PUT /api/portfolio          GET/PUT /api/settings/trading
GET/PUT /api/settings/llm
GET/PUT /api/settings/proxy     # 逐渠道代理：auto / direct / proxy，立即生效（存 kv_store）
GET/PUT /api/settings/paths     # 本机路径：查数据目录/DB/行情目录体积与可写性；改 data_dir 写 config/user.yaml
-- 回测 --
POST   /api/backtest/run        {engine: vectorized|event_driven, ...}
GET    /api/backtest/runs       GET /api/backtest/runs/{id}
GET    /api/backtest/runs/{id}/equity    GET /api/backtest/runs/{id}/trades
-- 模拟盘 --
POST   /api/paper/start         POST /api/paper/stop
DELETE /api/paper/accounts/{id} # 仅 stopped 可删（运行中 409），?cleanup= 控制清理记录
GET    /api/paper/status        GET /api/paper/account
GET    /api/paper/positions     GET /api/paper/trades
GET    /api/paper/equity        # 实时刷新由 UI 轮询（无 WebSocket 端点）
-- 系统 --
POST   /api/system/shutdown     # 一键关闭：停模拟盘账户（落库）→ 优雅退出 uvicorn
-- LLM（AI 策略工坊，第 14 步）--
GET    /api/llm/status          POST /api/llm/ping
POST   /api/llm/chat            # 多轮对话：自然语言 → 代码/规则 + 自动校验，会话持久化
                                # body: {message, session_id?, mode=code|rule,
                                #        intent=write|revise|ideas, code?, config?, context?}
                                # 返回 {session_id, message, result, version} 或 ideas 模式的 {ideas}
POST   /api/llm/generate-strategy  # 单轮无状态生成
POST   /api/llm/validate        # 校验代码（静态检查+冒烟回测）或规则配置
POST   /api/llm/save            # 保存为策略（code → source_code 列；rule → config_json）
                                # body: {mode, name?, key?, strategy_id?, code|config, params?, enabled?, session_id?}
GET    /api/llm/code/{id}       PUT /api/llm/code/{id}   # 读取/覆盖策略源码（手改，内置→403）
GET/POST /api/llm/sessions      GET/DELETE /api/llm/sessions/{id}
-- WorkBuddy 模型接入（复用本机登录态，可选模型来源）--
GET    /api/workbuddy/status      # 登录态/昵称/积分/模型列表（可用性 + 错误）
GET    /api/workbuddy/models      # 模型目录
POST   /api/workbuddy/ping        # 轻量对话自检（glm-5.3-flash）
POST   /api/workbuddy/chat        # SSE 流式对话（chunk JSON，末尾 [DONE]）
                                  # /api/llm/chat 带 provider=workbuddy + model 即走 WorkBuddy 生成
-- 金融时序模型（Kronos + 多模型集成）--
GET    /api/ml/models              # 全部可用模型及可用性（不触发加载）
GET    /api/ml/kronos/status       POST /api/ml/kronos/load
POST   /api/ml/kronos/predict      # {symbol, timeframe, bars, pred_len, model, sample_count}
GET/PUT /api/ml/ensemble/config    # 集成权重与门控（存 kv_store；负权重截断为 0）
POST   /api/ml/ensemble/predict    # 逐模型投票 + 门控明细（weights 可临时覆盖、不写库）
POST   /api/ml/ensemble/auto-weights  # 同样本命中率+IC 打分给建议权重（不写库）
-- 模型 × 因子融合--
GET    /api/fusion/config         # 融合配置 + 可选模型/因子清单 + 融合方式说明
PUT    /api/fusion/config         # 合并保存（权重只留非负有限值；未知 key 丢弃）
POST   /api/fusion/preview        # 用真实行情试算一次融合决策（config 可临时覆盖、不写库）
GET    /api/fusion/options        # 可选模型/因子清单 + 融合方式说明
```

---

## 14. Web UI 页面（Streamlit）

导航分四个模块（外加系统设置），侧边栏「模块」下拉 + 「页面」单选：

| 模块 | 页面 |
|---|---|
| 数据管理 | 数据管理 |
| 策略管理 | AI 策略、模型预测、策略创建、策略管理、策略融合、组合配置 |
| 策略回测 | 策略回测 |
| 交易模拟 | 交易模拟 |
| 系统设置 | 数据设置、交易设置、策略设置 |

页面之间用 `st.session_state["nav_target"] = "<页面名>"` 做程序化跳转（值为**页面名**，
由 `streamlit_app.py` 在 radio 实例化前换算成「模块 + 子页」选中值）。

| 页面 | 功能 |
|---|---|
| 数据管理 | 标的池增删、资产类型/数据源/周期选择、**一键更新**、数据状态表（行数/区间/更新时间）+ 数据源健康（始终展示全部）、行情预览 |
| AI 策略 | **AI 策略工坊（第 14 步）**：三标签（`AI 对话` / `代码编辑` / `我的 AI 策略`，用 `st.segmented_control` 切换而非 tabs，保证程序化跳转可靠）；三意图（`✨ 新策略` write / `🔧 改这版` revise / `💡 要建议` ideas）与模式（code / rule）正交；历史对话下拉（切换/新建，消息数显示）；右侧「🧾 当前版本」实时预览（校验状态、冒烟指标、完整代码/配置，每轮对话结束即刷新，无需手动刷新）；「保存到策略库」（一键保存用 AI 起的名字存草稿 / 保存选项表单改名描述并可选立即启用）；代码编辑标签可手改源码后保存（过同一套校验链）或「让 AI 改这段代码」回喂对话；我的 AI 策略列表（启停/编辑代码/表单编辑/删除/参数表单/快速回测）。未配置 API Key 时提示去「系统设置 → 交易设置 → 大模型设置」并提供跳转按钮 |
| 策略创建 | **手动条件规则表单**：指标白名单（11 个）→ 买卖/做空/平空条件组（all/any、9 操作符）→ 仓位（固定比例/金额/ATR/信号强度）→ 实时 JSON 预览 → 保存草稿 / 保存并回测 → 激活；**顶部「AI 辅助」面板**：自然语言生成/改写配置 → 校验通过后一键载入表单 → 继续逐项手调（两种产物共用 schema 与校验链） |
| 模型预测 | **Kronos 金融时序基础模型**：状态四指标（模型/设备/参数量/加载耗时）、按需加载按钮、预测表单（标的/周期/上下文长度/预测长度/采样次数）、预测幅度指标、历史+预测 ECharts 曲线（预测段虚线+高低区间）、预测明细表、去回测按钮；缺依赖时显示安装指引与模型简介 |
| 策略管理 | 内置 + 手动 + **AI 代码策略**卡片：启用开关、schema 参数表单、**编辑跳转创建页 / 改代码**、**删除**（内置置灰/二次确认/组合引用级联清理） |
| 策略融合 | **大模型 × 因子自由融合**（见 §16）：模型权重（按 dl/stat/external 分组，未装依赖的置灰）、5 道门控、因子权重、融合方式（加权混合/模型闸门/因子闸门/投票制）、模型侧份额、置信度加权、保存 → 试算（不写库）→ 去回测 |
| 组合配置 | 三模式、权重滑杆（-100~100）、信号阈值/再平衡、组合信号分解预览 |
| 数据设置 | **数据源代理**（逐渠道 `auto`/`direct`/`proxy`，存 `kv_store` 立即生效）+ **存储与路径**（软件根目录 / Python 解释器 / 数据目录 / 数据库 / 行情缓存 5 行路径表，含存在性与占用；可用表单改 `data_dir` 写 `config/user.yaml`，提示需重启后端且不自动迁移历史数据） |
| 交易设置 | 手续费、频度（周期/最小间隔/冷却/再平衡）、资金与仓位、做空/T+0/T+1、风控参数、基准下拉预置 |
| 策略设置 | **大模型设置**（API Key/Base URL/模型/重试/超时/温度 + 测试连接 + 探测可用模型）+ **WorkBuddy 连接**（状态/模型列表/自定义 exe 路径/测试连接/保存并校准） |
| 策略回测 | 引擎模式切换、运行、指标卡片（含**超额收益 vs 买入持有/Beta/Alpha/信息比率/基准最大回撤**）、**归一化收益对比图**（策略/买入持有/基准/标的股价）、**回撤对比图**（策略 vs 基准）、**月度收益热力图**、K线+买卖点+成交量（ECharts）、交易明细、多策略对比 |
| 交易模拟 | 开始/停止、**删除账户**（stopped 才可删）、调度信息、实时权益/持仓/成交、capital_split 子账户资金池 |
| 侧栏 | 模块/页面导航、环境标识（本地/沙箱）、**关闭系统**（红色按钮 + 二次确认 → `POST /api/system/shutdown` → 停模拟盘 → 优雅退出） |

---

## 15. 项目目录结构

```
QuantSim/
├── start.bat                   # 一条命令启动（自动建 venv、装依赖、起服务）
├── requirements.txt / requirements-optional.txt / requirements-ml.txt
├── pyproject.toml
├── .env.example                # LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
├── config/default.yaml         # 应用默认配置（数据源优先级/默认标的/默认周期）
├── config/user.example.yaml    # 本机覆盖层模板（复制成 user.yaml 后生效）
├── config/user.yaml            # 本机覆盖层（.gitignore，不进版本库）
├── docs/ARCHITECTURE.md        # 本文档
├── alpha/                      # ★ 业务能力层：因子化 + 模型化
│   ├── factors/strategies/     # 因子化：8 个经典规则策略（双均线/MACD/RSI/布林/
│   │                           # 动量/均值回归/网格/海龟），只用历史数据算指标
│   └── models/                 # 模型化：需要训练/权重的预测模型
│       ├── base.py             # BaseForecaster / prepare_bars / return_sigma / future_index
│       ├── baselines.py        # 5 个零依赖统计基线
│       ├── kronos.py           # Kronos 封装（按需加载，未装 torch 不影响主流程）
│       ├── external.py         # Chronos / TimesFM / Moirai 适配器
│       ├── ensemble.py         # 加权集成 + 5 道硬性门控 + auto_weights
│       ├── _kronos/            # 上游 Kronos 源码内联（MIT，见 NOTICE.md）
│       └── strategies/         # 模型化策略：kronos_forecast / ensemble_forecast
├── quant/                      # 基础设施
│   ├── core/                   # models / indicators / config / utils / market_hours
│   ├── data/                   # base / yfinance_provider / baostock_provider /
│   │                           # akshare_provider / tencent_provider / storage / manager
│   ├── strategies/             # base / registry / rule_strategy / code_strategy / llm/
│   ├── portfolio/portfolio.py
│   ├── engine/                 # backtest / paper / broker / risk / metrics
│   ├── llm/                    # client / prompts（OpenAI 兼容 + 提示词）
│   ├── api/                    # app / state / routers/（含 llm.py / ml.py）
│   └── storage/                # db / repositories（含 KVRepo / LLMRepo）
├── streamlit_app.py            # UI 入口（Streamlit，四模块 + 系统设置）
├── web/streamlit/              # api.py（薄客户端）/ components.py / pages/
├── scripts/                    # update_data.py / run_backtest.py / run_paper.py
├── tests/                      # pytest
└── data/                       # 运行时：market/*.parquet、quant.db
```

---

## 16. 配置与默认值

- 所有可调参数（策略参数、手续费、频度、仓位、权重、风控、LLM）**均带默认值**，空配置可直接运行。
- 优先级：代码默认值 < `config/default.yaml` < `config/user.yaml` < 环境变量 / `.env`。
- `ml.kronos`：`enabled`(true) / `model`(Kronos-small) / `tokenizer`(空=自动配对) / `device`(auto) /
  `max_context`(512) / `pred_len`(12) / `temperature`(1.0) / `top_p`(0.9) / `sample_count`(1) / `tz`(Asia/Shanghai)。
- `ml.ensemble`（存 `kv_store`，不进 yaml）：`weights`(见 `DEFAULT_WEIGHTS`) / `min_models`(2) /
  `min_return_sigma`(0.3) / `min_abs_return`(0.0) / `min_agreement`(0.6) / `max_dispersion`(2.0) /
  `position`(1.0) / `allow_short`(false) / `pred_len`(12) / `context`(240)。
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
| 7 | LLM 策略生成（DeepSeek 等 OpenAI 兼容接口）✅ | `quant/strategies/llm/`（schema+validator）+ `quant/llm/`（client/prompts）+ `quant/strategies/code_strategy.py`（AST 检查/受限执行/冒烟回测）；`POST /api/strategies/rule` 与 `POST /api/llm/*` 双通道 |
| 8 | 测试、文档、示例配置收尾 | `pytest -q`（138 全过）；README 全量验收 + `config/example.yaml` |
| 9 | **AI 策略生成 + 策略代码可手改** ✅ | 新增 `web/streamlit/pages/ai_page.py`（对话/代码编辑/我的策略三个视图，`st.segmented_control` 按需渲染）；`/api/llm/chat` 多轮 + `/api/llm/code/{id}` 读写源码；`strategy_type="code"` 由 registry 动态编译加载；167 测试全过 |
| 10 | **数据源精简与下载修复** ✅ | 删除 ccxt 加密源与期货支持；A股改为 `baostock → akshare → 腾讯` 三渠道按序降级（各 provider 声明 `supports()`/`max_lookback_days()`，调度前先问能力、不支持的周期直接跳过）；新增 `quant/data/tencent_provider.py`（日/周/月走 fqkline 分页、分钟线走独立的 mkline 端点）；周期由 10 项精简为 8 项（删 2h/4h）；新增 `GET /api/data/channels` 渠道能力表；176 测试全过 |
| 11 | **分钟级实时模拟（5m）** ✅ | 新增 `quant/core/market_hours.py`（A股/美股交易时段、午休、周末、下次开盘倒计时）；`paper.py` 分钟级调度由 interval 改 **cron 对齐 K 线边界**、新增交易时段门控（日线不门控）、**启动水位线防历史重放**（旧实现实测重放 336 根 5m K 线产生 6 笔历史成交）；水位线逐 tick 落库供 `recover()` 续跑；`GET /api/paper/status` 增加 `realtime` 字段（时段/延迟/跳过原因）；模拟盘页新增实时面板；新增 `tests/test_paper_realtime.py`（17 例）；193 测试全过 |
| 12 | **金融时序大模型 + 多模型加权集成 + `alpha/` 分层** ✅ | 新增顶层 `alpha/`（`factors/` 因子化 + `models/` 模型化），把 10 个策略从 `quant/strategies/builtin` 迁入；`alpha/models/`：`base.py`（`prepare_bars`/`return_sigma`/`future_index`/`BaseForecaster`）、`baselines.py`（5 个零依赖统计基线）、`kronos.py`（按需加载，未装 torch 不报错）、`external.py`（Chronos/TimesFM/Moirai 适配器）、`ensemble.py`（加权集成 + 5 道硬性门控 + auto_weights）；`alpha/models/_kronos/` 内联上游 MIT 源码（改相对导入）；策略阈值改为**按波动率归一**；新增 `GET /api/ml/models`、`/api/ml/kronos/*`、`/api/ml/ensemble/*`；新增「模型预测」页；策略创建页加「AI 辅助」面板（`rc_ver` 版本号解决 Streamlit key 缓存）；271 测试全过 |
| 13 | **导航四模块化 + 大模型 × 因子策略自由融合** ✅ | 导航重构为「数据管理 / 策略管理 / 策略回测 / 交易模拟」四模块 + 系统设置（`streamlit_app.py`：模块 selectbox + 子页 radio，`nav_target` 写页面名再换算），AI 大模型预测页归入策略管理；新增 `alpha/models/strategies/fusion.py`（`FusionStrategy`，key=`fusion`）：内部同时跑 `EnsembleForecaster` 与因子策略，四种融合方式（加权混合/模型闸门/因子闸门/投票制）+ 模型侧份额 + 置信度加权，单侧失效自动降级；新增 `quant/api/routers/fusion.py`（`GET/PUT /api/fusion/config`、`POST /api/fusion/preview`、`GET /api/fusion/options`）；新增「策略融合」页（模型/因子权重、门控、保存、试算、去回测）；融合配置存 kv_store `ml.fusion`，策略参数可覆盖；295 测试全过 |
| 14 | **AI 策略工坊（对话生成 / 反复修改 / 思路建议 / 版本实时预览 / 一键保存）** ✅ | `quant/llm/prompts.py` 增加 `intent`（write/revise/ideas，与 mode 正交）：`revise` 用 `CODE_REVISE_TEMPLATE`/`RULE_REVISE_TEMPLATE` 带上一版完整代码/配置，`ideas` 换 `IDEA_SYSTEM_PROMPT`（策略研究顾问）+ `parse_ideas`/`ideas_text` 解析想法数组；`llm_chat` 未显式带 code/config 时自动沿用会话里最近一版，返回 `version`（`_version_of`：版本号 n + 最近一版 code/config/valid）；`ai_page.py` 重构为三标签（AI 对话 / 代码编辑 / 我的 AI 策略）+ 三意图（✨新策略/🔧改这版/💡要建议）+ 历史对话下拉 + 右侧「🧾 当前版本」实时预览 + 「保存到策略库」（一键 / 保存选项表单）；思路卡片带「按这个思路写策略」一键转 write；未配置 Key 时引导去「系统设置 → 交易设置 → 大模型设置」；新增 `tests/test_ai_chat.py`（18 例）与 `tests/test_ai_page.py`（4 例）；317 测试全过 |
| 15 | **WorkBuddy 模型接入（复用本机登录态，可选模型来源）** ✅ | `quant/llm/workbuddy.py`：读 WorkBuddy 桌面 App 的 `workbuddy-desktop.info`，AES-256-GCM 解密 accessToken/refreshToken（密钥从 Electron binding `electron_browser_workbuddy_storage`.loggerGet() 取 atRestSecretKey），`WorkBuddyClient` 提供 models()/credits()/chat()/chat_stream()；`quant/api/routers/workbuddy.py`：`GET /status`、`GET /models`、`POST /ping`、`POST /chat`(SSE)；`/api/llm/chat` 加 `provider`/`model` 参数，provider=workbuddy 时走 WorkBuddy 生成；系统设置页加「WorkBuddy 连接」卡片，AI 策略工坊加「模型来源」下拉（DeepSeek/OpenAI 兼容 ↔ WorkBuddy）；`requirements.txt` 加 `cryptography>=42`；`tests/test_workbuddy.py`（9 例）；327 测试全过 |
| 16 | 绿色便携版打包（已回退，不作数）❌ | 曾实现 `launch/quantsim_launcher.py`（tkinter 启动器：一键启停/检查更新/卸载）+ `launch/build_portable.py`（PyInstaller onedir 打包 972MB / zip 278MB）。**用户已还原项目，此步作废**，根目录无 `launch/`、`dist_portable/`、`quantsim.spec` |
| 17 | **本机专属参数可配置化（为分发做准备）** ✅ | 新增 `quant/core/config.py::USER_CONFIG = ROOT/"config"/"user.yaml"` 本机覆盖层（优先级 代码默认 < default.yaml < **user.yaml** < 环境变量；传显式 yaml_path 时不叠加，便于测试）+ 模块级 `_deep_merge`（递归合并 dict、**列表整体替换**、None 覆盖）；新增 `GET/PUT /api/settings/paths`（`_path_info` 返回 path/exists/writable/is_dir/size_bytes；PUT 校验绝对路径、mkdir + `.write_test` 探活、写 user.yaml、空串则清除覆盖、返回 `restart_required`）；`web/streamlit/pages/settings_page.py` 新增「存储与路径」区（5 行路径表 + 目录迁移展开器）与「数据源代理」区（从 data_page 迁入）；`web/streamlit/components.py` 补 `PROXY_MODE_LABELS`/`PROXY_MODE_HINTS`；`quant/llm/workbuddy.py` 删除硬编码 `D:\Insert\WorkBuddy\WorkBuddy.exe`，改为纯环境变量候选链；新增 `config/user.example.yaml`；`.gitignore` 加 `/config/user.yaml`；新增 `tests/test_user_config.py`（11 例）；358 测试全过 |
| 18 | **数据源健康重构：腾讯并入 akshare + baostock 会话保持** ✅ | ①`tencent_provider.py` 抽出可复用原语 `_get_json`/`_fetch_minute_rows`/`_fetch_daily_rows`/`_to_tx_code`；②`akshare_provider.py` 变 `东财 → 腾讯 → 新浪` 三腿，`_channels(timeframe)` **顺序随周期动态决定**（1m 只有腾讯；分钟线 新浪 1970 根 ≫ 腾讯 640 根故新浪优先；日线以上腾讯更快故腾讯优先），新增 `_clip_to_window` 裁剪腾讯越界返回；③`manager.py` 移除 `TencentProvider`，provider 链变 `[baostock, akshare]`；④`_ping()` 改**并发逐腿独立计时**，`latency_ms` 只报主用通道耗时（旧串联探测把东财超时算到新浪头上，UI 假报「新浪 2010ms」）；⑤**baostock 会话保持**：`login()` 实测 11~13s、query 仅 2~3s，原「每条查询 login→logout」是「总超时」真因；改 `_ensure_login`/`_force_relogin`/`close_session`/`session_state` + `_QUERY_RETRIES=2` 掉线重登 + socket 超时 15s→30s + `DataManager.warm_up()` 启动预热（健康检查从超时降到约 2 秒）；⑥UI 健康表变 `baostock / akshare(通道明细) / yfinance` 三项，删掉多余 caption，`GET /api/data/channels` 增 `sub_channels`；⑦`TF_COVERAGE_HINT` 按实测重写；363 测试全过（1 skip） |
| 19 | **数据设置独立成页 + 健康检测去本机化 + AI 策略页回顶** ✅ | ①新增 `web/streamlit/pages/data_settings_page.py`（「数据源代理」+「存储与路径」两块，从 `settings_page.py` 整体迁出），导航「系统设置」变 `[数据设置, 交易设置, 策略设置]`，数据设置排最前；`data_page.py` 只留一行指向提示；②界面与接口**去掉一切「这台电脑才成立」的结论**：`_akshare_sub_channels()` 的 `timeframes` 由具体根数改布尔、删 `note` 字段，`data_page.py` 三腿表只显示 ✓/—、`_channel_state_label` 只返回「可用 / 暂不可用」（不再拼「N 秒后重试」）、`_channels_brief` 不显示熔断剩余秒数、`TF_COVERAGE_HINT` 改为无数字表述，代码里的「本机被东财侧风控拉黑」「实测本机被 baostock 拉黑」等表述改为中性描述；③`web/streamlit/components.py` 新增 `scroll_to_top()`（多延时点向 `section.main`/`stMain`/`body` 写 `scrollTop=0`），`streamlit_app.py` 记录 `_page_changed`，`ai_page.render()` 换页时消费一次 → 打开即停在页面顶部；④`quant/data/baostock_provider.py` 双冷却状态机（`_FATAL_COOLDOWN=1800` / `_DOWN_COOLDOWN=120`）+ `LoginRejected`（服务器明确拒绝不重试，避免被 4 次登录假失败打穿）+ 线程局部超时（修掉 `_ping(timeout=6.0)` 永久污染全局 `_SOCKET_TIMEOUT` 的老 bug）+ `_ping()` 去掉 6s 传参并先查 `fatal_reason()`/预热占锁短路；⑤`quant/llm/workbuddy.py` 新增 `_drive_candidates()` 全盘符探测（替代只删不补的硬编码）；364 测试全过（1 skip） |

---

## 18. 验收标准对照

| 验收项 | 落地位置 |
|---|---|
| 一条命令启动 | `start.bat`/`start.sh` |
| 一键更新数据并看到结果 | 数据管理页 + `POST /api/data/update`（含渠道能力表 `GET /api/data/channels`） |
| 下载失败可定位原因 | 数据管理页「下载渠道能力表」（各源 ✓/— + 可回溯深度）+ 结果反馈表的「下载渠道/跳过说明」列 |
| 内置策略选择运行回测 | 策略管理页 + 回测页 |
| 切换策略/调参/手续费/频度 | 策略管理页 + 交易设置页 |
| 策略权重与组合回测 | 策略管理页（权重滑杆）+ 组合模式 |
| 自然语言创建策略并回测 | **AI 策略工坊**（对话 → 代码/规则 → 自动校验 → 保存 → 快速回测）；`POST /api/llm/chat` |
| 复用 WorkBuddy 模型/积分生成策略 | **模型来源下拉选「WorkBuddy」**；`POST /api/llm/chat`（provider=workbuddy）；系统设置「WorkBuddy 连接」卡片看账号/积分/模型 |
| 在已有策略之上让 AI 改（而不是从零写） | **策略创建页「AI 辅助」面板**：勾选「把当前表单的条件一起发给 AI」→ 生成后一键载入表单 → 继续手调；`_ai_message` 把当前 JSON 拼进 prompt |
| 接入金融专用预测模型 | **模型预测页**（Kronos 状态/加载/预测/曲线）；`GET /api/ml/kronos/status`、`POST /api/ml/kronos/predict` |
| 多模型加权整合 | `alpha/models/ensemble.py`（`EnsembleForecaster`）+ `GET/PUT /api/ml/ensemble/config`、`POST /api/ml/ensemble/predict`（逐模型投票 + 门控明细） |
| 用硬性指标把不合格信号挡掉 | 5 道门控 `min_models`/`min_return_sigma`/`min_abs_return`/`min_agreement`/`max_dispersion` + `direction`，任一不过即空仓并在 `reason` 说明 |
| 生成结果可手动改代码 | **AI 策略页「代码编辑」视图**（校验/保存/回喂 AI）；`GET/PUT /api/llm/code/{id}` |
| 反复对话修改直到满意 | **AI 策略工坊「🔧 改这版」**（`intent=revise`）：上一版完整代码/配置 + 修改要求一起发给 LLM 整体重写；请求未显式带 code/config 时后端自动沿用会话里最近一版；右侧「🧾 当前版本」每轮实时刷新，无需手动刷新 |
| AI 给策略思路（而不只是写代码） | **「💡 要建议」**（`intent=ideas`）：`IDEA_SYSTEM_PROMPT` 以策略研究顾问身份返回 title/logic/scene/risk 想法数组，每条带「按这个思路写策略」一键转 `intent=write` |
| 自由配置需要哪些大模型 + 权重 | **策略融合页**（模型权重按 dl/stat/external 分组，未装依赖置灰）+ `PUT /api/fusion/config` |
| 自由配置需要哪些因子策略 + 权重 | **策略融合页**（8 个因子权重滑杆）+ 同一 `PUT /api/fusion/config` |
| 大模型与因子策略自由结合 | 四种融合方式 + 模型侧份额 + 置信度加权（`alpha/models/strategies/fusion.py`），改完即存、回测/模拟盘即时生效 |
| 导航按模块整合 | 四模块导航（数据管理/策略管理/策略回测/交易模拟）+ 系统设置；AI 大模型预测归入策略管理 |
| 完整指标/交易明细/图表 | 回测页 |
| 模拟交易持续运行并更新 | 模拟交易页（APScheduler + 推送） |
| 配置保存并下次自动加载 | SQLite 持久化 |
| 所有参数有默认值 | Pydantic 模型 + app_settings 默认值 |
