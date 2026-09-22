# QuantSim — 本地量化交易模拟系统

本地运行的量化交易模拟系统：历史回测（向量化/事件驱动双模式）、实时模拟交易（paper trading）、
一键更新数据、8+ 内置经典策略、手动创建条件规则策略（LLM 生成接口已预留）、策略组合与权重。

> 当前进度：**第 8 步（测试 / 文档收尾）已完成**——138 个测试全过（含 baostock 真实拉取用例）。
> 完整架构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，引擎设计见 [docs/BACKTEST_ENGINE_DESIGN.md](docs/BACKTEST_ENGINE_DESIGN.md)。

## 环境要求

- Python 3.11+（本机 3.14 已验证）
- 网络（拉取免费行情源：baostock / yfinance / akshare / ccxt）

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
pip install -r requirements-optional.txt   # 可选：baostock（A股股票/指数/ETF 主源）+ akshare（A股备用/期货）
```

> Python 3.14 若 akshare 安装报 jsonpath 构建失败：
> `pip install setuptools && pip install jsonpath --no-build-isolation` 后重试。

## 一键更新数据（已验证）

```bash
# 更新全部启用标的 × 默认周期（1d）
python scripts/update_data.py

# 指定标的与周期（美股 / A股 / 加密）
python scripts/update_data.py -s AAPL,MSFT,SPY -tf 1d
python scripts/update_data.py -s 000001.SZ -tf 1d --adjust qfq
python scripts/update_data.py -s BTC/USDT -tf 1h,4h

# 全量重刷（清空本地数据后重拉，复权方式改变时使用）
python scripts/update_data.py -s AAPL -tf 1d --full --adjust qfq

# 从指定日期补拉历史（backfill）
python scripts/update_data.py -s AAPL -tf 1d --start 2020-01-01

# 分钟级数据（免费源限制近期，可用 --start 限定窗口）
python scripts/update_data.py -s AAPL -tf 1m --start 2026-08-14

# 指定时间段（如 2024 上半年）
python scripts/update_data.py -s 000001.SZ -tf 1d --start 2024-01-01 --end 2024-06-30

# 下载全部历史（从最早可用日期起，数据源限制内）
python scripts/update_data.py -s 000001.SZ -tf 1d --start 2000-01-01
```

实测输出示例：

```
symbol        tf    source        rows  覆盖区间                                     状态
--------------------------------------------------------------------------------------------------------
AAPL          1d    yfinance       500  2024-08-16 13:30:00 → 2026-08-14 13:30:00 OK
BTC-USD       1d    yfinance       731  2024-08-16 00:00:00 → 2026-08-16 00:00:00 OK
完成: 2/2 成功
```

特点：
- **多数据源 fallback**：按资产类型配置优先级（`config/default.yaml` 的 `data.providers.priority`），
  每个数据源独立健康检查，失败自动切下一个；结果记录实际使用的数据源。
- **统一格式**：`datetime/open/high/low/close/volume/adj_factor`，UTC 存储；
  A股支持 qfq/hfq/none 复权（baostock，**分钟线/指数仅支持不复权**，自动按不复权处理并告警）；
  yfinance 自动计算复权因子；`market_data.adjust` 记录实际复权方式。
- **增量更新**：从上次最后 bar 增量拉取，按时间戳去重合并（时间戳保留微秒精度，避免分钟级边界错位）；
  分钟级数据按周期限制首拉范围（1m/5m≤30天，1h≤120天等）。
- **yfinance 韧性**：客户端被 Yahoo 限流（429）时自动回退原生 chart API（浏览器 UA + 分页）。
- **加密交易所可配**：`data.ccxt_exchanges: [okx, bitget, gateio, ...]`（按易访问性排序，可自行调整）。
- **文件安全**：parquet 文件名中 1M 编码为 1mo，避免 Windows 大小写不敏感文件系统与 1m 冲突。

实测多周期输出（已通过 1m/1h/1d/1w/1M 验证，增量无重复）：

```
symbol    tf    source    rows  覆盖区间                                      状态
AAPL      1d    yfinance   500  2024-08-16 13:30:00 → 2026-08-14 13:30:00  OK
AAPL      1m    yfinance   391  2026-08-14 13:30:00 → 2026-08-14 20:00:00  OK
AAPL      1w    yfinance   106  2024-08-12 04:00:00 → 2026-08-14 20:00:01  OK
AAPL      1M    yfinance    25  2024-09-01 04:00:00 → 2026-08-14 20:00:01  OK
BTC-USD   1h    yfinance  2882  2026-04-18 15:00:00 → 2026-08-16 15:30:23  OK
```

## 运行 API 服务（已启动于 http://127.0.0.1:8000）

```bash
python -m uvicorn quant.api.app:app --host 127.0.0.1 --port 8000
```

- 健康检查：http://127.0.0.1:8000/api/health
- 数据状态（含数据源健康）：http://127.0.0.1:8000/api/data/status
- 一键更新：`POST /api/data/update`，Swagger：http://127.0.0.1:8000/docs

## 运行测试

```bash
pytest -q        # 138 个用例（数据47/经纪商19/引擎11/指标7/模拟盘9/组合11/策略34；
                 # 含 baostock 真实拉取用例，需联网且已安装 baostock）
```

兼容性已验证：pandas 3.0.5 + numpy 2.5.2 + yfinance 1.6.0 + akshare 1.18.91 + ccxt 4.5.73 + pyarrow 25，
三个数据源 import/基础操作/真实拉取全程无警告（`tests/test_data.py` 的 provider 离线用例 + 实测）。
baostock（A股主源，0.9.3）已实测：股票（000001.SZ 日线 qfq 485 行两年窗口）、
ETF（510300.SH 日线 154 行 + 5 分钟 1104 行）、指数（000001.SH 上证指数 / 399001.SZ 深证成指 日/周线）
全周期真实拉取通过，增量更新无重复，健康检查 ~50-1100ms。

## 回测（第 3 步）

```bash
# 事件驱动（默认，完整撮合：费用/滑点/涨跌停/T+1/lot/保证金/风控）
python scripts/run_backtest.py -s AAPL -tf 1d -st dual_ma

# 向量化（快，执行近似，仅单标的）
python scripts/run_backtest.py -s AAPL -tf 1d -st dual_ma --engine vectorized

# 调参 / 组合
python scripts/run_backtest.py -s AAPL -tf 1d -st dual_ma --params '{"fast": 10, "slow": 30}'
python scripts/run_backtest.py --combo -s AAPL,SPY -st dual_ma,macd -w '{"dual_ma": 60, "macd": 40}'

# API（服务运行中）
POST /api/backtest/single   # 单策略回测
POST /api/backtest/combo    # 组合回测
GET  /api/backtest/runs     # 回测历史
```

实测示例（AAPL 1d，双均线，事件驱动，next_open）：

```
总收益率 8.39%  年化 4.15%  最大回撤 5.43%  夏普 0.7421  胜率 33.33%  盈亏比 3.101  交易 7 笔
 2024-12-02 buy  1264 @ 237.29  金叉 227.81>227.61
 2025-01-29 sell 1264 @ 234.10  pnl -4041
 2025-07-11 buy  1406 @ 210.59  金叉 203.70>203.64
 2026-01-13 sell 1406 @ 258.69  pnl +67633
```

向量化引擎同策略同数据结果几乎一致（总收益 8.41%，信号层逐点相同，执行层仅费用舍入差异）。

## 内置策略库（第 4 步，8 个，参数均有默认值）

| key | 策略 | 核心参数（默认值） |
|---|---|---|
| `dual_ma` | 双均线 | fast=20, slow=60, only_long=False |
| `macd` | MACD | fast=12, slow=26, signal=9, only_long=True |
| `rsi` | RSI 超买超卖 | period=14, oversold=30, overbought=70, only_long=True |
| `bollinger` | 布林带 | period=20, num_std=2.0, exit_at_mid=True, only_long=True |
| `momentum` | 动量 | lookback=60, hold_bars=10, threshold=0.0, only_long=True |
| `mean_reversion` | 均值回归 | lookback=20, entry_z=2.0, exit_z=0.0, only_long=True |
| `grid` | 网格交易 | grid_pct=0.02, levels=10, base_position=0.5, only_long=True |
| `turtle` | 海龟交易 | entry_n=20, exit_n=10, atr_period=14, atr_mult=2.0, only_long=False |

所有策略基于 `quant/core/indicators.py` 指标库（SMA/EMA/MACD/RSI/BOLL/ATR/Donchian/ROC/Z-Score），
该指标库同时是 LLM 声明式策略的白名单。调参直接改 `--params '{"key": {"fast": 10}}'` 或等 UI 参数表单。

```bash
# 组合回测（signal_merge 权重合成，含单策略独立指标）
python scripts/run_backtest.py -s AAPL -tf 1d -st dual_ma,macd,rsi \
    --weights '{"dual_ma": 50, "macd": 30, "rsi": 20}'
```

实测组合（AAPL 1d）：组合收益 8.79% / 回撤 4.93% / 夏普 0.786；
单策略对比：dual_ma 8.39%（夏普 0.74）、macd 2.89%（0.28）、rsi 8.71%（0.64）。

## 组合与权重管理（第 5 步）

**三种模式**（公式见 `docs/BACKTEST_ENGINE_DESIGN.md` §1.4）：
- `signal_merge`：`combined = Σ wᵢ·sᵢ·confᵢ / Σ|wᵢ·confᵢ|`，负权重反向，阈值 θ 过滤微调
- `equal`：等权特例
- `capital_split`：每策略独立子账户（资金按 |wᵢ| 分配，负权重反向），**回测与模拟盘均支持**

```bash
# API（服务运行中）
GET /api/strategies            # 策略列表（含启用/权重/参数 schema）
PUT /api/strategies/{id}       # 调参/权重/启停
POST /api/strategies/{id}/toggle
GET|PUT /api/portfolio         # 组合模式/权重/阈值/再平衡频率（持久化）

# 组合回测（不传 portfolio 时使用已保存的 /api/portfolio 配置）
POST /api/backtest/combo       # {symbols, timeframe, strategy_ids}
```

实测（AAPL 1d，dual_ma+macd 等权组合）：收益 8.09% / 回撤 4.93% / 夏普 0.74 / 33 笔。
模拟盘 capital_split：账户状态显示两个子账户资金池（如 60/40 分 60万/40万），持仓/成交按 `sub_key` 独立落库，净值合并。

## 模拟盘（第 3 步）

```bash
# CLI 前台常驻（或由 API 服务驱动，二者选一）
python scripts/run_paper.py start --symbols AAPL --timeframe 1d --strategy dual_ma
python scripts/run_paper.py status
python scripts/run_paper.py stop --account 1

# API
POST /api/paper/start    POST /api/paper/stop
GET  /api/paper/status   GET /api/paper/positions   GET /api/paper/trades   GET /api/paper/equity
```

- 调度：分钟级 interval 轮询（1m→65s、5m→315s…），日线按标的时区 cron（A股 15:30 / 美股 16:30 / crypto 00:05 UTC）
- 状态全落库，重启 `recover()` 恢复 running 账户并**丢弃 PENDING 挂单**（避免重复成交）

## 目录结构

```
quant/                # 核心包
  core/               # 模型 / 配置 / 工具（指标库在后续步骤）
  data/               # DataProvider 抽象 + baostock/yfinance/akshare/ccxt + Parquet 存储 + DataManager
  storage/            # SQLite 初始化 + 仓储层
  api/                # FastAPI（独立于 UI，后续 Streamlit 只消费 API）
config/default.yaml   # 默认配置（数据源优先级 / 默认标的 / 周期 / 交易所链）
scripts/update_data.py# 一键更新 CLI
tests/                # pytest（138 例）
docs/ARCHITECTURE.md  # 架构设计（含 10 项评审调整与实施路线图）
```

## Web UI（第 6 步，Streamlit 第一版）

```bash
start.bat    # 一键启动：FastAPI(8000) + Streamlit(8501)，首次自动装依赖
# 或手动：
python -m uvicorn quant.api.app:app --port 8000
python -m streamlit run streamlit_app.py --server.port 8501
```

访问 http://127.0.0.1:8501。七个页面：

| 页面 | 功能 |
|---|---|
| 数据管理 | 状态表（标的/周期/数据源/行数/**数据起始→结束**/最新bar/过期）、数据源健康（始终展示全部 + **实际通道** + **真实延迟分级着色** + **上次检查时间/已缓存 X 秒** + **重新检测按钮**）、**添加新标的**（自动推断类型 + 时间段选择）、**删除标的**、一键更新（含时间段）、**下载全部历史**、**持久化下载结果反馈**、周期说明、自动刷新 |
| 策略创建 | **手动条件规则策略表单**：指标白名单（11 个，中文全称+缩写）→ 买卖/做空/平空条件组（all/any、9 操作符）→ 仓位（固定比例/金额/ATR/信号强度）→ 实时 JSON 预览 → 保存草稿 / 保存并回测 → 激活 |
| 策略管理 | 8 内置策略 + 手动规则策略卡片：启停开关、schema 参数表单自动保存、**编辑跳转创建页**、**删除**（内置置灰/二次确认/组合引用级联清理） |
| 组合配置 | 三模式（信号加权/资金分配/等权）、权重滑杆 -100~100 步长 5%、信号阈值/再平衡、组合信号分解预览 |
| 交易设置 | TradingSettings 全字段表单（资金/费用/频度/仓位/做空保证金/风控/基准下拉预置），保存到 `/api/settings/trading` |
| 回测 | 多标的（支持手动输入新 symbol 自动拉取）+多策略+引擎 → 指标卡片（含**超额收益 vs 买入持有 / Beta / Alpha / 信息比率 / 基准最大回撤**）、**归一化收益对比图**（策略/买入持有/基准指数/标的股价）、**回撤对比图**（策略 vs 基准）、**月度收益热力图**、K线+买卖点+成交量、交易明细、单策略对比 |
| 模拟盘 | 账户状态卡片、start/stop、capital_split 子账户资金池、**删除账户**（stopped 才可删）、最新成交、自动刷新 |

UI 只消费 FastAPI（`web/streamlit/api.py` 薄客户端），后续可整体替换 React。

**一键关闭系统**：侧栏底部红色"关闭系统"按钮 → 二次确认 → `POST /api/system/shutdown`：
自动停止所有运行中模拟盘账户并落库 → 后端优雅退出（uvicorn lifespan shutdown，日志"Application shutdown complete"）
→ UI 显示"系统已关闭，可以安全关闭此窗口"并自行退出。手动分开启动 API/UI 时同样分别关闭各自进程。

**环境标识**：`/api/health` 返回 `environment`（local/sandbox，检测 DSH_* 环境变量），UI 侧栏显示"运行环境：本地运行/沙箱运行"彩色方框。

**模拟盘账户管理**：`DELETE /api/paper/accounts/{id}?cleanup=true` 删除账户——仅 stopped 状态可删（运行中返回 409），cleanup 控制是否同时清理持仓/成交/净值记录；UI 模拟盘页提供删除面板（运行中账户按钮置灰）。

**基准对比**：交易设置页基准改为下拉预置（SPY/QQQ/沪深300(000300.SS)/中证500(000905.SS)/BTC-USD）+ 自定义输入，标注"基准用于对比策略是否跑赢大盘"；回测时基准数据缺失自动拉取，并**独立传入引擎**（不与交易标耦合，不会把基准当交易标的）；未设置/无法加载基准时**默认用交易标的本体（买入持有）作为基准**。
回测结果可视化（判断策略是否真的有效）：
- **归一化收益对比图**（初始=100%）：策略收益 / 买入持有（等权持有全部交易标的）/ 基准指数 / 标的股价四条线同图，直观对比"策略 vs 无脑持有 vs 大盘"
- **回撤对比图**：策略回撤与基准回撤叠加
- **月度收益热力图**：策略月度收益年×月矩阵（红涨绿跌）
- **对比指标卡片**：超额收益(vs买入持有)、买入持有年化、基准最大回撤、Beta、Alpha(年化)、信息比率
K线图含成交量副图与清晰买卖点标记。

**手动策略创建（策略创建页）**：用白名单指标（SMA/EMA/MACD/RSI/BOLL/ATR/ROC/Z-Score/唐奇安/成交量均线/VWAP）编写条件规则策略——
买入/卖出/做空/平空条件组（内部 all/any 组合、9 种操作符含上穿/下穿/介于）、仓位设置（固定比例/固定金额/ATR 动态/信号强度映射）、允许做空/仅做多、最大持仓时间；
实时预览声明式 JSON → 保存为草稿 → 立即回测 → 激活启用。与未来 LLM 生成共用同一 schema 与校验链
（`quant/strategies/llm/`，`POST /api/strategies/rule`），已预留加载器接口（`registry.set_rule_loader`）。
保存的策略与内置策略一样出现在策略管理页（可调参/启停/参与组合/回测/模拟盘），可一键跳回创建页编辑。

**策略删除**：策略管理页每张卡片带"删除"按钮（内置策略置灰 + 提示"内置策略不可删除"；规则策略可删），
二次确认（st.warning + 确认/取消）；被组合引用的策略删除时提示"将从组合中移除"，确认后 **DELETE /api/strategies/{id}**
自动级联清理组合权重（内置 → 403）。

**标的添加/删除**：`POST /api/data/symbols` 自动按格式推断资产类型（含 `/` → crypto/ccxt、
6 位 A股代码 → baostock——51/56/58（沪）或 15/16（深）开头 → ETF；399xxx 或 000xxx（非 .SZ）→ 指数；
其他 → 股票；其余 → 股票/yfinance）；
`DELETE /api/data/symbols/{symbol}`（`{symbol:path}` 路由，支持 BTC/USDT 这类斜杠交易对）删除符号池 + 全部已存行情。

**UI 显示**：策略显示"中文全称 (English)"（如 双均线策略 (Dual Moving Average)），周期显示"日线 (Daily)"等中英文映射（1M 显示为 月线 (Monthly)，不会泄漏文件名中的 1mo）。

**数据安全**：`--full` 全量重刷采用**先拉取成功后再替换**策略——拉取失败时保留旧数据（曾修复"先删后拉、失败即丢数据"的缺陷）。

## 数据源可用性说明

- **沙箱限制**：本开发环境（DSH 沙箱）网络受限——baostock 官方服务器可能超时、交易所 API（ccxt）超时、Yahoo 对共享 IP 限流（已有浏览器 UA 兜底 + 分页）。**本地正常网络下均可用**。
- **A股（baostock 主源）**：官方客户端 TCP 直连证券宝服务器（https://www.baostock.com），免费开源，无东财反爬问题。覆盖范围（官方文档）：
  - **股票**：日/周/月线自 1990-12-19 至今；5/15/30/60 分钟线近 5 年
  - **ETF**：日/周/月线 + 5/15/30/60 分钟线，自 2026-01-05 至今
  - **指数**：日/周/月线自 2006-01-01 至今；**不提供分钟线**（分钟需求自动走 akshare/yfinance fallback）
- **加密货币（ccxt）**：默认链 okx→bitget→gateio→binance→bybit→kucoin；网络/代理受限时可在 `config/default.yaml` 调整 `ccxt_exchanges`。
- **美股（yfinance）**：客户端被 Yahoo 429 限流时自动回退原生 chart API（浏览器 UA + 分页）。
- **A股期货（akshare）**：baostock 不支持期货，期货仅由 akshare 提供。

**优先级**（`data.providers.priority`，失败/不健康自动切下一个）：
- `stock` / `etf` / `index`：`[baostock, akshare, yfinance]`——baostock 主源 → akshare 备用
  （akshare 内部还有东财→腾讯→新浪三通道）→ yfinance 兜底
- `future`：`[akshare, yfinance]`；`crypto`：`[ccxt]`
- 美股等非 A股代码会在 baostock/akshare 处快速校验失败后落到 yfinance；纯美股用户可改 `[yfinance]`。

**代理处理**：系统代理（如 Clash）会把东方财富等国内数据源拦到境外代理导致 akshare（A股备用/期货）失败。
默认 `data.akshare_use_proxy: false`——akshare 请求时把国内行情域名（eastmoney/sina/10jqka 等）写入 NO_PROXY 直连；
需要走代理时改为 `true`。同时**全局注入浏览器 User-Agent**（东方财富会拒绝 python-requests 默认 UA，导致 RemoteDisconnected/403）。
baostock 经官方客户端 TCP 直连服务器，不受 HTTP 系统代理影响，无需代理配置。

**A股（baostock）实现**：每次操作 login → query → logout（全程加锁串行化，避免全局会话冲突）；
复权 adjustflag 1=后复权 2=前复权 3=不复权；**分钟线不支持复权**（请求 qfq/hfq 时按不复权拉取并告警）；
**指数无复权概念**（同样按不复权处理并告警）；成交量单位为「股」（无需 ×100）。
查询字段按周期区分（实测 baostock 服务器严格校验）：日线含 `preclose`、**不含 `time`**；
周线/月线**不含 `preclose`**（传这两个字段分别报错 10004012）；分钟线含 `time` 不含 `preclose`。
指数分钟线直接快速失败并提示走 akshare/yfinance。健康检查在 5s 预算内探测 sz.000001 最近 5 天日线并标注实际数据源。

**akshare（A股备用源）多通道 fallback**：当 baostock 不可用或未覆盖（如期货、指数分钟线）时作为备用。
东方财富反爬会主动断开无 Referer 的请求（RemoteDisconnected），
akshare 数据源已实现三通道链（每通道最多 2 次尝试，指数退避 2s/4s/8s，连续失败 3 次暂停 30s，批量间隔 ≥1s）：
1. **东方财富** `push2his.eastmoney.com`（带 `Referer: quote.eastmoney.com` + 浏览器 UA，10s 超时）——主通道
2. **腾讯行情** `web.ifzq.gtimg.cn` fqkline（`Referer: gu.qq.com`）
3. **新浪/雪球** `ak.stock_zh_a_daily`（SH/SZ 前缀）
任一通道成功即返回并记录实际通道；健康检查在 5s 预算内依次快速探测并标注**实际使用的 A股通道**
（UI 数据管理页健康表"实际通道"列显示 东方财富/腾讯/新浪/雪球）。
实测：东方财富被断时自动切腾讯成功拉取 000001 日线 484 行。

**加密交易所**：默认链 `ccxt_exchanges: [okx, bitget, gateio, binance, bybit, kucoin]`（按易访问性排序，逐个尝试）；
单交易所超时 `ccxt_timeout_ms: 5000`。全部不可用时错误信息会提示检查网络/代理并给出调整建议；
也可自行调整（如 huobi/kraken）。

**端口占用**：start.bat 自动检测 UI 端口，8501 被占用时依次尝试 8502~8505 并打开对应地址；
后端 API 固定 8000（日志 api.log），失败不闪退并提示。

## 架构图

```
┌────────────────────────── Web UI (Streamlit :8501) ──────────────────────────┐
│  数据管理  策略创建  策略管理  组合配置  交易设置  回测  模拟盘   [关闭系统]      │
│        └────────────── web/streamlit/api.py 薄客户端（仅 HTTP）────────────┘  │
└────────────────────────────────────┬──────────────────────────────────────────┘
                                     │ REST (FastAPI :8000)
┌────────────────────────────────────▼──────────────────────────────────────────┐
│  quant/api (FastAPI)                                                          │
│   data │ backtest │ paper │ strategies(+rule) │ portfolio │ settings │ system │
├───────────────────────────────────────────────────────────────────────────────┤
│  引擎层：VectorizedBacktestEngine / EventDrivenBacktestEngine / PaperTrading   │
│  策略层：8 内置策略 + RuleStrategy（手动/LLM 声明式，registry 加载器）           │
│  组合层：StrategyPortfolio（signal_merge / capital_split / equal，共享 T1Ledger）│
│  数据层：DataManager → baostock/yfinance/akshare/ccxt（fallback+健康检查）→ Parquet      │
│  存储层：SQLite（symbols/market_data/strategies/portfolio/settings/paper/backtest）│
└───────────────────────────────────────────────────────────────────────────────┘
```

## 配置项全表（config/default.yaml，完整注释见 config/example.yaml）

| 配置 | 默认 | 说明 |
|---|---|---|
| `log_level` | INFO | 日志级别 |
| `api_host` / `api_port` | 127.0.0.1 / 8000 | API 监听地址与端口 |
| `data.timezone` | UTC | 行情统一存储时区 |
| `data.display_timezone` | null | UI 展示时区（null=本地） |
| `data.adjust` | qfq | A股复权：qfq/hfq/none |
| `data.lookback_days` | 730 | 首次拉取历史天数 |
| `data.incremental` | true | 默认增量更新 |
| `data.default_timeframes` | ["1d"] | 一键更新默认周期 |
| `data.akshare_use_proxy` | false | akshare（A股备用/期货）禁用系统代理（国内直连） |
| `data.ccxt_exchanges` | [okx, bitget, gateio, binance, bybit, kucoin] | 加密交易所 fallback 链 |
| `data.ccxt_timeout_ms` | 5000 | 单交易所请求超时 |
| `data.providers.priority` | stock/etf/index:[baostock,akshare,yfinance] 等 | 各资产类型数据源优先级 |
| `symbols` | AAPL/MSFT/SPY/000001.SZ/600519.SH/000300.SH/510300.SH/BTC-USDT/ETH-USDT | 初始标的池 |
| `llm.base_url` / `model` | api.deepseek.com / deepseek-chat | LLM 预留（未启用） |

## FAQ

**Q1: 本地有系统代理（Clash 等），akshare（A股备用/期货）报 ConnectionError/RemoteDisconnected？**
系统代理会拦截东方财富等国内域名。默认已启用 `akshare_use_proxy: false`（国内域名写入 NO_PROXY 直连）
并全局注入浏览器 UA（东方财富拒绝 python-requests 默认 UA）。仍失败时检查代理软件是否对 eastmoney.com 强制代理。
A股主源 baostock 为官方客户端 TCP 直连，不受 HTTP 系统代理影响。

**Q7: 回测页报 404 Not Found（`/api/backtest/combo`）？**
这是**后端没加载最新代码**的典型症状（通常是旧的后端进程仍占着 8000 端口，新进程起不来，
UI 连到了旧后端）。排查与修复：
1. 浏览器打开 http://127.0.0.1:8000/docs ，在 Swagger 页面搜索 `backtest`——
   - **没有** `/api/backtest/combo` → 当前 8000 上是旧代码/旧进程，执行第 2 步清理后重启
   - **有**该路径但 UI 仍 404 → 前端请求路径异常（检查浏览器是否连了别的端口/地址）
2. 杀掉旧后端：命令行执行 `netstat -ano | findstr :8000` 记下 PID，再 `taskkill /F /PID <PID>`；
   或在 UI 侧栏点"关闭系统"（`POST /api/system/shutdown` 优雅退出）
3. 重新运行 `start.bat`——新版启动器会**先自动清理 8000 上的旧 QuantSim 进程**（精准按命令行匹配 +
   netstat 兜底），就绪后校验 openapi 中 `/api/backtest/combo` 是否存在：存在打印 `Backend routes OK`，
   缺失/异常仅**警告并继续启动 UI**（不会再中途退出导致 UI 打不开）；可手动验证
   `http://127.0.0.1:8000/openapi.json` 里包含该路径
4. 注意：`start.bat` 启动的后端是**独立后台窗口**（QuantSim-API），日志写入 `api.log`（窗口本身空白是正常的）；
   关闭 Streamlit 主窗口**不会**停掉后端——需要重启后端时请用 UI"关闭系统"按钮或手动 kill 8000 进程后再启动。
5. 若启动后 UI 仍打不开：主窗口会打印到哪一步（`Backend API is ready.` / `Backend routes OK.` /
   `Starting Streamlit on ...`），把主窗口完整输出发回排查。

**Q2: 加密货币数据拉不到（ccxt 全部不可达）？**
默认链按易访问性排序（okx→bitget→gateio→binance→bybit→kucoin），逐个尝试、命中即缓存。
你的网络/代理若连这些都不通，在 `config/default.yaml` 调整 `ccxt_exchanges` 为实际可达的交易所，
或调整系统代理规则放行加密货币交易所域名。

**Q3: 8501 端口被占用？**
start.bat 自动检测，8501 被占用时依次尝试 8502~8505 并打开对应地址；全部占用会明确提示。

**Q4: 关闭系统后数据会丢吗？**
不会。所有配置（交易设置/组合/策略参数）均为即时写库；模拟盘账户状态与持仓/成交/净值随每笔操作落库；
关闭前还会自动停止运行中的模拟盘账户并做最终净值快照。

**Q5: 手动创建的策略与内置策略有何区别？**
功能等价：都可启停/调参/参与组合/回测/模拟盘。区别仅在于——内置策略是代码实现、不可删除；
手动策略是声明式配置（指标白名单+条件+仓位）、可删除/编辑，且与未来 LLM 生成共用同一校验链。

**Q6: 环境标识显示"沙箱运行"？**
`/api/health.environment` 检测 `DSH_*` 环境变量（沙箱注入）。本地正常运行时无该变量 → 显示"本地运行"。

## 后续步骤（已预留）

- LLM 策略生成：`quant/strategies/llm/`（schema+校验链）已就绪，`POST /api/strategies/rule` 接受 LLM 生成的同构配置；
  未来只需实现生成器（DeepSeek 调用 + 提示词）并配置 `llm.api_key`，零改动接入。当前策略创建页为手动表单。
