# QuantSim — 本地量化交易模拟系统

本地运行的量化交易模拟系统：历史回测（向量化 / 事件驱动双模式）、实时模拟交易（paper trading）、
一键更新数据、11 个内置策略（8 因子化 + 3 模型化，参数均有默认值）、
**AI 策略工坊（自然语言对话生成 / 反复修改策略）**、手动创建条件规则策略（支持 AI 辅助）、
**金融时序大模型预测与多模型加权集成**、**大模型 × 因子策略自由融合**、策略组合与权重管理。

全部数据与账本落在本地（Parquet + SQLite），后端 FastAPI + 前端 Streamlit，一条 `start.bat` 起全套。

完整架构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，引擎设计见 [docs/BACKTEST_ENGINE_DESIGN.md](docs/BACKTEST_ENGINE_DESIGN.md)。

## 功能特性

| 分类 | 能力 |
|---|---|
| 行情数据 | baostock / akshare（内部东财→腾讯→新浪）/ yfinance 三级容错，8 个周期，增量更新 |
| 回测 | 事件驱动（费用/滑点/涨跌停/T+1/lot/保证金/风控）与向量化双引擎 |
| 策略 | 8 个经典因子策略 + Kronos 预测 + 多模型集成 + 模型因子融合 + 声明式规则策略 + AI 代码策略 |
| 组合 | 信号加权 / 资金分配 / 等权三种模式，权重 -100~100 |
| 模拟盘 | 分钟级按 K 线边界调度的准实时模拟，状态全落库可恢复 |
| AI | 自然语言生成策略代码或规则配置，生成后可直接手改代码 |
| 分析 | 超额收益、Beta、Alpha、信息比率、归一化对比、回撤对比、月度热力图、K 线买卖点 |

## 环境要求

- Python 3.11+
- 网络（拉取免费行情源：baostock / akshare / yfinance）

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
pip install -r requirements-optional.txt   # 可选：baostock（A股股票/指数/ETF 主源）+ akshare（A股备选）
pip install -r requirements-ml.txt         # 可选：时序大模型（torch 等，约 200MB+）
```

> 若 akshare 安装报 jsonpath 构建失败：
> `pip install setuptools && pip install jsonpath --no-build-isolation` 后重试。

## 启动

```bash
start.bat    # 一键启动：FastAPI(8000) + Streamlit(8501)，首次自动建环境并装依赖
```

访问 http://127.0.0.1:8501 。也可分开手动启动：

```bash
python -m uvicorn quant.api.app:app --host 127.0.0.1 --port 8000
python -m streamlit run streamlit_app.py --server.port 8501
```

- 健康检查：http://127.0.0.1:8000/api/health
- 数据状态（含数据源健康）：http://127.0.0.1:8000/api/data/status
- Swagger：http://127.0.0.1:8000/docs

`start.bat` 会自动检测 UI 端口，8501 被占用时依次尝试 8502~8505 并打开对应地址；
后端 API 固定 8000（日志写 `api.log`），失败会打印原因而不闪退。

## 分发给其他用户（便携迁移）

本系统的所有路径都从**软件所在目录**推导（`quant/core/config.py` 的 `ROOT`，
不依赖当前工作目录、不写死任何一台机器的绝对路径），所以**整个目录可直接拷贝到别的机器运行**。

### 迁移步骤

1. **先清理本机专属内容**（复制前做，避免把自己的数据/密钥带给别人）：

   | 清理项 | 原因 |
   |---|---|
   | `config/user.yaml` | 本机覆盖层（数据目录、代理、LLM 密钥），已在 `.gitignore` 中 |
   | `.env` | LLM API Key |
   | `data/` | 自己的行情库与账本（可选保留示例数据） |
   | `api.log` / `streamlit.log` / `.tmp/` | 运行日志与临时文件 |

   若目标机**没装 Python**，则**不要**删 `.venv/`（它自带解释器，拷过去可直接用）；
   若目标机会自己建环境，删掉 `.venv/` 可省 ~1GB。

2. **拷贝整个目录**到目标机器（任意盘符、含空格和中文的路径都支持）。
3. 目标机运行 `start.bat`：首次会自动建 `.venv` 并装依赖（已带 `.venv` 则直接启动）。
4. 打开 UI → **系统设置 → 数据设置**，确认「存储与路径」里的数据目录指向正确位置；
   需要换盘/移动硬盘时在这里改（只接受绝对路径），保存后**重启后端**生效。

### 目标机首次配置

- **数据目录**：默认 `软件目录/data/`。要改到别处有两种方式：
  - UI：**系统设置 → 数据设置 → 存储与路径 → 数据目录**（推荐，会做写入测试）；
  - 文件：复制 `config/user.example.yaml` 为 `config/user.yaml`，填 `data_dir: D:\你的路径`。
- **数据源代理**：**系统设置 → 数据设置 → 数据源代理**，逐渠道选 `跟随系统/强制直连/强制走代理`。
  国内源选直连、`yfinance` 跟随系统（或走代理），保存即生效。
- **LLM 密钥**：在 `.env` 里填 `LLM_API_KEY=...`，或 UI 的 **系统设置 → 策略设置 → 大模型设置**。

### 环境变量（可选，优先级最高）

| 变量 | 作用 |
|---|---|
| `DATA_DIR` | 覆盖数据目录（绝对路径） |
| `API_PORT` | 覆盖后端端口 |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` / `LLM_PROTOCOL` | 大模型端点与密钥 |
| `QUANTSIM_PYTHON` | 指定启动用的 Python 解释器 |
| `HF_ENDPOINT` | 模型权重下载镜像（如 `https://hf-mirror.com`） |

## 一键更新数据

```bash
# 更新全部启用标的 × 默认周期（1d）
python scripts/update_data.py

# 指定标的与周期（美股 / A股）
python scripts/update_data.py -s AAPL,MSFT,SPY -tf 1d
python scripts/update_data.py -s 000001.SZ -tf 1d --adjust qfq
python scripts/update_data.py -s 600519.SH -tf 1h

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

工作机制：

- **两级降级链**：A股走 `baostock（主）→ akshare（备选）`，美股走 `yfinance`；
  **前一个拿不到、或该周期它根本不支持，才换下一个**（不并行、不全量拉）。
  第二级的 akshare 内部再降级 **东财 → 腾讯 → 新浪**（顺序随周期动态变化）。
  每个数据源通过 `supports(symbol, timeframe)` 声明能力，不支持的周期直接跳过，不发无效请求。
- **统一格式**：`datetime/open/high/low/close/volume/adj_factor`，UTC 存储；
  A股支持 qfq/hfq/none 复权（baostock，**分钟线/指数仅支持不复权**，自动按不复权处理并告警）；
  yfinance 自动计算复权因子；`market_data.adjust` 记录实际复权方式。
- **增量更新**：从上次最后 bar 增量拉取，按时间戳去重合并（时间戳保留微秒精度，避免分钟级边界错位）。
- **数据安全**：`--full` 全量重刷采用**先拉取成功后再替换**策略——拉取失败时保留旧数据。
- **yfinance 韧性**：客户端被 Yahoo 限流（429）时自动回退原生 chart API（浏览器 UA + 分页）。
- **文件安全**：parquet 文件名中 1M 编码为 1mo，避免 Windows 大小写不敏感文件系统与 1m 冲突。

## 支持的周期

| 周期 | 标签 | 默认首拉 | 说明 |
|---|---|---|---|
| `1m` | 1 分钟 | 近 8 天 | 最短周期，历史深度最浅，通常只够看最近一两个交易日 |
| `5m` | 5 分钟 | 近 1 年 | 分钟线，历史深度取决于数据源当下的限制 |
| `15m` | 15 分钟 | 近 1 年 | 同上 |
| `30m` | 30 分钟 | 近 1 年 | 同上 |
| `1h` | 1 小时 | 近 2 年 | 同上 |
| `1d` | 日线 | 近 2 年 | 多数数据源可回溯到上市首日 |
| `1w` | 周线 | 近 10 年 | 同上 |
| `1M` | 月线 | 近 20 年 | 最长周期，通常能覆盖全部历史 |

> 已删除 2 小时 / 4 小时两个周期（各源均无此粒度）与加密货币/期货支持。
> 想要更长历史时显式传 `--start`，会按各源能力裁剪到最早可得的日期；
> 实际拿到的区间会体现在下载结果里。

## 运行测试

```bash
pytest -q
```

> 含 baostock 真实拉取用例，需联网且已安装 baostock。

## 回测

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

两个引擎在同一策略同一数据上结果几乎一致（信号层逐点相同，执行层仅费用舍入差异）。

## 内置策略库

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
| `kronos_forecast` | **Kronos 预测**（模型化） | context=240, pred_len=12, rebalance_every=12, threshold_sigma=0.5, min_move=0.0, position=1.0, only_long=True |
| `ensemble_forecast` | **多模型集成**（模型化） | weights_json（默认 Kronos+EWMA+动量）、min_return_sigma=0.3, min_agreement=0.6, max_dispersion=2.0, min_models=2, context=240, pred_len=12 |
| `fusion` | **模型因子融合**（模型化 × 因子化） | model_weights_json、factor_weights_json、fusion_mode=blend、model_share=0.5、confidence_weighted=True、context=240、pred_len=12 |

前 8 个是**因子化**策略，只用历史数据算规则；后 3 个是**模型化**策略，要先拿到预测再映射成仓位。
其中 `fusion` 同时调用多模型集成与因子策略，把两条信号缝成一个仓位。
它们都注册在同一个 `quant.strategies.registry` 里，回测/模拟盘/组合用法完全一致。

所有策略基于 `quant/core/indicators.py` 指标库（SMA/EMA/MACD/RSI/BOLL/ATR/Donchian/ROC/Z-Score），
该指标库同时是 LLM 声明式策略的白名单。调参直接改 `--params '{"key": {"fast": 10}}'` 或等 UI 参数表单。

## 组合与权重管理

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

`capital_split` 下账户状态会显示多个子账户资金池，持仓/成交按 `sub_key` 独立落库，净值合并。

## 模拟盘

```bash
# CLI 前台常驻（或由 API 服务驱动，二者选一）
python scripts/run_paper.py start --symbols AAPL --timeframe 1d --strategy dual_ma
python scripts/run_paper.py status
python scripts/run_paper.py stop --account 1

# API
POST /api/paper/start    POST /api/paper/stop
GET  /api/paper/status   GET /api/paper/positions   GET /api/paper/trades   GET /api/paper/equity
```

- 调度：**分钟级按 K 线边界 cron 对齐**（1m→每小时第 20 秒、5m→每 5 分的第 30 秒、15m/30m 同构、
  1h→整点第 10 秒），日线按标的时区 cron（A股 15:30 / 美股 16:30）。
- **交易时段门控**：分钟级账户只在交易时段拉数（A股 9:30–11:30、13:00–15:00，美股 9:30–16:00，
  周末与午休自动跳过）。非交易时段的 tick 被直接跳过并记录原因，不空转网络请求；
  日线/周线/月线**不门控**（cron 本就排在收盘后）。
- **启动不重放历史**：启动时设水位线 = 倒数第 2 根 K 线，首次 tick 只补算最近 1 根。
  水位线逐 tick 落库，`recover()` 据此续跑，既不复读历史也不丢停机期间的增量。
- 状态全落库，重启 `recover()` 恢复 running 账户并**丢弃 PENDING 挂单**（避免重复成交）。
- **实时语义（重要）**：`fetch_realtime()` 未实现，各渠道均无 tick/盘口接口；
  撮合以**已收盘 K 线的收盘价**成交（`execution=next_open` 则下一根开盘价），
  滑点是固定 bps、不随下单量放大，且不按成交量限制成交比例。
  因此这是「按 K 线粒度推进的准实时模拟」——验证策略逻辑与资金曲线够用，
  **不适合评估执行质量（冲击成本、成交概率、排队）**。
- 模拟盘页的「最新K线延迟」= 本地最后一根 K 线距现在的秒数，这是延迟的可观测口径。

## Web UI

侧边栏按**四个模块**组织（外加系统设置）：

| 模块 | 页面 |
|---|---|
| 数据管理 | 数据管理 |
| 策略管理 | AI 策略、模型预测、策略创建、策略管理、策略融合、组合配置 |
| 策略回测 | 策略回测 |
| 交易模拟 | 模拟盘 |
| 系统设置 | 数据设置、交易设置、策略设置 |

| 页面 | 功能 |
|---|---|
| 数据管理 | 状态表（标的/周期/数据源/行数/数据起始→结束/最新bar/过期）、数据源健康（始终展示全部 + **实际通道** + **真实延迟分级着色** + 上次检查时间 + 重新检测按钮）、**添加新标的**（自动推断类型 + 时间段选择）、**删除标的**、一键更新（含时间段）、**下载全部历史**、持久化下载结果反馈、周期说明、自动刷新 |
| AI 策略 | **AI 策略工坊**：三标签（AI 对话 / 代码编辑 / 我的 AI 策略）+ 三意图（✨新策略 / 🔧改这版 / 💡要建议）→ 多轮对话（"再改改""加个止损"）→ AI 直接产出 Python 代码或规则 JSON → 自动静态检查 + 冒烟回测 → 右侧「🧾 当前版本」实时预览（版本号/校验状态/冒烟指标/完整代码）→ **内置代码编辑器可手动改**（校验/保存/让 AI 改这版）→ 「保存到策略库」一键入库 → 快速回测 |
| 模型预测 | **金融时序大模型**：Kronos 模型状态（模型/设备/参数量/加载耗时）、首次「加载模型」按钮、预测表单（标的/周期/上下文/预测长度/采样次数）→ 预测幅度指标 + 历史收盘 + 预测收盘 + 高低区间曲线 → 预测明细表 → 「去回测这个策略」一键跳转 |
| 策略创建 | **手动条件规则策略表单** + **AI 辅助**：指标白名单（11 个）→ 买卖/做空/平空条件组（all/any、9 操作符）→ 仓位（固定比例/金额/ATR/信号强度）→ 实时 JSON 预览 → 保存草稿 / 保存并回测 → 激活 |
| 策略管理 | 全部策略卡片：启停开关、schema 参数表单自动保存、**编辑跳转创建页/改代码**、**删除**（内置置灰/二次确认/组合引用级联清理） |
| 策略融合 | **大模型 × 因子自由融合**：① 启用哪些大模型 + 权重（按 深度学习/统计基线/通用时序 分组，未装依赖的置灰）+ 5 道门控；② 启用哪些因子策略 + 权重；③ 融合方式 + 模型侧份额 + 置信度加权；保存后写全局配置，**「模型因子融合」策略在回测/模拟盘直接生效**；下方可对真实行情**试算**（不写库） |
| 组合配置 | 三模式（信号加权/资金分配/等权）、权重滑杆 -100~100 步长 5%、信号阈值/再平衡、组合信号分解预览 |
| 数据设置 | **数据源代理**（逐渠道 `跟随系统 / 强制直连 / 强制走代理`，存 `kv_store` 立即生效）+ **存储与路径**（软件根目录 / Python 解释器 / 数据目录 / 数据库 / 行情缓存 5 行路径表，含存在性与占用；可改数据目录，写入 `config/user.yaml`，提示需重启后端且不自动迁移历史数据） |
| 交易设置 | TradingSettings 全字段表单（资金/费用/频度/仓位/做空保证金/风控/基准下拉预置），保存到 `/api/settings/trading` |
| 策略设置 | **大模型设置**（API Key/接口地址/模型/重试/超时/温度 + 测试连接 + 探测可用模型）+ **WorkBuddy 连接**（状态 / 模型列表 / 自定义 exe 路径 / 测试连接 / 保存并校准） |
| 策略回测 | 多标的 + 多策略 + 引擎 → 指标卡片（含**超额收益 vs 买入持有 / Beta / Alpha / 信息比率 / 基准最大回撤**）、**归一化收益对比图**、**回撤对比图**、**月度收益热力图**、K线+买卖点+成交量、交易明细、单策略对比 |
| 模拟盘 | 账户状态卡片、start/stop、capital_split 子账户资金池、**删除账户**（stopped 才可删）、最新成交、自动刷新；**分钟级实时面板**（市场/时段/最新K线延迟/最近tick/本tick，非交易时段自动暂停拉取） |

UI 只消费 FastAPI（`web/streamlit/api.py` 薄客户端），后续可整体替换 React。

**一键关闭系统**：侧栏底部红色"关闭系统"按钮 → 二次确认 → `POST /api/system/shutdown`：
自动停止所有运行中模拟盘账户并落库 → 后端优雅退出 → UI 显示"系统已关闭"并自行退出。

**模拟盘账户管理**：`DELETE /api/paper/accounts/{id}?cleanup=true` 删除账户——仅 stopped 状态可删
（运行中返回 409），cleanup 控制是否同时清理持仓/成交/净值记录。

**基准对比**：交易设置页基准为下拉预置（SPY/QQQ/沪深300(000300.SS)/中证500(000905.SS)/600519.SH）+ 自定义输入；
回测时基准数据缺失自动拉取，并**独立传入引擎**（不与交易标耦合）；
未设置/无法加载基准时**默认用交易标的本体（买入持有）作为基准**。

## AI 策略生成

**配置**：**系统设置 → 策略设置 → 大模型设置**，填 API Key（DeepSeek 默认 `https://api.deepseek.com` + `deepseek-chat`），
点「测试连接」验证；也可用环境变量 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`（UI 留空时不覆盖 `.env`）。
任何 OpenAI 兼容接口都可用（Moonshot / 智谱 / 通义 / OpenAI 等，改 Base URL 与模型名即可）。

**用法**：左侧「AI 策略」页 → 选模式 → 用自然语言描述想法
（"20 日均线上穿 60 日均线买入，跌破 20 日线卖出，单笔最多半仓"）。

- **代码策略模式**：AI 直接给出完整可运行的 `class Strategy(BaseStrategy)` Python 代码，
  带 `Params` 参数类（每个字段有类型、默认值、取值范围）。生成后自动：
  1. **静态检查**（`check_source`）：AST 扫描导入白名单（pandas/numpy/math/指标库…）、
     禁用 `eval/exec/open/__import__` 等、禁止访问 `__globals__` 等逃逸属性、必须实现 `generate_signals`；
  2. **冒烟回测**（`smoke_test`）：用合成 K 线真跑一遍，校验返回必须是带
     `target_weight/reason/confidence` 三列的 DataFrame、行数与行情一致、权重是数值。

  两项都过才能保存——**手改代码走完全相同的校验链**，改坏了存不进去。
- **声明式规则模式**：AI 输出 JSON 配置（指标白名单+条件组+仓位），经 `validate_config` 校验后入库，
  之后仍可在「策略创建」页用表单继续调参。

**多轮对话**：会话持久化在 `llm_sessions` / `llm_messages` 表；后续消息自动带上会话里最近一次生成/修改的代码
作为上下文，所以直接说"加个 ATR 止损""改成只在放量时买入"即可增量修改，不用重复描述。
历史会话可切换、可删除。

**手动改代码**：页内「代码编辑」视图是一个源码编辑器——

- 载入方式：AI 生成后点「载入代码编辑器」、或从「我的 AI 策略」/「策略管理 → 改代码」载入已保存策略；
- 「校验代码」随时跑静态检查+冒烟回测并显示结果（含信号数量、最大/最小仓位、最后一根K线的原因）；
- 「保存为策略」直接入库；已存在的策略用 `PUT /api/llm/code/{id}` 覆盖源码，**内置策略禁止改源码（403）**；
- 「让 AI 再改一版」把当前代码回填到对话里继续迭代——人工手改与 AI 生成可自由交替。

**安全边界**：代码在受限命名空间执行（`_safe_builtins` 白名单 + 仅暴露 pd/np/math/indicators/BaseStrategy），
导入与属性访问受 AST 白名单约束。这是防误操作的**软约束**，不是对抗恶意代码的沙箱——
只应运行你自己或可信 AI 生成的策略。

**关键 API**：

```
GET  /api/llm/status              # 是否已配置 + 模型信息
POST /api/llm/ping                # 连通性测试
POST /api/llm/chat                # 多轮对话生成/修改（返回内容 + 校验结果）
POST /api/llm/generate-strategy   # 单轮生成（无状态）
POST /api/llm/validate            # 校验代码或规则配置
POST /api/llm/save                # 保存为策略（代码策略自动挂载）
GET  /api/llm/code/{id}           # 读取策略源码（手改前）
PUT  /api/llm/code/{id}           # 覆盖策略源码（手改后保存）
GET  /api/llm/sessions            # 历史会话
```

生成的代码策略以 `strategy_type="code"` 存入 `strategies.source_code` 列，
由 `registry._default_rule_loader` 按 key 动态编译加载——因此**和内置策略完全同权**：
可回测、可加入组合、可跑模拟盘。每次生成/修改都写 `llm_prompts` 审计表。

**策略删除**：策略管理页每张卡片带"删除"按钮（内置策略置灰 + 提示"内置策略不可删除"；规则策略可删），
二次确认；被组合引用的策略删除时提示"将从组合中移除"，确认后 **DELETE /api/strategies/{id}**
自动级联清理组合权重（内置 → 403）。

**标的添加/删除**：`POST /api/data/symbols` 自动按格式推断资产类型
（6 位 A股代码 → 51/56/58（沪）或 15/16（深）开头 → ETF；399xxx 或 000xxx（非 .SZ）→ 指数；其他 → 股票；
其余 → 股票/yfinance，例如 `AAPL`）；`DELETE /api/data/symbols/{symbol}` 删除符号池 + 全部已存行情。

**UI 显示**：策略显示"中文全称 (English)"，周期显示"日线 (Daily)"等中英文映射
（1M 显示为 月线 (Monthly)，不会泄漏文件名中的 1mo）。

## AI 策略工坊

把「生成一次」升级成「聊到满意再保存」的工坊：左侧对话，右侧实时查看当前版本，全程不用离开页面。

**三意图（intent，与模式 code/rule 正交）**：

| intent | 界面 | 作用 |
|---|---|---|
| `write` | ✨ 新策略 | 从零写策略；改需求重写也走这里 |
| `revise` | 🔧 改这版 | 把**上一版完整代码/配置**连同修改要求一起发给 LLM，让它整体重写而不是给片段 |
| `ideas` | 💡 要建议 | LLM 以「策略研究顾问」身份只发散思路，返回 JSON 想法数组（title/logic/scene/risk），每条带「按这个思路写策略」一键转成 write |

**代码策略**（mode=code）：LLM 输出 ` ```python ` 代码块，后端 `extract_code` 抽出后走
`check_source`（AST + import 白名单）与 `smoke_test`（合成K线冒烟回测），
保存为 `strategy_type="code"` 的 CodeStrategy；**用户可在「代码编辑」标签里直接手改代码再保存**
（`PUT /api/llm/code/{id}`，同样过校验链）。

**规则策略**（mode=rule）：LLM 输出 ` ```json ` 配置（RuleStrategyConfig 同构 schema），
过 `validate_config` 后保存为 `strategy_type="llm_generated"`。

**会话与版本**：`llm_sessions` 存会话（title/mode/strategy_id），`llm_messages` 存每轮消息
（role/content/code/meta_json）；「改这版」以上一版代码为上下文——请求没显式带 code 时，
后端自动沿用会话里最近一次生成的代码，规则模式同理沿用最近一次 config。
前端「历史对话」下拉可切换/新建会话，刷新页面后也能从会话消息还原「当前版本」。

**保存入口**：ai_page 的「保存到策略库」区域（一键保存 / 保存选项表单）→ `POST /api/llm/save`，
body 含 `mode=code|rule` + `code` 或 `config` + `name`/`key`（可选 `enabled`/`params`/`session_id`/`strategy_id`，
后者用于原地更新已有策略）。

**降级与容错**：未配置 API Key 时页面提示去「系统设置 → 策略设置 → 大模型设置」（或 `.env` 的 `LLM_API_KEY`），
并给「前往大模型设置」按钮；LLM 调用失败时错误详情写回会话消息；
输出解析失败（`extract_code`/`extract_json` 抛错）时把错误与**模型原文**回显在对话里，让用户手动处理。

## WorkBuddy 模型接入

在不额外申请 API Key 的情况下，复用已登录的 WorkBuddy 桌面 App 登录态，把 WorkBuddy 的模型与积分接到量化系统里，
作为**可选的模型来源**：模型来源下拉里出现「WorkBuddy」分组，选中后用 WorkBuddy 的模型生成策略；
不选仍走 DeepSeek/OpenAI 兼容配置。

- **凭据**：读取 `%LOCALAPPDATA%\CodeBuddyExtension\Data\Public\auth\workbuddy-desktop.info`；
  WorkBuddy 5.6+ 把 accessToken/refreshToken 用 AES-256-GCM 加密成 `{$wbEncrypted:1, envelope}` 信封，
  密钥来自 WorkBuddy 自己的 Electron 二进制。
- **上游**（非官方开放 API）：chat `POST https://copilot.tencent.com/v2/chat/completions`（SSE 流式）、
  模型目录 `GET .../console/enterprises/personal/models`、积分 `POST https://www.codebuddy.cn/v2/billing/meter/get-user-resource`。
  请求须带 `Authorization: Bearer` + `X-User-Id` + `X-Domain` + `Origin/Referer/User-Agent/X-Requested-With`，缺了会 403。
- **接入点**：**系统设置 → 策略设置** 的「WorkBuddy 连接」卡片（账号/区域/积分/模型列表 + 刷新）；
  AI 策略工坊「模型来源」下拉可选 WorkBuddy，再选具体模型（默认 auto）。
  后端 `quant/llm/workbuddy.py`（凭据解密 + 客户端）+ `quant/api/routers/workbuddy.py`。
- **依赖**：`cryptography>=42`（AES-GCM 解密）。
- 注意：WorkBuddy `auto` 模型会路由到推理模型，返回空 content——`ping` 和测试要用具体轻量模型（如 `glm-5.3-flash`）。

## 策略创建页的 AI 辅助

策略创建页顶部有一个可折叠的「AI 辅助」面板，让 AI 参与**手动策略**的编写，而不是另起一套流程：

- 用中文描述想法（面板内置 3 个示例，点一下即填入），点「生成配置」→ 后端走 `/api/llm/chat`（mode=rule）返回声明式 JSON；
- 勾选「把当前表单的条件一起发给 AI」时，请求里会附上当前表单的完整配置，并明确要求在此基础上修改，
  保留没提到的部分——这就是「在已有策略上继续优化」而不是每次从零重写；
- 结果先过 `validate_config`（和白名单/操作符/仓位模式完全同一套校验），通过后点「载入到下方表单」
  把条件**灌进表单控件**，之后照常用下拉框/滑杆逐项手调；
- 生成结果与手填结果走同一个 `_build_config`，最终都落在同一个 schema 上，不存在两份配置。

## 金融时序模型与多模型集成

除了规则型策略，系统还能调用**金融时序基础模型**预测未来价格路径，并把多个模型的预测**加权融合成仓位**。
全部代码在 `alpha/` 下（与 `quant/` 解耦）：

```
alpha/factors/   因子化：只用历史数据算的规则型策略（8 个内置经典策略）
alpha/models/    模型化：需要加载权重或统计外推的预测模型
```

### 可用模型

`GET /api/models` 会实时探测每个模型的可用状态（**不触发加载**）：

| key | 类型 | 说明 | 依赖 |
|---|---|---|---|
| `kronos` | dl | Kronos 金融 K 线基础模型（AAAI 2026，arXiv:2508.02739），小型 24.7M 参数，512 根上下文 | torch/einops/safetensors/huggingface_hub/tqdm |
| `random_walk` | stat | 零假设锚点：预测 = 当前价格（任何模型跑不赢它就没有价值） | 无 |
| `drift` | stat | 长期对数漂移外推 | 无 |
| `momentum` | stat | 近 20 根动量外推（逐根衰减） | 无 |
| `reversion` | stat | 向 20 日线均值回归 | 无 |
| `ewma` | stat | 指数加权收益外推 | 无 |
| `chronos` | external | Amazon Chronos（T5 系，取 20 条采样路径中位数） | `pip install chronos-forecasting` |
| `timesfm` | external | Google TimesFM | `pip install timesfm` |
| `moirai` | external | Salesforce Moirai（uni2ts） | `pip install uni2ts` |

装模型权重：`pip install -r requirements-ml.txt`（torch CPU 轮子约 200MB+；有 GPU 时可自行换 CUDA 版本）。
国内网络可设 `HF_ENDPOINT=https://hf-mirror.com`。**没装 torch 的用户照样能用全部因子化策略与统计基线**——
`alpha/models` 之外的代码不会 import torch，`import alpha.models` 在缺依赖时也不会失败（懒加载 + find_spec 探测）。

### 加权集成 + 硬性门控

`ensemble_forecast` 策略把多个模型的预期收益按权重投票：

```
combined   = Σ(wᵢ · retᵢ) / Σwᵢ
agreement  = 与 combined 同向的权重占比
dispersion = 各模型预测的加权标准差
```

然后过 **5 道硬性门控**，任一道不过就空仓（并把实测值写进 `reason`，便于排查为什么没开仓）：

| 门控 | 默认 | 含义 |
|---|---|---|
| `min_models` | 2 | 至少要有几个模型跑出了预测 |
| `min_return_sigma` | 0.3 | 预测幅度至少 0.3 倍近 60 根收益标准差（**按波动率归一，跨周期通用**） |
| `min_abs_return` | 0.0 | 绝对幅度下限（防极低波动期噪声开仓） |
| `min_agreement` | 0.6 | 同向权重占比（模型意见必须基本一致） |
| `max_dispersion` | 2.0 | 分歧上限（模型互相矛盾时不开仓） |

再加一条方向门控：`only_long` 时看空直接空仓。

> **为什么按波动率归一**：绝对百分比阈值跨周期不成立——5 分钟线 0.1% 已是大波动，日线要 0.5% 才算。
> 所以用 `threshold_sigma × σ` 而不是写死一个绝对值。

**建议权重**（`POST /api/ml/ensemble/auto-weights`）会滚动回看若干评估点，算每个模型的
**方向命中率**与**预测-实现 IC**，归一化成建议权重。它只用于给一个初始猜测（同一样本上评估、不是严格样本外），
负相关或命中率低于 50% 的模型直接给 0 分而**不是用负权重反向**——负权重在样本外极不稳定。

### 关键 API

```
GET  /api/models                    # 所有预测模型（可用性/依赖/类型）
GET  /api/ml/kronos/status          # Kronos 状态（不触发加载）
POST /api/ml/kronos/load            # 加载权重（首次会下载）
POST /api/ml/kronos/predict         # 单次预测（返回历史+预测路径，供画图）
GET  /api/ml/ensemble/config        # 集成配置（权重+门控）
PUT  /api/ml/ensemble/config        # 保存集成配置（存 kv_store，负权重截断为 0）
POST /api/ml/ensemble/predict       # 跑一次集成：逐模型投票 + 逐门控实测值 + 目标仓位
POST /api/ml/ensemble/auto-weights  # 基于历史命中率/IC 给建议权重（不写库）
```

> 时序模型预测的是**价格路径**，不是收益承诺。它在本系统里的定位是另一个信号来源，
> 和因子化策略一样必须经过回测/模拟盘验证；模型输出未经校准，跨标的迁移效果可能差异很大。

## 大模型 × 因子策略融合

「模型化」和「因子化」是两条独立的策略线，若只用组合在**策略外部**按权重合并，粒度较粗
（信号加权/资金分配/等权），也没法表达「模型只当开关」这类逻辑。融合策略把两条线在**策略内部**缝起来。

### 策略：模型因子融合（`FusionStrategy`，key = `fusion`）

每根 K 线内部同时算两侧信号，再按用户选的方式合成一个仓位：

```
模型侧 signal_m ∈ [-1, 1]   = EnsembleForecaster 的预期收益映射（含 5 道门控）
因子侧 signal_f ∈ [-1, 1]   = 8 个因子策略信号的加权平均
最终 target_weight          = combine(signal_m, signal_f, 融合方式, 份额)
```

### 三种自由度（策略融合页）

| 维度 | 可选 | 说明 |
|---|---|---|
| 启用哪些模型 | Kronos / 5 个统计基线 / Chronos·TimesFM·Moirai | 权重 0 = 不启用；未装依赖的模型在 UI 上置灰 |
| 启用哪些因子 | dual_ma / macd / rsi / bollinger / momentum / mean_reversion / turtle / grid | 同上，权重 0 = 不启用 |
| 怎么结合 | 加权混合 / 模型闸门 / 因子闸门 / 投票制 | 另加**模型侧份额**（0~1）与**按置信度加权模型侧** |

四种融合方式的语义：

| 方式 | 语义 |
|---|---|
| 加权混合 | 两侧按份额加权平均，同向加强、反向抵消 |
| 模型闸门 | 模型不开门（空仓/门控未过）→ 因子一律作废；开门后按**模型方向**满仓 |
| 因子闸门 | 因子空仓 → 模型说了也不算；有方向后按**因子方向**满仓 |
| 投票制 | 两侧各自压成 多/空/观望 三档，按份额投票，平手空仓 |

> 具体收益取决于策略、标的与区间，请以实际回测为准。

### 降级与容错

- **单侧全 0 权重** → 自动退化成另一侧策略（`reason` 会写明「未启用任何大模型/因子」）；
- **K 线不足预热长度** → 全空仓，`reason` 写明预热进度，绝不在 NaN 上算指标；
- **模型侧报错**（缺依赖/推理失败）→ 该次决策记 model_signal=0 并降级为只用因子，
  不让整条策略崩掉（模拟盘里一次崩会连带整个账户）；
- **权重 JSON 写错/含 NaN·inf·负数** → 逐个丢弃坏项，不整份拒绝。

### 配置存哪

融合配置是**全局偏好**：存 `kv_store`（`ml.fusion` 键），回测页/模拟盘页/策略管理页看到同一套，
不必给 `strategies` 表加列。`PUT /api/fusion/config` 合并保存；
`POST /api/fusion/preview` 可带一份临时 `config` 试算而不写库（UI 调参数即时预览用）。

策略参数里也能覆盖（回测时想临时试另一套权重就改参数）：参数值**偏离字段默认值**才算
「显式设置」，否则用全局配置——否则「用户没动策略参数」会被误判成「用户显式要默认值」，
配置页调好的设置就被悄悄盖掉了。

### 关键 API

```
GET  /api/fusion/config        # 融合配置 + 可选模型/因子清单 + 融合方式说明
PUT  /api/fusion/config        # 合并保存（权重只留非负有限值；未知 key 丢弃）
POST /api/fusion/preview       # 用真实行情试算一次（config 可临时覆盖、不写库）
GET  /api/fusion/options       # 可选清单 + 融合方式说明
```

> 试算与回测走的是策略内部**同一个** `_decide()`，所以预览结果和回测最后一根的决策必然一致——
> 这是「UI 不骗人」的底线。

## 数据源可用性说明

**两级降级链**（`data.providers`，失败/不支持自动切下一个）：

- `stock` / `etf` / `index`：`[baostock, akshare]` ← **第二级 akshare 内部再降级：东财 → 腾讯 → 新浪**
- `us_stock`：`[yfinance]`
- 不并行、不全量拉；前一个源拿不到、或该周期它根本不支持，才换下一个。
- `TencentProvider` 类仍保留（可单独使用/测试），但**不在默认链里**——腾讯已并入 akshare 内部，
  避免同一个数据源在健康表上出现两遍、代理开关要设两处。
- **能力声明**：每个 Provider 提供 `supports(symbol, timeframe)` 与 `max_lookback_days(symbol, timeframe)`，
  `DataManager` 在调度前先问能力，不支持的周期直接跳过（不发无效请求、不计入失败），
  并把请求起点裁剪到该源实际可回溯的窗口内——避免必然失败的 `HTTP 422` / 空结果。

### 各渠道能力

| 渠道 | 个股 | ETF | 指数 | 1m | 5/15/30/60m | 1d/1w/1M |
|---|---|---|---|---|---|---|
| baostock | ✓ | ✓ | ✓ | ✗ | ✓（历史最深） | ✓ |
| akshare→东财 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| akshare→腾讯 | ✓ | ✓ | ✓ | ✓（唯一支持） | ✓ | ✓ |
| akshare→新浪 | ✓ | `fund_etf_hist_sina` | `stock_zh_index_daily` | ✗ | ✓ | ✓ |
| yfinance | 美股 ✓ | 美股 ETF ✓ | — | ✓ | ✓ | ✓ |

- **A股（baostock 主源）**：官方客户端 TCP 直连证券宝服务器，免费开源。
  - **股票**：日/周/月线自上市起；5/15/30/60 分钟线可回溯多年
  - **ETF**：日/周/月线 + 5/15/30/60 分钟线
  - **指数**：日/周/月线；**不提供分钟线**（`supports()` 返回 False，钩子交给下一渠道）
  - **不支持 1 分钟**（最小粒度 5 分钟）
  - **会话保持**：`login()` 单次要十几秒而一次查询只要几秒，所以进程内保持一条登录会话
    （首次用到登录一次，之后全部复用）；服务器中途掉线时自动 `logout→login` 重试；socket 超时 30s。
    app 启动时后台预热一次登录（`DataManager.warm_up()`），健康检查因此很快返回。
  - 复权 `adjustflag` 1=后复权 2=前复权 3=不复权；**分钟线不支持复权**（请求 qfq/hfq 时按不复权拉取并告警）；
    **指数无复权概念**。成交量单位为「股」（无需 ×100）。
  - 查询字段按周期区分（服务器严格校验）：日线含 `preclose`、**不含 `time`**；
    周/月线**不含 `preclose`**；分钟线含 `time` 不含 `preclose`。
- **A股（akshare 备选）**：**三条腿**——东方财富 → 腾讯 → 新浪（同 provider 内自动降级）。
  - 东财走 `push2his.eastmoney.com`（带 `Referer` + 浏览器 UA）；
  - 降级腾讯（复用 `tencent_provider` 的解析原语，不重复实现）；
  - 再降级新浪 `stock_zh_a_daily` / `stock_zh_a_minute` / `fund_etf_hist_sina` / `stock_zh_index_daily`
    （**日线只认 `qfq`/`hfq`，`adjust="none"` 返回空表**，代码里统一按 qfq 请求）。
  - **通道顺序随周期动态变化**（这是「数据广度」的直接体现，不能一刀切）：
    - `1m` → `[腾讯]`（东财/新浪都没有 1m）
    - `5m/15m/30m/1h` → `[东方财富, 新浪, 腾讯]`（新浪分钟线深度大于腾讯，**优先拿更深的历史**）
    - `1d/1w/1M` → `[东方财富, 腾讯, 新浪]`（深度相当但腾讯更快）
  - **通道熔断**：某条腿出现传输层失败（连接被断、超时、proxy 错）会立即熔断 5 分钟并直接换下一条，
    不再原地指数退避白等；限流（429）与 5xx 才原地退避 2s/4s/8s。
    熔断状态显示在健康表的「通道明细」列。用户点「重新检测」会真探一次，成功后自动解除熔断。
  - **区间裁剪**：腾讯端点没有区间参数、总返回最近 640 根，必须裁剪到请求窗口并显式对齐时区
    （通道索引是 tz-naive 本地时间，而调度传入的边界常是 tz-aware UTC）。
  - **健康检查逐腿独立计时**：并发探测三条腿、各自计时，`latency_ms` 只报**主用通道自己**的耗时，
    不会把「前一条腿失败等满超时」的等待累计算到后一条头上。
  - 非 A股代码会立即抛 `ValueError` 快速失败，不拖慢 fallback 链。
- **A股（腾讯，作为 akshare 内部通道）**：日/周/月走 `web.ifzq.gtimg.cn/appstock/app/fqkline/get`
  （640 根/页，按结束日期回退分页）；分钟线走 **`ifzq.gtimg.cn/appstock/app/kline/mkline`**
  （**不能带 `web.` 前缀**——`web.ifzq.gtimg.cn` 对 mkline 会 301 重定向到不可达的 `web3.ifzq.gtimg.cn`）。
  返回字段序是 `date, open, close, high, low, volume`（**open/close 在前、high/low 在后，按 OHLC 解析会读反**），
  成交量单位为「手」，需 ×100 换算成股。
- **美股（yfinance）**：客户端被 Yahoo 429 限流时自动回退原生 chart API（浏览器 UA + 分页）。

### 代理与超时

**系统代理**（如 Clash / v2rayN）可能把国内域名拦到境外代理导致 `ProxyError`/`RemoteDisconnected`。
现在可以**逐渠道独立设置**：`auto`（跟随系统）/ `direct`（强制直连）/ `proxy`（强制走代理），
在 UI 的 **系统设置 → 数据设置 → 数据源代理** 里改，**立即生效、无需重启**（存 `kv_store`）。
默认值：baostock / akshare / tencent = `direct`，yfinance = `auto`。

> 判断「直连」开关是否真的生效：`yfinance` 直连返回 403、走代理 200。
> akshare 内部的腾讯腿沿用 **akshare 这一档**代理设置（对用户来说它们是同一个数据源）。

`data.akshare_use_proxy` 仅为兼容旧配置的开关，**当前实现不改写代理环境变量**。
请求统一带浏览器 UA（东方财富会拒绝 python-requests 默认 UA）。
baostock 经官方客户端 TCP 直连，不受 HTTP 系统代理影响；但官方 `socketutil` **不设超时**，
本实现在建 socket 后注入了读写超时（默认 30s，健康检查探测用 6s），避免服务器无响应时 `login()` 无限阻塞。

## 架构图

```
┌────────────────────────── Web UI (Streamlit :8501) ──────────────────────────┐
│ 侧栏模块：数据管理 │ 策略管理(AI策略 模型预测 策略创建 策略管理 策略融合 组合配置)│
│                    │ 策略回测 │ 交易模拟 │ 系统设置(数据/交易/策略设置) [关闭]  │
│        └────────────── web/streamlit/api.py 薄客户端（仅 HTTP）────────────┘  │
└────────────────────────────────────┬──────────────────────────────────────────┘
                                     │ REST (FastAPI :8000)
┌────────────────────────────────────▼──────────────────────────────────────────┐
│  quant/api (FastAPI)                                                          │
│  data │ backtest │ paper │ strategies(+rule) │ llm │ ml │ portfolio │ settings │ system │
├───────────────────────────────────────────────────────────────────────────────┤
│  引擎层：VectorizedBacktestEngine / EventDrivenBacktestEngine / PaperTrading   │
│  策略层：alpha.factors（8 因子化）+ alpha.models.strategies（2 模型化）        │
│          + RuleStrategy（声明式）+ CodeStrategy（AI 生成/手改代码）            │
│  模型层：alpha.models（Kronos / 5 统计基线 / Chronos·TimesFM·Moirai 适配器     │
│          / EnsembleForecaster 加权集成 + 5 条硬性门控）                        │
│  组合层：StrategyPortfolio（signal_merge / capital_split / equal，共享 T1Ledger）│
│  数据层：DataManager → baostock→akshare(东财·腾讯·新浪)（A股）/ yfinance（美股）→ Parquet │
│  LLM 层：quant/llm（OpenAI 兼容 httpx 客户端 + 提示词 + 校验链）               │
│  存储层：SQLite（symbols/market_data/strategies+source_code/portfolio/settings/│
│          paper/backtest/llm_sessions/llm_messages/llm_prompts/kv_store）       │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
quant/                # 基础设施：数据、存储、引擎、接口
  core/               # 模型 / 配置 / 工具 / 指标库
  data/               # DataProvider 抽象 + baostock/akshare(东财·腾讯·新浪)/tencent/yfinance
                      #   + Parquet 存储 + DataManager + 逐渠道代理
  storage/            # SQLite 初始化 + 仓储层（含 LLMRepo / KVRepo）
  api/                # FastAPI（独立于 UI，Streamlit 只消费 API）
  engine/             # 向量化 / 事件驱动回测引擎 + 模拟盘 + 撮合 + 指标
  portfolio/          # StrategyPortfolio（信号加权/资金分配/等权）
  strategies/         # BaseStrategy + 注册表 + RuleStrategy + CodeStrategy + llm(schema/校验)
  llm/                # 大模型客户端（OpenAI 兼容 httpx）+ 提示词 + 结果解析 + WorkBuddy
alpha/                # 信号来源：因子化 + 模型化（与 quant/ 解耦，可独立替换）
  factors/            # 因子化：只用历史数据算的规则型指标/策略
    strategies/       # 8 个内置经典策略（双均线/MACD/RSI/布林/动量/均值回归/网格/海龟）
  models/             # 模型化：需要加载权重或统计外推的预测模型
    base.py           #   BaseForecaster / prepare_bars / return_sigma / future_index
    baselines.py      #   零依赖统计基线（随机游走/漂移/动量/均值回归/EWMA）
    kronos.py         #   Kronos 金融 K 线基础模型封装
    external.py       #   Chronos / TimesFM / Moirai 适配器（按需安装）
    ensemble.py       #   多模型加权集成 + 5 条硬性门控
    _kronos/          #   上游 Kronos 源码内联（MIT，见 NOTICE.md）
    strategies/       #   kronos_forecast / ensemble_forecast / fusion
config/default.yaml   # 默认配置（数据源链 / 默认标的 / 周期 / llm / ml.kronos）
config/user.example.yaml  # 本机覆盖层模板（复制成 user.yaml 后生效）
config/user.yaml      # 本机覆盖层（已在 .gitignore，不进版本库）
scripts/              # CLI：update_data / run_backtest / run_paper
tests/                # pytest
web/streamlit/        # Streamlit UI（api.py 薄客户端 + pages/ 页面）
docs/                 # ARCHITECTURE.md / BACKTEST_ENGINE_DESIGN.md
start.bat             # 一键启动（FastAPI + Streamlit）
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
| `data.akshare_use_proxy` | false | 兼容开关；当前实现不改写代理环境变量 |
| `data.providers` | stock/etf/index: [baostock, akshare]；us_stock: [yfinance] | 各资产类型数据源渠道链 |
| `symbols` | AAPL/MSFT/SPY/000001.SZ/600519.SH/000300.SH/510300.SH | 初始标的池 |
| `llm.base_url` / `model` | api.deepseek.com / deepseek-chat | AI 策略生成（也可在 UI 策略设置页配置） |
| `llm.api_key` | ""（或 `LLM_API_KEY`） | 大模型 API Key；为空时 AI 页提示去配置 |
| `llm.temperature` / `max_retries` / `timeout_seconds` | 0.2 / 3 / 60 | 生成随机度 / 失败重试次数 / 单次超时 |
| `ml.kronos.enabled` | true | 关掉后 Kronos 相关调用直接报"已在配置中关闭" |
| `ml.kronos.model` / `tokenizer` | Kronos-small / 自动配对 | 可选 Kronos-mini(4.1M) / small(24.7M) / base(102.3M) |
| `ml.kronos.device` | auto | auto / cpu / cuda:0（auto 依次探测 cuda → mps → cpu） |
| `ml.kronos.max_context` / `pred_len` | 512 / 12 | 喂给模型的上下文根数 / 预测未来几根 |
| `ml.kronos.temperature` / `top_p` / `sample_count` | 1.0 / 0.9 / 1 | 采样参数；sample_count>1 时对多条路径取均值 |
| `ml.kronos.tz` | Asia/Shanghai | 送入模型前把 UTC 索引转成本地时区 |

**本机覆盖层 `config/user.yaml`**（不进版本库，随软件分发时不应带上）：

| 可以覆盖 | 说明 |
|---|---|
| `data_dir` | 数据目录绝对路径；留空/删除 = 用软件目录下的 `data/` |
| `api_host` / `api_port` | 后端监听地址与端口 |
| `providers.*` | 逐渠道代理模式（`auto`/`direct`/`proxy`） |
| `llm.*` | LLM 端点、模型、密钥 |
| `log_level` | 日志级别 |

优先级：**代码默认值 < `config/default.yaml` < `config/user.yaml` < 环境变量 / `.env`**。
模板见 `config/user.example.yaml`（复制成 `user.yaml` 再改）。
也可以在 UI 里改：**系统设置 → 数据设置 → 存储与路径 / 数据源代理**。

## FAQ

**Q1: 本地有系统代理（Clash / v2rayN 等），akshare 报 ConnectionError / RemoteDisconnected？**

先分清两种失败：

1. **代理配置问题（可修）**：某些代理会把国内域名也送进境外节点。到
   **系统设置 → 数据设置 → 数据源代理**，把 `baostock` / `akshare` 设为 **强制直连**、
   `yfinance` 设为 **跟随系统或强制走代理**，保存即生效。
   判断代理是否真的在起作用：如果 `yfinance` 直连返回 403、走代理 200，说明直连开关正常。
2. **数据源侧限制（应用层不可修，靠兜底）**：TCP 能握手、TLS 证书正常，但 HTTP 请求被对端直接断开。
   这类限制与出口 IP、请求频率、网络环境都有关，**换一台机器/换一条网络结论可能完全不同**，
   所以界面只显示「可用 / 暂不可用」，不把任何一次观测写成固定说明。
   此时 akshare 会**自动降级到腾讯腿**，腾讯也挂再降级**新浪腿**，无需手工干预；三条腿全挂才会整体失败。
   首次失败后会**熔断 5 分钟**，后续下载直接走下一腿，不再白等；
   健康表的「通道明细」列会显示实时状态。
   如果想让 akshare 更快失败，可在 `config/user.yaml` 里把 A股链改成 `[baostock]`
   （跳过 akshare 也就失去 1m 与新浪的分钟线）。

**Q2: 某个周期一直下不下来 / 提示"所有数据源均失败"？**

先看「数据管理」页的**下载渠道能力表**（每个源支持哪些周期 ✓/—）。
下载前会先问各渠道 `supports(symbol, timeframe)`：**不支持的周期直接跳过**，不会空转。常见情形：

- **指数 + 分钟线**：baostock 明确不提供（表里是 —），会自动跳到 akshare（内部 腾讯/新浪 腿）。
- **1 分钟**：baostock 与新浪都没有，只有 akshare→腾讯和 Yahoo。
- **请求了很早的起点**：各源会按 `max_lookback_days` 把起点裁剪到可得范围，
  结果里会标注「已从 X 起」而不是报错。
- 若某个源整条链都不通，可在 `data.providers` 里去掉它，或调整系统代理规则。

**Q3: 8501 端口被占用？**

`start.bat` 自动检测，8501 被占用时依次尝试 8502~8505 并打开对应地址；全部占用会明确提示。

**Q4: 关闭系统后数据会丢吗？**

不会。所有配置（交易设置/组合/策略参数）均为即时写库；模拟盘账户状态与持仓/成交/净值随每笔操作落库；
关闭前还会自动停止运行中的模拟盘账户并做最终净值快照。

**Q5: 手动创建的策略与内置策略有何区别？**

功能等价：都可启停/调参/参与组合/回测/模拟盘。区别仅在于——
内置策略是代码实现、不可删除；手动策略是声明式配置（指标白名单+条件+仓位）、可删除/编辑；
AI 生成的代码策略可改源码。

**Q6: 环境标识显示"沙箱运行"？**

`/api/health.environment` 检测 `DSH_*` 环境变量（沙箱注入）。本地正常运行时无该变量 → 显示"本地运行"。

**Q7: 回测页报 404 Not Found（`/api/backtest/combo`）？**

这是**后端没加载最新代码**的典型症状（通常是旧的后端进程仍占着 8000 端口，新进程起不来，
UI 连到了旧后端）。排查与修复：

1. 浏览器打开 http://127.0.0.1:8000/docs ，在 Swagger 页面搜索 `backtest`——
   没有 `/api/backtest/combo` → 当前 8000 上是旧代码/旧进程，执行第 2 步清理后重启。
2. 杀掉旧后端：`netstat -ano | findstr :8000` 记下 PID，再 `taskkill /F /PID <PID>`；
   或在 UI 侧栏点"关闭系统"（`POST /api/system/shutdown` 优雅退出）。
3. 重新运行 `start.bat`——启动器会**先自动清理 8000 上的旧 QuantSim 进程**，
   就绪后校验 openapi 中 `/api/backtest/combo` 是否存在：存在打印 `Backend routes OK`，
   缺失/异常仅**警告并继续启动 UI**。
4. 注意：`start.bat` 启动的后端是**独立后台窗口**（QuantSim-API），日志写入 `api.log`
   （窗口本身空白是正常的）；关闭 Streamlit 主窗口**不会**停掉后端。

**Q8: AI 页提示"未配置大模型"？**

去「策略设置」页「大模型设置」填 API Key 并保存，点「测试连接」确认。
也可在 `.env` 里写 `LLM_API_KEY=sk-xxx`（UI 里的 Key 输入框留空保存不会覆盖 `.env`）。
DeepSeek 默认地址 `https://api.deepseek.com`，模型 `deepseek-chat`；
用其他 OpenAI 兼容服务时同时改 Base URL 和模型名。

**Q9: AI 生成的策略报"未通过冒烟回测"？**

说明生成的 `generate_signals` 没有满足返回契约（必须返回行数与行情一致、含
`target_weight/reason/confidence` 三列的 DataFrame，权重在 0~1 之间）。
直接点「让 AI 再改一版」把错误贴回去让它修，或者在「代码编辑」视图里手动改。
保存时同一套校验会再跑一次，改不好就存不进去——这是有意为之，避免坏策略污染回测。

**Q10: 「模型预测」页提示依赖未安装 / 加载很慢？**

`pip install -r requirements-ml.txt`（torch CPU 轮子约 200MB+）。
国内网络加 `set HF_ENDPOINT=https://hf-mirror.com`。首次加载要下权重，之后走本地缓存。
不装这些依赖也不影响其它功能——统计基线（random_walk/drift/momentum/reversion/ewma）是零依赖的。

**Q11: 集成策略一直空仓，怎么看原因？**

`POST /api/ml/ensemble/predict` 会在 `reason` 里逐条列出没通过的门控及其实测值，比只看仓位数字有用。
常见原因：模型意见分歧太大（`max_dispersion`）、预测幅度不足波动率的 0.3 倍（`min_return_sigma`）、
或集成整体看空但策略只做多（`direction`）。想让策略更积极就降低对应门控，但**先看回测**。

**Q12: 策略创建页点了「载入到下方表单」控件没变？**

Streamlit 控件带 `key` 时只在首次出现时读 `value=`，之后只认会话状态；前端控件本身也记住值，
单纯删 `session_state` 的键会被写回。实现上给控件 key 加了版本号（`rc_ver`）让整批控件重建。
旧版本上遇到时重启 Streamlit 即可。

## 许可

见 [LICENSE](LICENSE)。`alpha/models/_kronos/` 为上游 Kronos 源码内联（MIT），
其许可与出处见该目录下的 `NOTICE.md`。
