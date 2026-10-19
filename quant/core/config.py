"""应用配置。

优先级：代码默认值 < config/default.yaml < config/user.yaml < 环境变量 / .env。
所有可调参数均带默认值，空配置可直接运行。

**可分发设计**：任何路径都从 `ROOT`（= 本文件向上三级，即软件目录）推导，
不写死某台机器的绝对路径。数据目录可通过 `config/user.yaml` 的 `data_dir`
或环境变量 `DATA_DIR` 改到别处（UI 里也能改：系统设置 → 存储与路径）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent.parent

# 用户本机覆盖层：与 default.yaml 同结构，**深度合并**在上层。
# 该文件不进版本库（见 .gitignore），用于放「这台机器专属」的参数：
# 数据目录、监听端口、代理开关、LLM 端点等。升级重装不会覆盖它。
USER_CONFIG = ROOT / "config" / "user.yaml"

# 把 .env 读入 os.environ（幂等）
load_dotenv(ROOT / ".env")


class ProviderPriority(BaseModel):
    """各资产类型的数据源优先级（靠前优先，前一个失败才换下一个）。

    国内 A股/ETF/指数：
      1. baostock —— 主渠道，官方 TCP 直连
      2. akshare  —— 备选渠道，内部**三通道**：东方财富 → 腾讯 → 新浪
                     （前一条失败自动换下一条；腾讯是唯一支持 1m 的通道）。

    具体哪条通道能通取决于当时的网络环境与数据源状态，不是固定结论，
    所以这里只描述顺序，不对任何通道的可达性下断言。
    国外：
      yfinance —— 美股/ETF 主渠道（Yahoo）

    > 腾讯原为独立的第三级 provider，现并入 akshare 内部：健康检查按
    > 「baostock / akshare / yfinance」三项呈现，akshare 再展开逐通道明细，
    > 不会把同一家腾讯在健康表里显示两遍，代理开关也只需一处。
    > ``TencentProvider`` 类仍保留在 ``quant/data/tencent_provider.py``。

    可通过配置自行调整顺序。
    """

    stock: list[str] = ["baostock", "akshare"]
    etf: list[str] = ["baostock", "akshare"]
    index: list[str] = ["baostock", "akshare"]  # 指数（baostock 无分钟线，自动切下一级）
    us_stock: list[str] = ["yfinance"]  # 美股/海外 ETF（Yahoo）


class DataConfig(BaseModel):
    timezone: str = "UTC"  # 行情统一存储时区
    display_timezone: str | None = None  # UI 展示时区，None = 本地时区
    adjust: str = "qfq"  # A股复权：qfq 前复权 | hfq 后复权 | none 不复权
    lookback_days: int = 730  # 首次拉取历史天数
    incremental: bool = True  # 默认增量更新
    default_timeframes: list[str] = ["1d"]
    akshare_use_proxy: bool = False  # akshare 国内数据源直连（不走系统代理）
    providers: ProviderPriority = ProviderPriority()


class LLMConfig(BaseModel):
    """大模型配置（DeepSeek 等，OpenAI 兼容 / Anthropic / Gemini 接口）。

    api_protocol 决定客户端如何拼端点、发请求、解析响应：
      - openai    OpenAI 兼容 Chat Completions（DeepSeek/Qwen/Moonshot/GLM/OpenAI）
      - anthropic Anthropic Messages API（Claude）
      - gemini    Google Gemini generateContent
    """

    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    model: str = "deepseek-chat"  # deepseek-chat(V3) | deepseek-reasoner(R1)
    api_protocol: str = "openai"  # openai | anthropic | gemini
    timeout_seconds: float = 60.0
    max_retries: int = 3
    temperature: float = 0.2


class KronosConfig(BaseModel):
    """Kronos K 线基础模型配置（可选组件，缺依赖时自动降级）。

    模型权重在首次使用时从 Hugging Face 下载（NeoQuasar/Kronos-*）。
    device="auto" → 有 CUDA 用 CUDA，否则 CPU（CPU 推理 400 根上下文约 0.7s）。
    """

    enabled: bool = True
    model: str = "Kronos-small"  # Kronos-mini(4.1M) | Kronos-small(24.7M) | Kronos-base(102.3M)
    tokenizer: str = ""  # 空 → 按 model 自动配对
    device: str = "auto"  # auto | cpu | cuda:0
    max_context: int = 512
    pred_len: int = 12  # 默认预测未来多少根 K 线
    temperature: float = 1.0
    top_p: float = 0.9
    sample_count: int = 1  # 采样路径数（多条取平均，越大越稳但越慢）
    tz: str = "Asia/Shanghai"  # 时间特征时区（模型吃「几点/周几」这类特征）


class MLConfig(BaseModel):
    """机器学习辅助模型（当前仅 Kronos）。"""

    kronos: KronosConfig = Field(default_factory=KronosConfig)


class AppConfig(BaseModel):
    """聚合配置。可被 config/default.yaml 覆盖；LLM_*/DATA_DIR 等环境变量优先。"""

    data_dir: Path = ROOT / "data"
    db_path: Path | None = None  # None → data_dir/quant.db
    market_dir: Path | None = None  # None → data_dir/market
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    data: DataConfig = DataConfig()
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    symbols: list[dict[str, Any]] = [
        {"symbol": "AAPL", "name": "Apple", "asset_type": "stock", "source": "yfinance"},
        {"symbol": "MSFT", "name": "Microsoft", "asset_type": "stock", "source": "yfinance"},
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "asset_type": "etf", "source": "yfinance"},
        {"symbol": "000001.SZ", "name": "平安银行", "asset_type": "stock", "source": "baostock"},
        {"symbol": "600519.SH", "name": "贵州茅台", "asset_type": "stock", "source": "baostock"},
    ]

    # ---- 派生路径 ----
    @property
    def effective_db_path(self) -> Path:
        return self.db_path or self.data_dir / "quant.db"

    @property
    def effective_market_dir(self) -> Path:
        return self.market_dir or self.data_dir / "market"

    @classmethod
    def load(cls, yaml_path: str | Path | None = None) -> "AppConfig":
        """加载配置：代码默认值 < config/default.yaml < config/user.yaml < 环境变量。

        ``user.yaml`` 是**本机覆盖层**（不进版本库），深度合并到 default.yaml 之上，
        用来放这台机器专属的参数（数据目录、端口、代理、LLM 端点等），
        这样升级/重装不会把用户设置冲掉。传了显式 ``yaml_path`` 时只用该文件，
        不叠加 user.yaml（便于测试与一次性覆盖）。
        """
        path = Path(yaml_path) if yaml_path else ROOT / "config" / "default.yaml"
        raw: dict[str, Any] = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        if yaml_path is None and USER_CONFIG.exists():
            user_raw = yaml.safe_load(USER_CONFIG.read_text(encoding="utf-8")) or {}
            raw = _deep_merge(raw, user_raw)

        cfg = cls(**raw)

        # 环境变量覆盖（LLM 密钥等）
        env_map: dict[str, tuple[str, ...]] = {
            "LLM_API_KEY": ("llm", "api_key"),
            "LLM_BASE_URL": ("llm", "base_url"),
            "LLM_MODEL": ("llm", "model"),
            "LLM_PROTOCOL": ("llm", "api_protocol"),
            "LLM_TIMEOUT": ("llm", "timeout_seconds"),
            "DATA_DIR": ("data_dir",),
            "API_PORT": ("api_port",),
        }
        for env_name, fields in env_map.items():
            val = os.getenv(env_name)
            if val is None:
                continue
            node: Any = cfg
            for f in fields[:-1]:
                node = getattr(node, f)
            try:
                setattr(node, fields[-1], type(getattr(node, fields[-1]))(val))
            except (TypeError, ValueError):
                setattr(node, fields[-1], val)
        return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两层 dict（override 优先）；列表整体替换而非逐项合并。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
