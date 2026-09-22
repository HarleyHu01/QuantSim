"""应用配置。

优先级：代码默认值 < config/default.yaml < .env（LLM 密钥等环境变量）。
所有可调参数均带默认值，空配置可直接运行。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent.parent

# 把 .env 读入 os.environ（幂等）
load_dotenv(ROOT / ".env")


class ProviderPriority(BaseModel):
    """各资产类型的数据源优先级（fallback 链，靠前优先）。

    - A股股票/指数/ETF：baostock 主源 → akshare 备用（内部东财→腾讯→新浪三通道）→ yfinance 兜底
    - 期货：akshare（baostock 不支持期货）
    - 美股：yfinance；加密货币：ccxt
    """

    stock: list[str] = ["baostock", "akshare", "yfinance"]
    etf: list[str] = ["baostock", "akshare", "yfinance"]
    index: list[str] = ["baostock", "akshare", "yfinance"]  # 指数（无分钟线；分钟需求走 akshare/yfinance）
    future: list[str] = ["akshare", "yfinance"]
    crypto: list[str] = ["ccxt"]


class DataConfig(BaseModel):
    timezone: str = "UTC"  # 行情统一存储时区
    display_timezone: str | None = None  # UI 展示时区，None = 本地时区
    adjust: str = "qfq"  # A股复权：qfq 前复权 | hfq 后复权 | none 不复权
    lookback_days: int = 730  # 首次拉取历史天数
    incremental: bool = True  # 默认增量更新
    default_timeframes: list[str] = ["1d"]
    akshare_use_proxy: bool = False  # akshare（A股ETF/期货）请求禁用系统代理（国内数据源直连）
    ccxt_exchanges: list[str] = ["okx", "bitget", "gateio", "binance", "bybit", "kucoin"]
    ccxt_timeout_ms: int = 5_000  # 单交易所请求超时（健康检查总预算见 DataManager）
    providers: ProviderPriority = ProviderPriority()


class LLMConfig(BaseModel):
    """大模型配置（DeepSeek，OpenAI 兼容接口）。"""

    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    model: str = "deepseek-chat"  # deepseek-chat(V3) | deepseek-reasoner(R1)
    timeout_seconds: float = 60.0
    max_retries: int = 3
    temperature: float = 0.2


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

    symbols: list[dict[str, Any]] = [
        {"symbol": "AAPL", "name": "Apple", "asset_type": "stock", "source": "yfinance"},
        {"symbol": "MSFT", "name": "Microsoft", "asset_type": "stock", "source": "yfinance"},
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "asset_type": "etf", "source": "yfinance"},
        {"symbol": "000001.SZ", "name": "平安银行", "asset_type": "stock", "source": "baostock"},
        {"symbol": "600519.SH", "name": "贵州茅台", "asset_type": "stock", "source": "baostock"},
        {"symbol": "BTC/USDT", "name": "Bitcoin", "asset_type": "crypto", "source": "ccxt"},
        {"symbol": "ETH/USDT", "name": "Ethereum", "asset_type": "crypto", "source": "ccxt"},
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
        """加载配置：yaml 覆盖代码默认值，环境变量再覆盖（密钥类）。"""
        path = Path(yaml_path) if yaml_path else ROOT / "config" / "default.yaml"
        raw: dict[str, Any] = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = cls(**raw)

        # 环境变量覆盖（LLM 密钥等）
        env_map: dict[str, tuple[str, ...]] = {
            "LLM_API_KEY": ("llm", "api_key"),
            "LLM_BASE_URL": ("llm", "base_url"),
            "LLM_MODEL": ("llm", "model"),
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
