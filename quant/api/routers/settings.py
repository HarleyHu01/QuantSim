"""设置 API：交易设置 / LLM 配置持久化 / 逐渠道代理 / 本机路径。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from quant.api.state import app_state
from quant.core.models import TradingSettings

router = APIRouter(prefix="/settings", tags=["settings"])


def _repo():
    repo = app_state.get("settings_repo")
    if repo is None:
        raise HTTPException(status_code=503, detail="设置仓储未就绪")
    return repo


class TradingSettingsIn(BaseModel):
    """与 TradingSettings 同构（全部有默认值，UI 表单逐字段提交）。"""
    initial_capital: float | None = None
    commission_rate: float | None = None
    commission_fixed: float | None = None
    commission_min: float | None = None
    stamp_tax_rate: float | None = None
    transfer_fee_rate: float | None = None
    slippage_bps: float | None = None
    margin_rate: float | None = None
    margin_call_threshold: float | None = None
    timeframe: str | None = None
    min_trade_interval_bars: int | None = None
    cooldown_bars: int | None = None
    rebalance_freq: str | None = None
    signal_threshold: float | None = None
    execution: str | None = None
    max_position_pct: float | None = None
    max_exposure_pct: float | None = None
    allow_short: bool | None = None
    t_plus_enabled: bool | None = None
    max_drawdown_halt: float | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    daily_loss_limit_pct: float | None = None
    max_positions: int | None = None
    benchmark_symbol: str | None = None
    periods_per_year: int | None = None


class LLMSettingsIn(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    api_protocol: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    temperature: float | None = None


def _merged(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in updates.items():
        if v is not None:
            out[k] = v
    return out


@router.get("/trading")
def get_trading_settings() -> dict:
    """返回完整交易设置（默认值 + 已保存覆盖）。"""
    saved = _repo().get_trading() or {}
    return TradingSettings(**saved).model_dump()


@router.put("/trading")
def put_trading_settings(body: TradingSettingsIn) -> dict:
    saved = _repo().get_trading() or {}
    merged = _merged(saved, body.model_dump(exclude_none=True))
    # 校验（非法字段抛 pydantic 错误）
    settings = TradingSettings(**merged)
    _repo().put_trading(settings.model_dump())
    return settings.model_dump()


@router.get("/llm")
def get_llm_settings() -> dict:
    from quant.core.config import LLMConfig
    saved = _repo().get_llm() or {}
    return LLMConfig(**saved).model_dump()


@router.put("/llm")
def put_llm_settings(body: LLMSettingsIn) -> dict:
    from quant.core.config import LLMConfig
    saved = _repo().get_llm() or {}
    merged = _merged(saved, body.model_dump(exclude_none=True))
    cfg = LLMConfig(**merged)
    _repo().put_llm(cfg.model_dump())
    return cfg.model_dump()


# ---- 数据源逐渠道代理设置 ----

class ProxySettingsIn(BaseModel):
    """逐数据源代理模式：auto=跟随系统 / direct=强制直连 / proxy=强制走代理。"""
    modes: dict[str, str] | None = None  # {provider_name: mode}；None = 用默认值


@router.get("/proxy")
def get_proxy_settings() -> dict:
    """返回每个数据源当前的代理模式（默认值 + 已保存覆盖）。"""
    from quant.data.manager import load_proxy_modes

    modes = load_proxy_modes()
    from quant.data.proxy import PROXY_MODES
    return {"modes": modes, "available_modes": list(PROXY_MODES)}


@router.put("/proxy")
def put_proxy_settings(body: ProxySettingsIn) -> dict:
    """保存逐渠道代理模式并立即生效（写 kv_store + 更新运行时注册表）。"""
    from quant.core.config import AppConfig
    from quant.data.manager import save_proxy_modes
    from quant.data.proxy import PROXY_MODES

    modes = body.modes or {}
    # 过滤非法模式，避免写入坏值
    clean = {k: v for k, v in modes.items() if v in PROXY_MODES}
    saved = save_proxy_modes(AppConfig.load(), clean)
    return {"modes": saved, "available_modes": list(PROXY_MODES)}


# ---- 本机路径与环境（可读可改，避免「这台电脑专属」参数写死在代码里） ----

def _path_info(p, *, must_exist: bool = False) -> dict:
    path = Path(p)
    try:
        exists = path.exists()
        writable = os.access(path if exists else path.parent, os.W_OK)
    except OSError:
        exists, writable = False, False
    size = 0
    if exists:
        try:
            if path.is_dir():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            else:
                size = path.stat().st_size
        except OSError:
            size = 0
    return {
        "path": str(path),
        "exists": exists,
        "writable": writable,
        "is_dir": path.is_dir(),
        "size_bytes": size,
    }


class PathsIn(BaseModel):
    """本机路径覆盖。留空/None = 用默认（项目内 data/）。

    改动会写入 ``config/user.yaml``（本机覆盖层，不随升级覆盖），
    但**数据目录的切换需要重启后端**才彻底生效（已加载的 sqlite 连接不会自动迁移）。
    """

    data_dir: str | None = None


@router.get("/paths")
def get_paths() -> dict:
    """返回运行时的关键路径、默认值与来源（代码默认 / user.yaml / 环境变量）。"""
    from quant.core.config import ROOT, USER_CONFIG, AppConfig

    cfg = AppConfig.load()
    env_data = os.getenv("DATA_DIR")
    user_yaml: dict = {}
    if USER_CONFIG.exists():
        try:
            import yaml

            user_yaml = yaml.safe_load(USER_CONFIG.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            user_yaml = {}

    if env_data:
        source = "环境变量 DATA_DIR"
    elif user_yaml.get("data_dir"):
        source = "config/user.yaml"
    else:
        source = "默认（项目内 data/）"

    return {
        "root": str(ROOT),
        "python": sys.executable,
        "user_config": str(USER_CONFIG),
        "user_config_exists": USER_CONFIG.exists(),
        "data_dir": _path_info(cfg.effective_db_path.parent),
        "db_path": _path_info(cfg.effective_db_path),
        "market_dir": _path_info(cfg.effective_market_dir),
        "defaults": {
            "data_dir": str(ROOT / "data"),
            "db_path": str(ROOT / "data" / "quant.db"),
            "market_dir": str(ROOT / "data" / "market"),
        },
        "data_dir_source": source,
        "api": {"host": cfg.api_host, "port": cfg.api_port},
    }


@router.put("/paths")
def put_paths(body: PathsIn) -> dict:
    """写本机路径覆盖到 config/user.yaml（深度合并，保留其他字段）。"""
    from quant.core.config import USER_CONFIG

    raw: dict = {}
    if USER_CONFIG.exists():
        try:
            import yaml

            raw = yaml.safe_load(USER_CONFIG.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            raw = {}

    value = (body.data_dir or "").strip()
    if value:
        p = Path(value).expanduser()
        if not p.is_absolute():
            # 相对路径不允许：数据目录写成相对于「当前工作目录」极易漂移
            raise HTTPException(status_code=400,
                                detail="数据目录请填绝对路径（例如 D:\\QuantSimData）")
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as e:
            raise HTTPException(status_code=400,
                                detail=f"该目录不可写或无法创建: {e}") from e
        raw["data_dir"] = str(p)
    else:
        raw.pop("data_dir", None)

    try:
        import yaml

        USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        if raw:
            USER_CONFIG.write_text(
                "# QuantSim 本机覆盖配置（不进版本库；覆盖 config/default.yaml）\n"
                + yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
                encoding="utf-8")
        elif USER_CONFIG.exists():
            USER_CONFIG.unlink()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"写入 {USER_CONFIG} 失败: {e}") from e

    return {"saved": True, "data_dir": raw.get("data_dir"),
            "restart_required": True}
