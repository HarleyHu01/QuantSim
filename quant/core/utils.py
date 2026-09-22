"""通用工具函数。"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone

import pandas as pd

ISO_FMT = "%Y-%m-%d %H:%M:%S"
ISO_FMT_US = "%Y-%m-%d %H:%M:%S.%f"


def now_iso() -> str:
    """当前时间的统一 ISO 字符串。"""
    return datetime.now().strftime(ISO_FMT)


def parse_iso(s: str | None) -> datetime | None:
    """解析统一 ISO 字符串（支持带/不带微秒、T 分隔、纯日期）；失败返回 None。"""
    if not s:
        return None
    for fmt in (ISO_FMT_US, ISO_FMT, "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def fmt_ts(ts) -> str:
    """pandas Timestamp/datetime → 统一字符串（保留微秒，避免增量边界精度丢失）。"""
    if ts is None:
        return ""
    t = pd.Timestamp(ts)
    return t.strftime(ISO_FMT_US) if t.microsecond else t.strftime(ISO_FMT)


def ensure_utc_index(df: pd.DataFrame, assume_tz: str | None = None) -> pd.DataFrame:
    """把 DatetimeIndex 统一为 UTC（tz-aware）。

    无时区的 index 按 assume_tz（或本地时区）解释后再转 UTC；
    已有时区则直接转换。bar 时间语义 = bar 开盘时间。
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("index 必须是 DatetimeIndex")
    if idx.tz is None:
        tz = assume_tz or datetime.now().astimezone().tzinfo
        idx = idx.tz_localize(tz)
    return df.set_axis(idx.tz_convert("UTC"), axis=0)


REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """校验/清洗统一 OHLCV：必需列、去重、排序。返回新 DataFrame。"""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必需列: {missing}")
    cols = REQUIRED_COLUMNS + (["adj_factor"] if "adj_factor" in df.columns else [])
    out = df[cols].copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def safe_filename(symbol: str) -> str:
    """symbol → 安全文件名（去除 / . 等特殊字符）。"""
    return re.sub(r"[^0-9A-Za-z._-]", "_", symbol)


def detect_environment() -> str:
    """运行环境标识：存在 DSH_* 环境变量视为沙箱，否则本地。"""
    return "sandbox" if any(k.startswith("DSH_") for k in os.environ) else "local"
