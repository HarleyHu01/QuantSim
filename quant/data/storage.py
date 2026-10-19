"""Parquet 行情存储：按 (symbol, timeframe) 一个文件，增量合并去重。

文件内列：datetime / open / high / low / close / volume / adj_factor
（datetime 以普通列写入，读取时还原为 index，避免索引序列化歧义）
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant.core.models import ALL_COLUMNS
from quant.core.utils import safe_filename

# 周期 → 文件名安全 token（Windows 文件系统大小写不敏感，1m/1M 必须区分）
TF_FILE_TOKEN = {"1M": "1mo"}


def _tf_token(timeframe: str) -> str:
    return TF_FILE_TOKEN.get(timeframe, timeframe)


class MarketStore:
    def __init__(self, market_dir: Path):
        self.market_dir = Path(market_dir)
        self.market_dir.mkdir(parents=True, exist_ok=True)

    # ---- 路径 ----
    def path_for(self, symbol: str, timeframe: str) -> Path:
        return self.market_dir / f"{safe_filename(symbol)}_{_tf_token(timeframe)}.parquet"

    def exists(self, symbol: str, timeframe: str) -> bool:
        return self.path_for(symbol, timeframe).exists()

    # ---- 读取 ----
    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        p = self.path_for(symbol, timeframe)
        if not p.exists():
            return self._empty_frame()
        df = pd.read_parquet(p)
        if "datetime" in df.columns:
            df = df.set_index(pd.to_datetime(df.pop("datetime"), utc=True))
        return df

    # ---- 写入（增量合并） ----
    def save(self, symbol: str, timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
        """与本地已有数据按时间戳去重合并后写回，返回合并结果。"""
        existing = self.load(symbol, timeframe)
        new = df.copy()
        # 列对齐：缺失列补默认值（防止 concat 产生 NaN 列）
        for col in ALL_COLUMNS:
            if col not in new.columns:
                new[col] = 1.0 if col == "adj_factor" else 0.0
        merged = pd.concat([existing, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        out = merged.copy()
        out.index.name = "datetime"
        out.reset_index().to_parquet(self.path_for(symbol, timeframe), index=False)
        return out

    def delete(self, symbol: str, timeframe: str) -> None:
        p = self.path_for(symbol, timeframe)
        if p.exists():
            p.unlink()

    def _empty_frame(self) -> pd.DataFrame:
        idx = pd.DatetimeIndex([], tz="UTC", name="datetime")
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in ALL_COLUMNS}, index=idx)
