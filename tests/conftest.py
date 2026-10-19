"""pytest 公共夹具：临时配置、Fake 数据源、DataManager。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from quant.core.config import AppConfig
from quant.data.base import DataProvider
from quant.data.manager import DataManager
from quant.data.storage import MarketStore
from quant.storage.db import init_db
from quant.storage.repositories import MarketDataRepo, SymbolsRepo


class FakeProvider(DataProvider):
    """可控假数据源：healthy 可切换，失败可注入，数据窗口可配置。

    - window_start/periods/freq: 固定数据窗口（默认 10 根日线 2024-01-01 起）
    - use_start=True: 窗口起点取 fetch_history 的 start 参数（用于增量/补历史测试）
    """

    name = "fake"
    display_name = "Fake"
    asset_types = ("stock",)

    def __init__(self, healthy: bool = True, fail_fetch: bool = False,
                 window_start: str = "2024-01-01", periods: int = 10,
                 freq: str = "D", use_start: bool = False, flat: bool = False,
                 anchor_end: bool = False):
        self.healthy = healthy
        self.fail_fetch = fail_fetch
        self.window_start = window_start
        self.periods = periods
        self.freq = freq
        self.use_start = use_start
        self.flat = flat  # 平价数据（open==close），避免权重漂移
        # anchor_end=True：把 periods 根 bar 的**最后一根对齐到请求的 end**（即「现在」）。
        # 模拟盘/实时测试需要数据紧贴当下，否则以一年前为起点算出的 bar 会让
        # 「首次 tick 只补算最近 N 根」的断言随运行时刻漂移。
        self.anchor_end = anchor_end
        self.calls = 0

    def _ping(self) -> None:
        if not self.healthy:
            raise RuntimeError("fake down")

    def fetch_history(self, symbol, start, end, timeframe="1d", adjust="none"):
        self.calls += 1
        if not self.healthy:
            raise RuntimeError("fake down")
        if self.fail_fetch:
            raise RuntimeError("fake fetch boom")
        if self.use_start and self.anchor_end and end is not None:
            idx = pd.date_range(end=end, periods=self.periods, freq=self.freq, tz="UTC")
        elif self.use_start:
            idx = pd.date_range(start=start, periods=self.periods, freq=self.freq, tz="UTC")
            # 不编造落在请求区间之外（尤其未来）的 bar —— 真实数据源不会返回 end 之后的数据。
            # 否则增量拉取时 start 已经贴近当下，periods 根会一路排到未来，
            # 让「首次 tick 只补算最近 N 根」这类断言随运行时刻漂移而失败。
            if end is not None:
                end_ts = pd.Timestamp(end)
                if end_ts.tzinfo is None:
                    end_ts = end_ts.tz_localize("UTC")
                idx = idx[idx <= end_ts]
        else:
            idx = pd.date_range(self.window_start, periods=self.periods, freq=self.freq, tz="UTC")
        if self.flat:
            return pd.DataFrame(
                {"open": 10.5, "high": 10.5, "low": 10.5, "close": 10.5, "volume": 1000.0},
                index=idx,
            )
        return pd.DataFrame(
            {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000.0},
            index=idx,
        )


class BrokenProvider(DataProvider):
    name = "broken"
    display_name = "Broken"
    asset_types = ("stock",)

    def _ping(self) -> None:
        raise RuntimeError("always down")

    def fetch_history(self, symbol, start, end, timeframe="1d", adjust="none"):
        raise RuntimeError("always down")


@pytest.fixture
def tmp_config(tmp_path) -> AppConfig:
    cfg = AppConfig(
        data_dir=tmp_path,
        db_path=tmp_path / "quant.db",
        market_dir=tmp_path / "market",
        symbols=[{"symbol": "TEST", "name": "Test", "asset_type": "stock", "source": "fake"}],
    )
    return cfg


@pytest.fixture
def manager(tmp_config) -> DataManager:
    init_db(tmp_config.effective_db_path)
    tmp_config.data.providers.stock = ["fake"]  # 测试环境只注册 fake 数据源
    store = MarketStore(tmp_config.effective_market_dir)
    dm = DataManager(
        tmp_config, store,
        MarketDataRepo(tmp_config.effective_db_path),
        SymbolsRepo(tmp_config.effective_db_path),
        providers={"fake": FakeProvider()},
    )
    dm.symbols_repo.upsert(symbol="TEST", name="Test", asset_type="stock", source="fake")
    return dm
