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
                 freq: str = "D", use_start: bool = False, flat: bool = False):
        self.healthy = healthy
        self.fail_fetch = fail_fetch
        self.window_start = window_start
        self.periods = periods
        self.freq = freq
        self.use_start = use_start
        self.flat = flat  # 平价数据（open==close），避免权重漂移
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
        if self.use_start:
            idx = pd.date_range(start=start, periods=self.periods, freq=self.freq, tz="UTC")
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
