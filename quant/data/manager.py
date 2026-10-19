"""DataManager：数据管理统一入口。

职责：
- 增量/全量更新（按 symbol × timeframe）
- 数据源 fallback：**按优先级链逐个渠道尝试，前一个失败/不支持才换下一个**
- 数据源健康检查
- 行情读取

渠道设计（不变量）：
    国内 A股/ETF/指数 → baostock（主） → akshare（备选，内部三通道：东财→腾讯→新浪）
    美股/海外        → yfinance

    腾讯原本是独立的第三级 provider，现**并入 akshare 内部**作为第二条腿：
    - 健康检查按「baostock / akshare / yfinance」三项呈现，akshare 再展开逐通道明细，
      不会把同一家腾讯在健康表里显示两遍
    - akshare 的代理开关一处即可覆盖三条腿（东财/腾讯/新浪都是国内直连）
    - TencentProvider 类保留在 ``quant/data/tencent_provider.py``，供直接引用与测试

「跳过」而不是「浪费一次请求」：每个 Provider 都声明了 ``supports(symbol, timeframe)``
与 ``max_lookback_days(symbol, timeframe)``，所以：
- baostock 没有 1m、指数没有分钟线 → 直接跳到 akshare，不发无效请求
- Yahoo 1m 只有 8 天 / 5m 只有 60 天 / 1h 只有 730 天 → 起点自动裁进窗口，避免必然 422
- 腾讯腿分钟线只有最近 640 根 → 起点按周期裁剪
这是「有时候下载不下来」的主要修复点之一。
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from quant.core.config import AppConfig
from quant.core.models import UpdateResult
from quant.core.utils import fmt_ts, parse_iso
from quant.data.base import DataProvider, ProviderHealth, is_domestic
from quant.data.storage import MarketStore
from quant.storage.repositories import MarketDataRepo, SymbolsRepo

log = logging.getLogger(__name__)

# 逐渠道代理模式持久化键（kv_store）
PROXY_CONFIG_KEY = "data.proxy"


def _load_proxy_modes(cfg: AppConfig) -> None:
    """从 kv_store 载入持久化的逐渠道代理模式，写入 proxy 模块运行时注册表。"""
    from quant.data import proxy
    from quant.storage.repositories import KVRepo

    saved = KVRepo(cfg.effective_db_path).get(PROXY_CONFIG_KEY)
    if isinstance(saved, dict):
        proxy.set_modes(saved)


def save_proxy_modes(cfg: AppConfig, modes: dict[str, str]) -> None:
    """持久化逐渠道代理模式并立即生效（写 kv_store + 更新运行时注册表）。"""
    from quant.data import proxy
    from quant.storage.repositories import KVRepo

    proxy.set_modes(modes)
    KVRepo(cfg.effective_db_path).put(PROXY_CONFIG_KEY, proxy.get_all_modes())


def load_proxy_modes(cfg: AppConfig | None = None) -> dict[str, str]:
    """读取当前逐渠道代理模式（默认值 + 持久化覆盖）。"""
    from quant.data import proxy

    return proxy.get_all_modes()

# 增量更新时多取的重叠 bar 数，避免边界缺口
OVERLAP_BARS = {"1m": 60, "5m": 12, "15m": 4, "30m": 2, "1h": 2, "1d": 2, "1w": 1, "1M": 1}
BAR_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
    "1d": 86400, "1w": 604800, "1M": 2592000,
}
# 各周期首次拉取的历史天数上限（**按真实数据源能力 + 实测耗时设定**，2026-10 逐源校验）
#
#   能力（能拿到多少）：
#     Yahoo：1m 8 天 / 5m,15m,30m 60 天 / 1h 730 天 / 日周月无限
#     baostock：无 1m / 5m,15m,30m,1h 至少 6 年 / 日周月到上市首日
#     腾讯：1m 约 2 个交易日 / 其余分钟 640 根 / 日周月可分页到 2001 年
#
#   耗时（baostock 5m 实测，含网络）：
#     30 天 1.7s / 90 天 2.6s / 180 天 16.8s / 365 天 21s / 730 天 65s（2.3 万根）
#
# 取值原则：**够用就好**。分钟线默认只取近 1 年，避免首次下载卡住几分钟；
# 需要更长的历史请显式传 start_date（不受此表限制）。
# 具体 Provider 还会用 max_lookback_days() 再收窄到自己的真实窗口。
TF_MAX_LOOKBACK_DAYS = {
    "1m": 8,        # 最紧的一档：Yahoo 8 天 / 腾讯约 2 个交易日
    "5m": 365,
    "15m": 365,
    "30m": 365,
    "1h": 730,      # baostock 实测 2020 年仍可取，但 730 天已够用
    "1d": 730,      # 由 config.data.lookback_days 进一步约束（默认 730）
    "1w": 3650,
    "1M": 7300,
}


class DataManager:
    def __init__(
        self,
        config: AppConfig,
        store: MarketStore,
        market_repo: MarketDataRepo,
        symbols_repo: SymbolsRepo,
        providers: dict[str, DataProvider],
    ):
        self.config = config
        self.store = store
        self.market_repo = market_repo
        self.symbols_repo = symbols_repo
        self.providers = providers
        self._health_cache: tuple[float, dict[str, ProviderHealth]] | None = None

    @classmethod
    def create(cls, config: AppConfig | None = None) -> "DataManager":
        """按配置构建完整 DataManager（初始化 DB、存储与数据源）。"""
        cfg = config or AppConfig.load()
        from quant.data.akshare_provider import AkshareProvider
        from quant.data.baostock_provider import BaostockProvider
        from quant.data.yfinance_provider import YFinanceProvider
        from quant.storage.db import init_db

        init_db(cfg.effective_db_path)
        store = MarketStore(cfg.effective_market_dir)
        yf_cache_dir = str(cfg.data_dir / ".yf_cache")
        # 载入持久化的「逐渠道代理模式」，供各 Provider 请求时读取
        _load_proxy_modes(cfg)
        providers: dict[str, DataProvider] = {}
        # 腾讯不再作为独立 provider：它现在是 akshare 内部的第二条腿
        # （东财 → 腾讯 → 新浪），这样健康检查与「akshare」这一个代理开关就能覆盖三条腿，
        # 也不会在健康表里把同一家腾讯显示两遍。TencentProvider 类保留，供直接引用。
        for p in (BaostockProvider(),                     # 国内主渠道
                  AkshareProvider(use_proxy=cfg.data.akshare_use_proxy),  # 国内备选（三通道）
                  YFinanceProvider(cache_dir=yf_cache_dir)):  # 美股/海外
            try:
                providers[p.name] = p
            except Exception as e:  # noqa: BLE001
                log.warning("数据源 %s 初始化失败，已跳过: %s", p.name, e)
        return cls(
            cfg, store,
            MarketDataRepo(cfg.effective_db_path),
            SymbolsRepo(cfg.effective_db_path),
            providers,
        )

    def provider_names(self) -> list[str]:
        return sorted(self.providers)

    def warm_up(self) -> None:
        """后台预热：提前建立 baostock 登录会话。

        为什么需要：baostock 的 ``login()`` 实测要 **11~13 秒**（服务器侧握手慢），
        而 ``DataManager.health()`` 的总预算只有 8 秒。若等到第一次健康检查才登录，
        必然报「健康检查超时」——用户看到的是「baostock 不通」，实际只是没预热。

        因此在 app 启动时后台起一个线程先登录一次，等用户打开页面时
        会话已就绪，健康检查只需 1~2 秒。失败也无所谓（下次查询会自行登录）。
        """
        def _run() -> None:
            try:
                from quant.data.baostock_provider import _ensure_login, _bs_lock
                with _bs_lock:
                    _ensure_login()
                log.info("baostock 会话预热完成")
            except Exception as e:  # noqa: BLE001
                log.debug("baostock 会话预热失败（不影响后续使用）: %s", e)

        threading.Thread(target=_run, name="baostock-warmup", daemon=True).start()

    # ---- 健康检查（并发 + 总预算超时 + TTL 缓存，避免 UI 阻塞） ----
    HEALTH_CACHE_TTL = 10.0  # 秒

    def health(self, timeout: float = 8.0, force: bool = False) -> dict[str, ProviderHealth]:
        """健康检查；force=True 强制绕过 TTL 缓存做真实检测。

        - latency_ms 始终为真实网络请求耗时
        - 缓存命中时保留上次真实延迟，并标注 cached=True / cached_age_seconds / checked_at
        """
        now = time.monotonic()
        if (not force and self._health_cache is not None
                and now - self._health_cache[0] < self.HEALTH_CACHE_TTL):
            ts, out = self._health_cache
            age = now - ts
            for h in out.values():
                h.cached = True
                h.cached_age_seconds = round(age, 1)
            return out
        if not self.providers:
            return {}
        ex = ThreadPoolExecutor(max_workers=len(self.providers))
        try:
            futures = {name: ex.submit(p.health_check) for name, p in self.providers.items()}
            done, _ = wait(futures.values(), timeout=timeout)
            out: dict[str, ProviderHealth] = {}
            for name, fut in futures.items():
                if fut in done:
                    try:
                        out[name] = fut.result()
                    except Exception as e:  # noqa: BLE001
                        out[name] = ProviderHealth(healthy=False, error=f"{type(e).__name__}: {e}")
                else:
                    out[name] = ProviderHealth(healthy=False, error="健康检查超时")
        finally:
            # 关键：wait=False，不等残留线程（否则未完成的 ping 会拖慢整体）
            ex.shutdown(wait=False, cancel_futures=True)
        checked_ts = time.time()
        for h in out.values():
            h.checked_at = checked_ts
            h.cached = False
            h.cached_age_seconds = None
        self._health_cache = (now, out)
        return out

    # ---- 更新 ----
    def update(
        self,
        symbols: Iterable[str] | None = None,
        timeframes: Iterable[str] | None = None,
        incremental: bool | None = None,
        adjust: str | None = None,
        full_reload: bool = False,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[UpdateResult]:
        """一键更新。

        - symbols/timeframes 缺省时使用全部启用标的 / 配置默认周期
        - full_reload=True：清空本地数据后全量重拉（复权方式改变等场景）
        - start_date="YYYY-MM-DD"：拉取起点（backfill 历史）
        - end_date="YYYY-MM-DD"：拉取终点（默认今天）；与 start_date 一起可精确指定时间段
        """
        cfg = self.config
        tf_list = list(timeframes) if timeframes else list(cfg.data.default_timeframes)
        sym_list = list(symbols) if symbols else [s["symbol"] for s in self.symbols_repo.list_enabled()]
        adjust = adjust or cfg.data.adjust

        results: list[UpdateResult] = []
        for symbol in sym_list:
            meta = self.symbols_repo.get(symbol)
            if meta is None:
                # 自动注册未知标的（按格式推断资产类型/数据源），方便 UI/CLI 直接输入新代码
                from quant.data.base import infer_symbol_type
                asset_type, source = infer_symbol_type(symbol)
                self.symbols_repo.upsert(
                    symbol=symbol, name=symbol, asset_type=asset_type, source=source,
                    price_limit_pct=0.10 if asset_type == "stock" else None,
                )
                meta = self.symbols_repo.get(symbol)
            for tf in tf_list:
                results.append(self._update_one(symbol, tf, meta, incremental, adjust,
                                                full_reload, start_date, end_date))
        return results

    def _update_one(
        self,
        symbol: str,
        timeframe: str,
        symbol_meta: dict | None,
        incremental: bool | None,
        adjust: str,
        full_reload: bool = False,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> UpdateResult:
        cfg = self.config
        now = datetime.now(timezone.utc)
        do_incremental = cfg.data.incremental if incremental is None else incremental
        stored = self.market_repo.get_meta(symbol, timeframe)

        # 全量重拉（--full）：拉取范围改为全量；数据替换在拉取成功后才执行，
        # 避免"先删后拉、拉取失败导致数据永久丢失"。
        if full_reload:
            do_incremental = False

        end = now if end_date is None else _as_utc_ts(end_date).to_pydatetime()
        # 拉取起点
        if start_date is not None:
            start = _as_utc_ts(start_date).to_pydatetime()
        elif do_incremental and stored and stored.get("last_bar_time"):
            last_bar = parse_iso(stored["last_bar_time"])
            if last_bar is not None and last_bar.tzinfo is None:
                last_bar = last_bar.replace(tzinfo=timezone.utc)  # 存储值按 UTC 解释
            overlap = timedelta(seconds=BAR_SECONDS.get(timeframe, 86400) * OVERLAP_BARS.get(timeframe, 2))
            start = last_bar - overlap if last_bar else now - timedelta(days=cfg.data.lookback_days)
        else:
            lookback = min(cfg.data.lookback_days, TF_MAX_LOOKBACK_DAYS.get(timeframe, cfg.data.lookback_days))
            start = end - timedelta(days=lookback)
        if end < start:
            return UpdateResult(symbol, timeframe, False, error="时间段不合法",
                                message="结束日期早于开始日期")

        chain = self._provider_chain(symbol_meta, symbol)
        if not chain:
            return UpdateResult(symbol, timeframe, False, error="无可用数据源",
                                message="该资产类型没有可用数据源（可能未安装或全部不可用）")

        errs: list[str] = []
        skipped: list[str] = []
        for provider in chain:
            # ---- 能力预检：不支持就别浪费一次网络请求 ----
            try:
                if not provider.supports(symbol, timeframe):
                    skipped.append(f"{provider.name}: 不支持 {timeframe}")
                    continue
            except Exception:  # noqa: BLE001
                pass
            # ---- 起止裁剪：超出该源窗口的部分必然失败，先裁掉 ----
            p_start = self._clip_start_by_provider(provider, symbol, timeframe, start, end)
            if p_start is not None and p_start >= end:
                skipped.append(f"{provider.name}: 该周期可回溯范围不含请求区间")
                continue
            use_start = p_start or start
            try:
                df = provider.fetch_history(symbol, use_start, end, timeframe, adjust=adjust)
                if df is None or df.empty:
                    raise ValueError("返回空数据")
                # 本次实际拉取区间（供结果反馈；与合并后的存储区间区分）
                fetched_min = df.index.min()
                fetched_max = df.index.max()
                if full_reload:
                    # 拉取成功后才替换旧数据（清空本地 + 元信息）
                    self.store.delete(symbol, timeframe)
                    self.market_repo.delete_meta(symbol, timeframe)
                merged = self.store.save(symbol, timeframe, df)
                merged_min = merged.index.min()
                merged_max = merged.index.max()
                self.market_repo.upsert_meta(
                    symbol=symbol, timeframe=timeframe, source=provider.name,
                    rows=len(merged), start_date=fmt_ts(merged_min), end_date=fmt_ts(merged_max),
                    last_bar_time=fmt_ts(merged_max), adjust=adjust,
                    file_path=str(self.store.path_for(symbol, timeframe)),
                )
                note = "更新成功"
                if skipped:
                    note += f"（已跳过：{'; '.join(skipped)}）"
                return UpdateResult(
                    symbol=symbol, timeframe=timeframe, ok=True, source=provider.name,
                    rows=len(df), start=fmt_ts(fetched_min), end=fmt_ts(fetched_max),
                    last_bar=fmt_ts(fetched_max), message=note,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("[%s/%s] 渠道 %s 拉取失败: %s", symbol, timeframe, provider.name, e)
                errs.append(f"{provider.name}: {e}")

        detail = "; ".join([*skipped, *errs]) if (skipped or errs) else "无可用渠道"
        return UpdateResult(
            symbol, timeframe, False,
            error=detail,
            message="所有下载渠道均失败",
        )

    @staticmethod
    def _clip_start_by_provider(provider: DataProvider, symbol: str, timeframe: str,
                                start: datetime, end: datetime) -> datetime | None:
        """按数据源自身能力裁剪起点；None = 该源对该周期完全不可用。"""
        try:
            days = provider.max_lookback_days(symbol, timeframe)
        except Exception:  # noqa: BLE001
            return start
        if days is None:
            return start
        earliest = end - timedelta(days=days)
        return start if start > earliest else earliest

    def _provider_chain(self, symbol_meta: dict | None, symbol: str = "") -> list[DataProvider]:
        """按优先级链返回候选渠道（前一个失败才用下一个）。

        国内 A股/ETF/指数 → providers.stock/etf/index（默认 baostock → akshare → tencent）
        美股/海外        → providers.us_stock（默认 yfinance）
        显式声明的 source 若不在链上（自定义/测试数据源）会被前插，保证它仍被使用。
        """
        asset_type = (symbol_meta or {}).get("asset_type") or "stock"
        source = (symbol_meta or {}).get("source")
        sym = (symbol_meta or {}).get("symbol") or symbol
        prio = self.config.data.providers
        if is_domestic(asset_type, source, sym):
            names = list(getattr(prio, asset_type, None) or prio.stock)
        else:
            names = list(prio.us_stock)
        if source and source not in names and source in self.providers:
            names.insert(0, source)
        chain = [self.providers[n] for n in names if n in self.providers]
        if not chain and self.providers:
            chain = list(self.providers.values())  # 兜底：配置里的数据源都已注册
        return chain

    # ---- 读取 ----
    def load_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        df = self.store.load(symbol, timeframe)
        if start is not None:
            df = df[df.index >= _as_utc_ts(start)]
        if end is not None:
            df = df[df.index <= _as_utc_ts(end)]
        return df

    # ---- 删除标的 ----
    def delete_symbol(self, symbol: str) -> int:
        """删除标的：移除符号池记录 + 全部已存行情（parquet 与元信息）。返回删除的数据文件数。"""
        self.symbols_repo.delete(symbol)
        removed = 0
        for meta in self.market_repo.list_meta():
            if meta["symbol"] == symbol:
                self.store.delete(symbol, meta["timeframe"])
                self.market_repo.delete_meta(symbol, meta["timeframe"])
                removed += 1
        return removed


def _as_utc_ts(s: str) -> pd.Timestamp:
    """字符串 → UTC tz-aware Timestamp（与存储的 UTC index 比较）。"""
    ts = pd.Timestamp(s)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
