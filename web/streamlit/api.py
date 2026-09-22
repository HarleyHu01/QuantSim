"""FastAPI 客户端（UI 只消费后端 API，便于以后整体替换 React）。"""
from __future__ import annotations

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class ApiClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=180)

    # ---- 基础 ----
    def health(self) -> dict:
        return self._get("/api/health")

    def shutdown(self) -> dict:
        return self._post("/api/system/shutdown", {})

    # ---- 数据 ----
    def data_status(self, refresh: bool = False) -> dict:
        return self._get("/api/data/status", refresh=1 if refresh else 0)

    def data_update(self, symbols=None, timeframes=None, incremental=None, adjust=None,
                    start_date=None, end_date=None) -> dict:
        body = {"symbols": symbols, "timeframes": timeframes,
                "incremental": incremental, "adjust": adjust,
                "start_date": start_date, "end_date": end_date}
        return self._post("/api/data/update", body)

    def add_symbol(self, symbol: str, asset_type: str | None = None,
                   source: str | None = None) -> dict:
        return self._post("/api/data/symbols", {"symbol": symbol,
                                                "asset_type": asset_type, "source": source})

    def delete_symbol(self, symbol: str) -> dict:
        # safe="/"：保留交易对斜杠，由后端的 {symbol:path} 路由捕获
        from urllib.parse import quote
        r = self._client.delete(self.base + f"/api/data/symbols/{quote(symbol, safe='/')}")
        r.raise_for_status()
        return r.json()

    def data_bars(self, symbol: str, timeframe: str = "1d", start=None, end=None) -> dict:
        return self._get(f"/api/data/bars/{symbol}",
                         timeframe=timeframe, start=start or "", end=end or "")

    def data_symbols(self) -> list[dict]:
        return self._get("/api/data/symbols")["symbols"]

    # ---- 策略 ----
    def strategies(self) -> list[dict]:
        return self._get("/api/strategies")["strategies"]

    def strategy_update(self, sid: int, **kw) -> dict:
        return self._put(f"/api/strategies/{sid}", kw)

    def strategy_toggle(self, sid: int) -> dict:
        return self._post(f"/api/strategies/{sid}/toggle")

    def indicators_meta(self) -> dict:
        return self._get("/api/strategies/indicators")

    def add_rule_strategy(self, config: dict, key: str | None = None,
                          enabled: bool = False) -> dict:
        body = {**config, "key": key, "enabled": enabled}
        return self._post("/api/strategies/rule", body)

    def delete_strategy(self, sid: int) -> dict:
        r = self._client.delete(self.base + f"/api/strategies/{sid}")
        r.raise_for_status()
        return r.json()

    # ---- 组合 ----
    def portfolio_get(self) -> dict:
        return self._get("/api/portfolio")

    def portfolio_put(self, cfg: dict) -> dict:
        return self._put("/api/portfolio", cfg)

    # ---- 设置 ----
    def settings_trading(self) -> dict:
        return self._get("/api/settings/trading")

    def settings_trading_put(self, cfg: dict) -> dict:
        return self._put("/api/settings/trading", cfg)

    # ---- 回测 ----
    def backtest_single(self, body: dict) -> dict:
        return self._post("/api/backtest/single", body)

    def backtest_combo(self, body: dict) -> dict:
        return self._post("/api/backtest/combo", body)

    def backtest_runs(self) -> list[dict]:
        return self._get("/api/backtest/runs")["runs"]

    def backtest_equity(self, run_id: int) -> list[dict]:
        return self._get(f"/api/backtest/runs/{run_id}/equity")["equity"]

    def backtest_trades(self, run_id: int) -> list[dict]:
        return self._get(f"/api/backtest/runs/{run_id}/trades")["trades"]

    # ---- 模拟盘 ----
    def paper_status(self, account_id: int | None = None) -> dict:
        return self._get("/api/paper/status", account_id=account_id)

    def paper_start(self, body: dict) -> dict:
        return self._post("/api/paper/start", body)

    def paper_stop(self, account_id: int) -> dict:
        return self._post("/api/paper/stop", {"account_id": account_id})

    def delete_account(self, account_id: int, cleanup: bool = True) -> dict:
        r = self._client.delete(self.base + f"/api/paper/accounts/{account_id}",
                                params={"cleanup": cleanup})
        r.raise_for_status()
        return r.json()

    def paper_positions(self, account_id: int) -> list[dict]:
        return self._get(f"/api/paper/positions", account_id=account_id)["positions"]

    def paper_trades(self, account_id: int) -> list[dict]:
        return self._get(f"/api/paper/trades", account_id=account_id)["trades"]

    # ---- HTTP 基元 ----
    def _get(self, path: str, **params) -> dict:
        r = self._client.get(self.base + path, params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json: dict | None = None) -> dict:
        r = self._client.post(self.base + path, json=json or {})
        r.raise_for_status()
        return r.json()

    def _put(self, path: str, json: dict | None = None) -> dict:
        r = self._client.put(self.base + path, json=json or {})
        r.raise_for_status()
        return r.json()
