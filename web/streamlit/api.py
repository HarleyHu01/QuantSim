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
        # quote：symbol 可能带 . 或特殊字符；后端 {symbol:path} 路由会原样捕获
        from urllib.parse import quote
        r = self._client.delete(self.base + f"/api/data/symbols/{quote(symbol, safe='/')}")
        r.raise_for_status()
        return r.json()

    def data_channels(self) -> dict:
        """下载渠道能力表：每个源支持哪些周期 / 最长回溯天数。"""
        return self._get("/api/data/channels")

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

    def settings_llm(self) -> dict:
        return self._get("/api/settings/llm")

    def settings_llm_put(self, cfg: dict) -> dict:
        return self._put("/api/settings/llm", cfg)

    def proxy_get(self) -> dict:
        """数据源逐渠道代理模式（auto/direct/proxy）。"""
        return self._get("/api/settings/proxy")

    def proxy_put(self, modes: dict) -> dict:
        return self._put("/api/settings/proxy", {"modes": modes})

    def paths_get(self) -> dict:
        """本机运行时路径（数据目录 / 数据库 / 行情目录 / 覆盖配置来源）。"""
        return self._get("/api/settings/paths")

    def paths_put(self, data_dir: str | None) -> dict:
        """写本机数据目录覆盖（config/user.yaml），需重启后端生效。"""
        return self._put("/api/settings/paths", {"data_dir": data_dir})

    # ---- AI 策略生成 ----
    def llm_status(self) -> dict:
        return self._get("/api/llm/status")

    def llm_ping(self) -> dict:
        return self._post("/api/llm/ping")

    def llm_models(self) -> dict:
        return self._get("/api/llm/models")

    def llm_chat(self, message: str, session_id: int | None = None, mode: str = "code",
                 intent: str = "write", code: str | None = None,
                 config: dict | None = None, context: dict | None = None,
                 temperature: float | None = None, provider: str | None = None,
                 model: str | None = None) -> dict:
        """一轮对话。

        intent: write=写新策略 / revise=改当前版本 / ideas=只要思路
        带上 code（代码模式）或 config（规则模式）即视为「改当前版本」。
        provider: None=DeepSeek/OpenAI 兼容 / "workbuddy"=WorkBuddy 模型。
        """
        return self._post("/api/llm/chat", {
            "message": message, "session_id": session_id, "mode": mode,
            "intent": intent, "code": code, "config": config,
            "context": context, "temperature": temperature,
            "provider": provider, "model": model})

    # ---- WorkBuddy（可选模型来源）----
    def workbuddy_status(self) -> dict:
        return self._get("/api/workbuddy/status")

    def workbuddy_models(self) -> dict:
        return self._get("/api/workbuddy/models")

    def workbuddy_config(self) -> dict:
        return self._get("/api/workbuddy/config")

    def workbuddy_calibrate(self, exe_path: str = "") -> dict:
        return self._post("/api/workbuddy/calibrate", {"exe_path": exe_path})

    def llm_validate(self, mode: str = "code", code: str | None = None,
                     config: dict | None = None, params: dict | None = None) -> dict:
        return self._post("/api/llm/validate", {
            "mode": mode, "code": code, "config": config, "params": params})

    def llm_sessions(self) -> list[dict]:
        return self._get("/api/llm/sessions")["sessions"]

    def llm_session(self, sid: int) -> dict:
        return self._get(f"/api/llm/sessions/{sid}")

    def llm_session_new(self, mode: str = "code") -> dict:
        return self._post("/api/llm/sessions", mode=mode)

    def llm_save(self, **kw) -> dict:
        return self._post("/api/llm/save", kw)

    def llm_code(self, sid: int) -> dict:
        return self._get(f"/api/llm/code/{sid}")

    # ---- 机器学习（Kronos / 多模型集成） ----
    def ml_kronos_status(self) -> dict:
        return self._get("/api/ml/kronos/status")

    def ml_kronos_load(self) -> dict:
        return self._post("/api/ml/kronos/load")

    def ml_kronos_predict(self, symbol: str, timeframe: str = "5m", bars: int = 400,
                          pred_len: int = 12, model: str | None = None,
                          sample_count: int | None = None) -> dict:
        return self._post("/api/ml/kronos/predict", {
            "symbol": symbol, "timeframe": timeframe, "bars": bars,
            "pred_len": pred_len, "model": model, "sample_count": sample_count or 1})

    def ml_models(self) -> dict:
        return self._get("/api/ml/models")

    def ml_ensemble_config(self) -> dict:
        return self._get("/api/ml/ensemble/config")

    def ml_ensemble_config_put(self, **kw) -> dict:
        return self._put("/api/ml/ensemble/config", {k: v for k, v in kw.items() if v is not None})

    def ml_ensemble_predict(self, symbol: str, timeframe: str = "5m", bars: int = 400,
                            pred_len: int | None = None,
                            weights: dict | None = None) -> dict:
        return self._post("/api/ml/ensemble/predict", {
            "symbol": symbol, "timeframe": timeframe, "bars": bars,
            "pred_len": pred_len, "weights": weights})

    def ml_auto_weights(self, symbol: str, timeframe: str = "5m", pred_len: int = 12,
                        eval_bars: int = 240, step: int = 20, context: int = 240,
                        keys: list[str] | None = None) -> dict:
        return self._post("/api/ml/ensemble/auto-weights", {
            "symbol": symbol, "timeframe": timeframe, "pred_len": pred_len,
            "eval_bars": eval_bars, "step": step, "context": context, "keys": keys})

    # ---- 模型 × 因子融合 ----
    def fusion_config(self) -> dict:
        """融合配置 + 可选模型/因子清单 + 融合方式说明。"""
        return self._get("/api/fusion/config")

    def fusion_config_put(self, **kw) -> dict:
        return self._put("/api/fusion/config", {k: v for k, v in kw.items() if v is not None})

    def fusion_preview(self, symbol: str, timeframe: str = "5m", bars: int = 400,
                       config: dict | None = None) -> dict:
        """用真实行情试算一次融合决策（不写库）。"""
        return self._post("/api/fusion/preview", {
            "symbol": symbol, "timeframe": timeframe, "bars": bars, "config": config})

    # ---- 回测 ----
    def backtest_combo(self, body: dict) -> dict:
        return self._post("/api/backtest/combo", body)

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
