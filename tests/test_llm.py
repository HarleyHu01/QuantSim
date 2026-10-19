"""LLM 策略生成 + 代码策略（可手改）测试。

覆盖：
- 源码静态检查（导入白名单/禁用调用/缺 generate_signals）
- 冒烟回测与返回契约校验
- CodeStrategy 执行、Params 解析、key 覆写
- 源码入库 → 注册表按 key 动态加载 → 产生信号
- LLM 路由：validate / save / get_code / put_code（含拒绝坏代码）
- 声明式配置容错归一化（{"all": [...]} 形式）
- LLM 客户端：无 key 报错、错误分类、URL 拼接
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from quant.storage.db import init_db
from quant.storage.repositories import LLMRepo, SettingsRepo, StrategiesRepo
from quant.strategies.code_strategy import (CodeStrategy, check_source, inspect_source,
                                           smoke_test, validate_signal_output)

GOOD_CODE = '''
class Strategy(BaseStrategy):
    key = "sma_cross"
    name = "均线交叉策略"
    description = "收盘价在 N 日均线上方做多"

    class Params(BaseModel):
        window: int = Field(20, ge=2, le=250)
        max_position: float = Field(1.0, ge=0.05, le=1.0)

    def generate_signals(self, bars, ctx):
        close = bars["close"]
        ma = indicators.sma(close, self.params.window)
        target = (close > ma).astype(float).fillna(0.0) * self.params.max_position
        reason = pd.Series("", index=bars.index, dtype=object)
        reason[close > ma] = "站上均线"
        return pd.DataFrame({"target_weight": target.clip(0.0, 1.0),
                             "reason": reason, "confidence": 1.0})
'''

BAD_COLUMNS_CODE = '''
class Strategy(BaseStrategy):
    def generate_signals(self, bars, ctx):
        return bars
'''


# ---------------- 源码检查 ----------------

def test_check_source_accepts_valid_code():
    assert check_source(GOOD_CODE) == []


def test_check_source_rejects_syntax_error():
    errors = check_source("class Strategy(BaseStrategy):\n    def generate_signals(:\n")
    assert errors and "语法错误" in errors[0]


def test_check_source_rejects_dangerous_calls():
    code = ('class Strategy(BaseStrategy):\n'
            '    def generate_signals(self, bars, ctx):\n'
            '        eval("1")\n'
            '        return bars\n')
    errors = check_source(code)
    assert any("eval" in e for e in errors)


def test_check_source_rejects_bad_imports():
    code = ('import os\n'
            'class Strategy(BaseStrategy):\n'
            '    def generate_signals(self, bars, ctx):\n'
            '        return bars\n')
    errors = check_source(code)
    assert any("os" in e for e in errors)


def test_check_source_requires_generate_signals():
    errors = check_source("class Foo:\n    pass\n")
    assert errors and any("generate_signals" in e for e in errors)


# ---------------- 执行与冒烟 ----------------

def test_code_strategy_runs_and_exposes_params():
    strat = CodeStrategy(GOOD_CODE, key="my_cross", name="我的交叉")
    assert strat.key == "my_cross"
    assert strat.name == "我的交叉"
    schema = strat.schema()
    assert "window" in schema["properties"]
    assert schema["properties"]["window"]["default"] == 20

    df = strat.generate_signals(_bars(), None)
    assert list(df.columns) == ["target_weight", "reason", "confidence"]
    assert len(df) == len(_bars())
    assert df["target_weight"].between(0.0, 1.0).all()
    assert df["target_weight"].abs().sum() > 0


def test_code_strategy_params_override():
    strat = CodeStrategy(GOOD_CODE, params={"window": 5})
    assert strat.params.window == 5


def test_code_strategy_rejects_bad_source():
    with pytest.raises(Exception):
        CodeStrategy("this is not python")


def test_validate_signal_output_detects_missing_columns():
    df = pd.DataFrame({"target_weight": [0.0, 1.0]})
    errors = validate_signal_output(df, 2)
    assert any("reason" in e for e in errors)


def test_smoke_test_passes_good_code():
    r = smoke_test(GOOD_CODE)
    assert r["ok"] is True
    assert r["signal_count"] > 0
    assert 0.0 <= r["min_weight"] <= r["max_weight"] <= 1.0


def test_smoke_test_fails_wrong_columns():
    r = smoke_test(BAD_COLUMNS_CODE)
    assert r["ok"] is False
    assert r["errors"]


def test_inspect_source_reports_metadata():
    info = inspect_source(GOOD_CODE)
    assert info["ok"] is True
    assert info["name"] == "均线交叉策略"
    assert info["params_defaults"]["window"] == 20


# ---------------- 入库 + 注册表加载 ----------------

def test_code_strategy_roundtrip_through_registry(tmp_path, monkeypatch):
    db = tmp_path / "quant.db"
    init_db(db)
    repo = StrategiesRepo(db)
    repo.upsert(key="my_cross", name="我的交叉", description="测试",
                strategy_type="code", class_name="Strategy",
                config_json={}, enabled=1, source_code=GOOD_CODE)

    row = repo.get("my_cross")
    assert row["source_code"] == GOOD_CODE
    assert row["strategy_type"] == "code"

    # 注册表默认加载器按 key 从库里还原
    import quant.strategies.registry as reg
    from quant.core.config import AppConfig

    monkeypatch.setattr(
        AppConfig, "load",
        classmethod(lambda cls, yaml_path=None: cls(data_dir=tmp_path, db_path=db,
                                                    market_dir=tmp_path / "market",
                                                    symbols=[])))
    strat = reg._default_rule_loader("my_cross")
    assert strat is not None
    assert strat.key == "my_cross"
    signals = strat.generate_signals(_bars(), None)
    assert signals["target_weight"].abs().sum() > 0


def test_code_strategy_key_is_unique_per_instance():
    a = CodeStrategy(GOOD_CODE, key="a")
    b = CodeStrategy(GOOD_CODE, key="b")
    assert a.key != b.key


def test_upsert_keeps_source_when_none(tmp_path):
    db = tmp_path / "quant.db"
    init_db(db)
    repo = StrategiesRepo(db)
    repo.upsert(key="k", name="n", strategy_type="code", source_code=GOOD_CODE)
    # 不传 source_code 不应该清空已存源码
    repo.upsert(key="k", name="n2", strategy_type="code", enabled=0)
    row = repo.get("k")
    assert row["source_code"] == GOOD_CODE
    assert row["enabled"] == 0
    assert row["name"] == "n2"


# ---------------- LLM 路由 ----------------

@pytest.fixture()
def llm_env(tmp_path, monkeypatch):
    """把路由依赖的三个仓储指向临时库。"""
    db = tmp_path / "quant.db"
    init_db(db)

    from quant.core.config import AppConfig
    monkeypatch.setattr(
        AppConfig, "load",
        classmethod(lambda cls, yaml_path=None: cls(data_dir=tmp_path, db_path=db,
                                                    market_dir=tmp_path / "market",
                                                    symbols=[])))

    from quant.api.state import app_state
    app_state["strategies_repo"] = StrategiesRepo(db)
    app_state["settings_repo"] = SettingsRepo(db)
    app_state["llm_repo"] = LLMRepo(db)
    try:
        yield {"db": db}
    finally:
        app_state.clear()


def test_llm_status_reports_unconfigured(llm_env):
    from quant.api.routers import llm as llm_router

    status = llm_router.llm_status()
    assert status["configured"] is False
    assert status["model"]
    assert {m["key"] for m in status["modes"]} == {"code", "rule"}


def test_llm_validate_code(llm_env):
    from quant.api.routers import llm as llm_router

    ok = llm_router.validate(llm_router.ValidateIn(mode="code", code=GOOD_CODE))
    assert ok["ok"] is True
    assert ok["inspect"]["ok"] is True

    bad = llm_router.validate(llm_router.ValidateIn(mode="code", code=BAD_COLUMNS_CODE))
    assert bad["ok"] is False
    assert bad["errors"]


def test_llm_save_and_edit_code(llm_env):
    from quant.api.routers import llm as llm_router

    saved = llm_router.save_strategy(
        llm_router.SaveIn(mode="code", name="我的均线", description="d",
                          code=GOOD_CODE, enabled=True))
    assert saved["ok"] is True
    sid = saved["strategy"]["id"]
    assert saved["strategy"]["strategy_type"] == "code"

    # 读取源码
    got = llm_router.get_code(sid)
    assert got["code"].strip().startswith("class Strategy")
    assert got["inspect"]["ok"] is True

    # 手改后保存
    edited = GOOD_CODE.replace('name = "均线交叉策略"', 'name = "均线交叉策略V2"')
    put = llm_router.put_code(sid, llm_router.CodeIn(code=edited))
    assert put["ok"] is True
    assert llm_router.get_code(sid)["inspect"]["name"] == "均线交叉策略V2"

    # 坏代码必须被拒绝且不入库
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        llm_router.put_code(sid, llm_router.CodeIn(code=BAD_COLUMNS_CODE))
    assert ei.value.status_code == 400
    assert "errors" in ei.value.detail
    assert llm_router.get_code(sid)["code"] == edited.strip()


def test_llm_save_rejects_bad_code(llm_env):
    from fastapi import HTTPException

    from quant.api.routers import llm as llm_router

    with pytest.raises(HTTPException) as ei:
        llm_router.save_strategy(llm_router.SaveIn(mode="code", name="坏策略",
                                                   code=BAD_COLUMNS_CODE))
    assert ei.value.status_code == 400


def test_llm_save_rule_config_normalizes_all_form(llm_env):
    from quant.api.routers import llm as llm_router

    cfg = {
        "name": "规则测试", "description": "",
        "indicators": [{"name": "sma", "params": {"window": 20}, "as": "ma20"}],
        "entry_long": {"all": [{"lhs": "close", "op": ">", "rhs": "ma20"}]},
        "exit_long": {"all": [{"lhs": "close", "op": "<", "rhs": "ma20"}]},
        "position": {"mode": "fixed_pct", "value": 1.0, "max_position": 1.0},
        "params": {"only_long": True},
    }
    v = llm_router.validate(llm_router.ValidateIn(mode="rule", config=cfg))
    assert v["ok"] is True
    assert v["config"]["entry_long"]["logic"] == "all"
    assert v["config"]["entry_long"]["conditions"][0]["lhs"] == "close"

    saved = llm_router.save_strategy(llm_router.SaveIn(mode="rule", config=cfg))
    assert saved["ok"] is True
    assert saved["strategy"]["strategy_type"] == "llm_generated"


def test_llm_put_code_forbids_builtin(llm_env):
    from fastapi import HTTPException

    from quant.api.routers import llm as llm_router

    repo = StrategiesRepo(llm_env["db"])
    sid = repo.upsert(key="dual_ma", name="双均线", strategy_type="builtin",
                      class_name="DualMaStrategy")
    with pytest.raises(HTTPException) as ei:
        llm_router.put_code(sid, llm_router.CodeIn(code=GOOD_CODE))
    assert ei.value.status_code == 403


def test_llm_sessions_persist_messages(llm_env):
    repo = LLMRepo(llm_env["db"])
    s = repo.create_session(title="测试会话", mode="code")
    repo.add_message(s["id"], "user", "写个策略")
    m = repo.add_message(s["id"], "assistant", "已生成", code=GOOD_CODE,
                         meta={"valid": True, "errors": []})
    msgs = repo.list_messages(s["id"])
    assert len(msgs) == 2
    assert msgs[1]["code"].strip().startswith("class Strategy")
    assert msgs[1]["meta"]["valid"] is True
    assert repo.get_message(m["id"]) is not None
    assert repo.list_sessions()[0]["message_count"] == 2

    repo.delete_session(s["id"])
    assert repo.get_session(s["id"]) is None
    assert repo.list_messages(s["id"]) == []


def test_llm_prompt_audit_log(llm_env):
    repo = LLMRepo(llm_env["db"])
    repo.log_prompt("写一个策略", model="deepseek-chat", mode="code", ok=1)
    rows = repo.list_prompts()
    assert rows and rows[0]["prompt"] == "写一个策略"
    assert rows[0]["ok"] == 1


# ---------------- 客户端 ----------------

def test_llm_client_requires_key():
    from quant.core.config import LLMConfig
    from quant.llm.client import LLMClient, LLMError

    client = LLMClient(LLMConfig(api_key=""))
    with pytest.raises(LLMError) as ei:
        client.chat([{"role": "user", "content": "hi"}])
    assert ei.value.status == 401
    assert "Key" in str(ei.value)


def test_llm_client_url_building():
    from quant.core.config import LLMConfig
    from quant.llm.client import LLMClient

    c = LLMClient(LLMConfig(base_url="https://api.deepseek.com", api_key="k"))
    # DeepSeek 官方 base_url 直接拼 /chat/completions（无需 /v1）
    assert c._url() == "https://api.deepseek.com/chat/completions"
    c2 = LLMClient(LLMConfig(base_url="https://x.com/v1", api_key="k"))
    assert c2._url() == "https://x.com/v1/chat/completions"
    c3 = LLMClient(LLMConfig(base_url="https://open.bigmodel.cn/api/paas/v4", api_key="k"))
    assert c3._url().endswith("/chat/completions")


def test_llm_client_protocol_urls():
    from quant.core.config import LLMConfig
    from quant.llm.client import LLMClient

    # Anthropic
    ca = LLMClient(LLMConfig(base_url="https://api.anthropic.com", api_key="k",
                             api_protocol="anthropic", model="claude-3-5-sonnet-20241022"))
    assert ca._url() == "https://api.anthropic.com/v1/messages"
    # Anthropic base_url 已含 /v1 时不重复拼
    ca2 = LLMClient(LLMConfig(base_url="https://api.anthropic.com/v1", api_key="k",
                              api_protocol="anthropic"))
    assert ca2._url() == "https://api.anthropic.com/v1/messages"

    # Gemini
    cg = LLMClient(LLMConfig(base_url="https://generativelanguage.googleapis.com",
                             api_key="k", api_protocol="gemini", model="gemini-1.5-pro"))
    assert cg._url() == ("https://generativelanguage.googleapis.com/v1beta/models/"
                         "gemini-1.5-pro:generateContent")


def test_llm_client_protocol_headers():
    from quant.core.config import LLMConfig
    from quant.llm.client import LLMClient

    co = LLMClient(LLMConfig(base_url="https://api.deepseek.com", api_key="k"))
    assert co._headers()["Authorization"] == "Bearer k"
    assert "x-api-key" not in co._headers()

    ca = LLMClient(LLMConfig(base_url="https://api.anthropic.com", api_key="k",
                             api_protocol="anthropic"))
    assert ca._headers()["x-api-key"] == "k"
    assert "anthropic-version" in ca._headers()
    assert "Authorization" not in ca._headers()

    cg = LLMClient(LLMConfig(base_url="https://generativelanguage.googleapis.com",
                             api_key="k", api_protocol="gemini"))
    assert cg._headers()["x-goog-api-key"] == "k"
    assert "Authorization" not in cg._headers()


def test_llm_client_protocol_default_is_openai():
    from quant.core.config import LLMConfig
    from quant.llm.client import LLMClient

    # 缺省 api_protocol 时默认 openai，行为与旧版一致
    c = LLMClient(LLMConfig(base_url="https://api.deepseek.com", api_key="k"))
    assert c.protocol == "openai"
    assert c._url().endswith("/chat/completions")


def test_llm_client_anthropic_chat_parsing(monkeypatch):
    from quant.core.config import LLMConfig
    from quant.llm.client import LLMClient

    client = LLMClient(LLMConfig(base_url="https://api.anthropic.com", api_key="k",
                                 api_protocol="anthropic", model="claude-3-5-sonnet-20241022"))
    sent = {}

    def fake_post(url, payload, headers):
        sent["url"] = url
        sent["payload"] = payload
        sent["headers"] = headers
        return {
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "```python\nclass Strategy(BaseStrategy):\n    pass\n```"}],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

    monkeypatch.setattr(client, "_post_json", fake_post)
    r = client.chat([
        {"role": "system", "content": "你是量化策略助手"},
        {"role": "user", "content": "写一个均线策略"},
    ], max_tokens=512)
    assert "class Strategy" in r.content
    assert r.model == "claude-3-5-sonnet-20241022"
    # system 从 messages 拆到顶层；content 是纯字符串
    assert sent["payload"]["system"] == "你是量化策略助手"
    assert sent["payload"]["messages"] == [{"role": "user", "content": "写一个均线策略"}]
    assert sent["payload"]["max_tokens"] == 512


def test_llm_client_gemini_chat_parsing(monkeypatch):
    from quant.core.config import LLMConfig
    from quant.llm.client import LLMClient

    client = LLMClient(LLMConfig(base_url="https://generativelanguage.googleapis.com",
                                 api_key="k", api_protocol="gemini", model="gemini-1.5-pro"))
    sent = {}

    def fake_post(url, payload, headers):
        sent["url"] = url
        sent["payload"] = payload
        return {
            "candidates": [{"content": {"parts": [{"text": "策略思路：均线金叉做多"}]}}],
            "usageMetadata": {"totalTokenCount": 42},
        }

    monkeypatch.setattr(client, "_post_json", fake_post)
    r = client.chat([{"role": "user", "content": "给个策略思路"}], max_tokens=256)
    assert "均线金叉" in r.content
    assert r.model == "gemini-1.5-pro"
    # user 角色保留，assistant → model；system → systemInstruction
    assert sent["payload"]["contents"] == [
        {"role": "user", "parts": [{"text": "给个策略思路"}]}]
    assert sent["payload"]["generationConfig"]["maxOutputTokens"] == 256


def test_resolve_llm_config_prefers_db_but_keeps_env_key(tmp_path, monkeypatch):
    from quant.core.config import AppConfig, LLMConfig
    from quant.llm.client import resolve_llm_config

    db = tmp_path / "quant.db"
    init_db(db)
    monkeypatch.setattr(
        AppConfig, "load",
        classmethod(lambda cls, yaml_path=None: cls(
            data_dir=tmp_path, db_path=db, market_dir=tmp_path / "market", symbols=[],
            llm=LLMConfig(api_key="from-env", model="deepseek-chat"))))
    repo = SettingsRepo(db)
    # UI 存了空 Key（未填写）时不能冲掉 .env 里的 Key
    repo.put_llm({"api_key": "", "model": "deepseek-reasoner"})
    cfg = resolve_llm_config(settings_repo=repo)
    assert cfg.api_key == "from-env"
    assert cfg.model == "deepseek-reasoner"


# ---------------- 提示词工具 ----------------

def test_extract_code_from_fenced_reply():
    from quant.llm.prompts import extract_code

    reply = "说明文字\n```python\n" + GOOD_CODE + "\n```\n结尾说明"
    assert "class Strategy" in extract_code(reply)


def test_extract_json_from_fenced_reply():
    from quant.llm.prompts import extract_json

    reply = '```json\n{"name": "x", "indicators": []}\n```'
    assert extract_json(reply)["name"] == "x"


def test_extract_json_raises_on_garbage():
    from quant.llm.prompts import extract_json

    with pytest.raises(ValueError):
        extract_json("完全没有 JSON")


def _bars(n: int = 200) -> pd.DataFrame:
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    # 单调上行 + 轻微波动，保证均线类策略能给出非零仓位
    close = pd.Series(100.0 + np.arange(n, dtype=float) * 0.5
                      + np.sin(np.arange(n) / 3.0), index=idx)
    high = close * 1.01
    low = close * 0.99
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": 10000.0}, index=idx)
