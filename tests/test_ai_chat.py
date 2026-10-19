"""AI 策略对话闭环的单元测试：意图分流、版本追踪、保存落库。

覆盖用户需求「实时查看 + 反复聊天修改 + AI 给想法 + 确认保存」对应的后端行为：
- intent=write/revise/ideas 三种意图各自走对模板
- 会话里每生成一版，_version_of 的版本号 +1（UI「当前版本」靠它）
- revise 时后端自动把上一版代码/配置带给模型（用户不用手动贴）
- ideas 模式不产出代码/配置，也不写 strategy_id
- 保存：一键保存走 SaveIn（无 name 时用模型给的名字），已有 strategy_id 时原地更新
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from quant.storage.db import init_db
from quant.storage.repositories import (LLMRepo, SettingsRepo, StrategiesRepo)
from tests.test_llm import GOOD_CODE

# ---------------- 提示词层 ----------------


def test_build_messages_write_uses_write_template():
    from quant.llm.prompts import build_messages

    msgs = build_messages("code", "均线金叉买入")
    assert msgs[0]["role"] == "system"
    assert "均线金叉买入" in msgs[-1]["content"]
    assert "上一版" not in msgs[-1]["content"]


def test_build_messages_revise_carries_current_code():
    from quant.llm.prompts import build_messages

    msgs = build_messages("code", "加一个 ATR 止损", code=GOOD_CODE, intent="revise")
    last = msgs[-1]["content"]
    assert "加一个 ATR 止损" in last
    assert GOOD_CODE.strip()[:60] in last  # 当前版本代码必须带过去


def test_build_messages_rule_revise_carries_current_config():
    from quant.llm.prompts import build_messages

    cfg = {"name": "规则", "indicators": []}
    msgs = build_messages("rule", "把窗口改成 30", config=cfg, intent="revise")
    last = msgs[-1]["content"]
    assert "把窗口改成 30" in last
    assert json.loads(last[last.find("{"):last.rfind("}") + 1])["name"] == "规则"


def test_build_messages_ideas_switches_system_prompt():
    from quant.llm.prompts import IDEA_SYSTEM_PROMPT, build_messages

    msgs = build_messages("code", "假突破太多怎么办", intent="ideas")
    assert msgs[0]["content"] == IDEA_SYSTEM_PROMPT
    assert "假突破太多怎么办" in msgs[-1]["content"]
    # 思路模式不该带上「请写代码」的骨架
    assert "generate_signals" not in msgs[0]["content"]


def test_parse_ideas_extracts_structured_list():
    from quant.llm.prompts import parse_ideas

    reply = """总体评价：你的思路偏趋势，震荡市容易反复挨打。
```json
[{"title": "ATR 过滤突破", "logic": "突破前一日区间且 ATR 放大时入场", "scene": "高波动趋势市", "risk": "假突破"}]
```"""
    ideas = parse_ideas(reply)
    assert len(ideas) == 1
    assert ideas[0]["title"] == "ATR 过滤突破"
    assert ideas[0]["scene"] == "高波动趋势市"
    assert ideas[0]["logic"].endswith("入场")


def test_parse_ideas_survives_garbage():
    from quant.llm.prompts import parse_ideas

    assert parse_ideas("完全没有 JSON 数组") == []
    assert parse_ideas("```json\n[{\"logic\": \"只有逻辑\"}]\n```")[0]["logic"] == "只有逻辑"


def test_ideas_text_strips_array():
    from quant.llm.prompts import ideas_text

    text = '点评文字 [{"title": "x"}] 后续文字'
    out = ideas_text(text)
    assert "点评文字" in out and "后续文字" in out
    assert '"title"' not in out


# ---------------- 路由层（会话 / 版本 / 保存） ----------------


@pytest.fixture()
def llm_env(tmp_path, monkeypatch):
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
        yield {"db": db, "repo": app_state["llm_repo"]}
    finally:
        app_state.clear()


def _add_version(repo: LLMRepo, sid: int, *, code=GOOD_CODE, valid=True,
                 intent="write"):
    return repo.add_message(sid, "assistant", "已生成", code=code,
                            meta={"intent": intent, "valid": valid, "errors": [],
                                  "config": None})


def test_version_of_counts_every_generated_version(llm_env):
    """反复聊天修改：每生成一版 version +1，UI 才能显示「第 N 版」。"""
    from quant.api.routers import llm as llm_router

    repo = llm_env["repo"]
    s = repo.create_session(title="聊一聊", mode="code")
    sid = s["id"]
    assert llm_router._version_of(repo, sid)["n"] == 0

    repo.add_message(sid, "user", "写个策略")
    _add_version(repo, sid)
    v1 = llm_router._version_of(repo, sid)
    assert v1["n"] == 1 and v1["valid"] is True and v1["code"]

    repo.add_message(sid, "user", "加个止损")
    _add_version(repo, sid, valid=False)
    v2 = llm_router._version_of(repo, sid)
    assert v2["n"] == 2
    assert v2["valid"] is False  # 最新一版没过校验，UI 要提示


def test_version_of_ignores_ideas_and_errors(llm_env):
    from quant.api.routers import llm as llm_router

    repo = llm_env["repo"]
    sid = repo.create_session(title="t", mode="code")["id"]
    _add_version(repo, sid)
    repo.add_message(sid, "assistant", "几个思路",
                     meta={"intent": "ideas", "ideas": [{"title": "x"}]})
    repo.add_message(sid, "assistant", "生成失败：超时",
                     meta={"error": True, "intent": "write"})
    v = llm_router._version_of(repo, sid)
    assert v["n"] == 1  # 思路/失败不计入版本


def test_version_of_returns_latest_rule_config(llm_env):
    from quant.api.routers import llm as llm_router

    repo = llm_env["repo"]
    sid = repo.create_session(title="t", mode="rule")["id"]
    repo.add_message(sid, "assistant", "配置", code="",
                     meta={"intent": "write", "valid": True, "errors": [],
                           "config": {"name": "规则V2", "indicators": []}})
    v = llm_router._version_of(repo, sid)
    assert v["code"] is None and v["config"]["name"] == "规则V2"


def test_chat_revise_reuses_previous_code_from_history(llm_env, monkeypatch):
    """用户不贴代码也能「改这版」：后端从会话历史里取上一版。"""
    from quant.api.routers import llm as llm_router

    repo = llm_env["repo"]
    sid = repo.create_session(title="t", mode="code")["id"]
    _add_version(repo, sid)

    captured: dict = {}

    class _FakeResult:
        content = "说明\n```python\n" + GOOD_CODE + "\n```"
        model = "fake-model"
        usage = {"total_tokens": 1}

    class _FakeClient:
        def chat(self, messages, **kw):
            captured["messages"] = messages
            return _FakeResult()

    monkeypatch.setattr(llm_router, "_client", lambda: _FakeClient())
    out = llm_router.llm_chat(llm_router.ChatIn(
        message="加个 ATR 止损", session_id=sid, mode="code", intent="revise"))

    last = captured["messages"][-1]["content"]
    assert "加个 ATR 止损" in last and GOOD_CODE.strip()[:60] in last
    assert out["session_id"] == sid
    assert out["result"]["ok"] is True
    # 会话里应留下两条新消息（user + assistant），版本号变 2
    assert len(repo.list_messages(sid)) == 3
    assert out["version"]["n"] == 2


def test_chat_ideas_returns_ideas_without_code(llm_env, monkeypatch):
    from quant.api.routers import llm as llm_router

    repo = llm_env["repo"]
    sid = repo.create_session(title="t", mode="code")["id"]

    class _R:
        content = '点评 [{"title": "ATR过滤", "logic": "突破且ATR放大入场"}]'
        model = "m"
        usage = {}

    class _C:
        def chat(self, messages, **kw):
            return _R()

    monkeypatch.setattr(llm_router, "_client", lambda: _C())
    out = llm_router.llm_chat(llm_router.ChatIn(
        message="假突破太多", session_id=sid, mode="code", intent="ideas"))

    assert out["ideas"][0]["title"] == "ATR过滤"
    assert out["result"]["code"] is None and out["result"]["config"] is None
    m = repo.list_messages(sid)[-1]
    assert m["meta"]["intent"] == "ideas" and m["meta"]["ideas"]
    assert m["code"] == ""
    # 思路不算一个策略版本
    assert out["version"]["n"] == 0


def test_chat_error_is_logged_and_raises(llm_env, monkeypatch):
    from fastapi import HTTPException

    from quant.api.routers import llm as llm_router
    from quant.llm.client import LLMError

    repo = llm_env["repo"]
    sid = repo.create_session(title="t", mode="code")["id"]

    class _BoomClient:
        def chat(self, messages, **kw):
            raise LLMError("请求超时", status=408)

    monkeypatch.setattr(llm_router, "_client", lambda: _BoomClient())

    with pytest.raises(HTTPException) as ei:
        llm_router.llm_chat(llm_router.ChatIn(message="写个策略", session_id=sid))
    assert ei.value.status_code == 400
    assert repo.list_messages(sid)[-1]["meta"]["error"] is True


def test_save_without_name_uses_model_metadata(llm_env):
    """一键保存：不传 name 时用源码里的策略名，用户不用自己想名字。"""
    from quant.api.routers import llm as llm_router

    out = llm_router.save_strategy(llm_router.SaveIn(mode="code", code=GOOD_CODE))
    assert out["ok"] is True
    row = out["strategy"]
    assert row["name"] == "均线交叉策略"
    assert row["strategy_type"] == "code"
    assert row["enabled"] == 0  # 默认草稿，用户确认后再启用


def test_save_updates_existing_strategy_in_place(llm_env):
    from quant.api.routers import llm as llm_router

    first = llm_router.save_strategy(llm_router.SaveIn(mode="code", name="V1",
                                                       code=GOOD_CODE))
    sid = first["strategy"]["id"]
    key = first["strategy"]["key"]

    edited = GOOD_CODE.replace('name = "均线交叉策略"', 'name = "均线交叉策略V3"')
    second = llm_router.save_strategy(llm_router.SaveIn(mode="code", name="V3",
                                                        code=edited, strategy_id=sid))
    assert second["strategy"]["id"] == sid
    assert second["strategy"]["key"] == key
    repo = StrategiesRepo(llm_env["db"])
    assert repo.get(key)["source_code"].strip().startswith("class Strategy")
    assert repo.get(key)["name"] == "V3"
    # 原地更新不应产生第二行
    assert len([r for r in repo.list_all() if r["key"] == key]) == 1


def test_save_links_session_to_strategy(llm_env):
    from quant.api.routers import llm as llm_router

    repo = llm_env["repo"]
    sid = repo.create_session(title="t", mode="code")["id"]
    out = llm_router.save_strategy(llm_router.SaveIn(mode="code", code=GOOD_CODE,
                                                     session_id=sid))
    assert repo.get_session(sid)["strategy_id"] == out["strategy"]["id"]


def test_save_rejects_code_failing_smoke(llm_env):
    from tests.test_llm import BAD_COLUMNS_CODE

    from fastapi import HTTPException

    from quant.api.routers import llm as llm_router

    with pytest.raises(HTTPException) as ei:
        llm_router.save_strategy(llm_router.SaveIn(mode="code", code=BAD_COLUMNS_CODE))
    assert ei.value.status_code == 400
    assert "errors" in ei.value.detail


def test_bars_fixture_is_ohlcv():
    from tests.test_llm import _bars

    df = _bars(50)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
