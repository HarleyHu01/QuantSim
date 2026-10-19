"""AI 策略页的纯逻辑冒烟检查（不启动 Streamlit）。

只验证模块能导入、引用的符号都存在、关键常量齐全 —— 这类错误
（拼错方法名、漏 import）在真实跑 Streamlit 前就能抓到。
"""
from __future__ import annotations

import ast
import pathlib

PAGE = pathlib.Path(__file__).resolve().parents[1] / "web" / "streamlit" / "pages" / "ai_page.py"
API = pathlib.Path(__file__).resolve().parents[1] / "web" / "streamlit" / "api.py"
API_SRC = API.read_text(encoding="utf-8")


def test_ai_page_parses():
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.FunctionDef) and n.name == "render" for n in tree.body)


def test_ai_page_calls_existing_api_client_methods():
    """页面里调的每个 api.xxx 都必须在 ApiClient 里存在（拼错立刻暴露）。"""
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            val = node.func.value
            if isinstance(val, ast.Name) and val.id in ("api", "self"):
                called.add(node.func.attr)
    missing = sorted(m for m in called if f"def {m}(" not in API_SRC)
    assert not missing, f"ApiClient 缺少方法: {missing}"
    # 关键方法确实被用到（防止测试空转）
    assert {"llm_status", "llm_chat", "llm_sessions", "llm_session", "llm_save",
            "llm_validate", "llm_code", "strategies", "strategy_toggle",
            "backtest_combo"} <= called


def test_ai_page_defines_chat_actions_and_tabs():
    src = PAGE.read_text(encoding="utf-8")
    assert 'ACTIONS = ["✨ 新策略", "🔧 改这版", "💡 要建议"]' in src
    assert 'TAB_NAMES = ["AI 对话", "代码编辑", "我的 AI 策略"]' in src
    # 三种意图都要有对应的请求参数
    for intent in ("write", "revise", "ideas"):
        assert f'"{intent}"' in src


def test_ai_page_has_live_version_panel():
    """「实时查看」面板：_render_current 必须读当前版本并给出保存入口。"""
    src = PAGE.read_text(encoding="utf-8")
    assert "def _render_current(api)" in src
    assert "def _apply_result(r: dict)" in src
    assert "def _sync_current(" in src
    assert "确认保存到策略库" in src
    assert "def _quick_save(api)" in src
