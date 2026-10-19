"""策略创建页「AI 辅助」相关纯逻辑的单元测试。

只测不依赖 Streamlit 运行时的那几个纯函数：它们决定了 AI 生成的配置
能不能和手填配置完全等价地落到表单/JSON 上。
"""
from __future__ import annotations

import json

from web.streamlit.pages import strategy_create_page as page

META = {
    "indicators": [
        {"key": "sma", "label": "简单移动平均线 (SMA)",
         "params": [{"name": "window", "type": "integer", "default": 20,
                     "min": 2, "max": 500}],
         "outputs": ["{as}"]},
        {"key": "macd", "label": "MACD",
         "params": [],
         "outputs": ["{as}", "{as}_signal", "{as}_hist"]},
    ],
    "operators": [{"key": "crosses_above", "label": "上穿"}],
    "refs": [{"key": "close", "label": "收盘价"}],
    "position_modes": [{"key": "fixed_pct", "label": "固定比例仓位"}],
}


def _cfg() -> dict:
    """AI 常见写法：ma5 作左值、ma20 只作右值。"""
    return {
        "name": "均线交叉",
        "description": "",
        "indicators": [
            {"name": "sma", "params": {"window": 5}, "as": "ma5"},
            {"name": "sma", "params": {"window": 20}, "as": "ma20"},
        ],
        "entry_long": {"logic": "all",
                       "conditions": [{"lhs": "ma5", "op": "crosses_above",
                                       "rhs": "ma20"}]},
        "exit_long": {"logic": "all",
                      "conditions": [{"lhs": "ma5", "op": "crosses_below",
                                      "rhs": "ma20"}]},
        "entry_short": None,
        "exit_short": None,
        "position": {"mode": "fixed_pct", "value": 0.8},
        "params": {"allow_short": False, "only_long": True},
    }


def test_edit_rows_restores_indicator_left_side():
    rows = page._edit_rows(META, _cfg(), "entry_long")
    assert len(rows) == 1
    row = rows[0]
    assert row["src"] == "indicator"
    assert row["ind"] == "sma"
    assert row["ind_params"]["window"] == 5
    assert row["_as_override"] == "ma5"
    assert row["op"] == "crosses_above"
    assert row["rhs_ref"] == "ma20"


def test_edit_rows_keeps_price_field_left_side():
    cfg = _cfg()
    cfg["entry_long"] = {"logic": "all",
                         "conditions": [{"lhs": "close", "op": ">", "rhs": 0.0}]}
    rows = page._edit_rows(META, cfg, "entry_long")
    assert rows[0]["src"] == "close"
    assert rows[0]["rhs_type"] == "value"


def test_extra_indicators_covers_right_side_only_indicators():
    cfg = _cfg()
    groups = {
        "buy": page._edit_rows(META, cfg, "entry_long"),
        "sell": page._edit_rows(META, cfg, "exit_long"),
        "short": [],
        "close": [],
    }
    extra = page._extra_indicators(groups, cfg)
    # ma20 只出现在右值里，必须被单独保留，否则会被静默丢掉
    assert [s["as"] for s in extra] == ["ma20"]
    assert extra[0]["params"]["window"] == 20


def test_extra_indicators_empty_when_all_used_or_no_edit():
    cfg = _cfg()
    groups = {"buy": page._edit_rows(META, cfg, "entry_long"),
              "sell": [], "short": [], "close": []}
    assert page._extra_indicators(groups, None) == []
    assert page._extra_indicators(groups, {"indicators": []}) == []


def test_ai_message_without_current_config():
    assert page._ai_message("  写个均线策略  ", {"name": "x"}, False) == "写个均线策略"


def test_ai_message_includes_current_config_json():
    cfg = {"name": "旧策略", "indicators": []}
    msg = page._ai_message("把仓位改成半仓", cfg, True)
    assert msg.startswith("把仓位改成半仓")
    assert "=== 当前配置（请在此基础上修改，保留没提到的部分）===" in msg
    payload = msg.split("===\n", 1)[1]
    assert json.loads(payload) == cfg


def test_ai_message_skips_current_when_flag_off_or_empty():
    cfg = {"name": "旧策略"}
    assert page._ai_message("改一下", cfg, False) == "改一下"
    assert page._ai_message("改一下", {}, True) == "改一下"


def test_ai_examples_are_non_empty_strings():
    assert page.AI_EXAMPLES
    for text in page.AI_EXAMPLES:
        assert isinstance(text, str) and text.strip()


def test_err_text_extracts_errors_from_detail():
    class _Resp:
        status_code = 400
        text = "bad"

        def json(self):
            return {"detail": {"errors": ["未知指标: foo", "条件为空"]}}

    class _Err(Exception):
        response = _Resp()

    assert page._err_text(_Err()) == "未知指标: foo；条件为空"


def test_err_text_falls_back_to_message_and_plain():
    class _Resp:
        status_code = 400
        text = "raw body"

        def json(self):
            return {"detail": {"message": "服务不可用"}}

    class _Err(Exception):
        response = _Resp()

    assert page._err_text(_Err()) == "服务不可用"

    class _Resp2(_Resp):
        def json(self):
            return {"detail": "太短"}

    class _Err2(Exception):
        response = _Resp2()

    assert page._err_text(_Err2()) == "太短"
    assert page._err_text(ValueError("boom")).startswith("ValueError: boom")
