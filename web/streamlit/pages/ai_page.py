"""AI 策略页：对话生成 / 反复聊天修改 / 实时查看当前版本 / 确认保存到策略库。

工作流（四步闭环，全在对话里完成）：
1. 用户在左侧对话框用中文描述思路 → AI 直接写出可回测的策略代码
2. 每一版结果立即出现在右侧「当前版本」面板：版本号、校验状态、冒烟指标、完整代码
3. 不满意就在对话框里说「加个 ATR 止损」（动作切到「改这版」），AI 基于当前版本重写，
   右侧面板同步刷新 —— 可以反复聊很多轮
4. 没有思路时切到「要建议」，AI 会给几个方向不同的策略思路，点「按这个写」直接生成
5. 满意了点「✅ 确认保存到策略库」，一键入库；也可以载入编辑器手动改代码
"""
from __future__ import annotations

import json

import streamlit as st

from web.streamlit.components import (TIMEFRAMES, inject_theme, metrics_row, render_params_form, scroll_to_top, tf_label)

MODE_OPTIONS = {"code": "Python 代码策略（自由表达，可手改）",
                "rule": "声明式规则策略（JSON，表单可调）"}

# 三个动作：这一轮到底想干什么。与 mode（代码/规则）正交。
ACTIONS = ["✨ 新策略", "🔧 改这版", "💡 要建议"]
ACTION_OF = {"✨ 新策略": "write", "🔧 改这版": "revise", "💡 要建议": "ideas"}
ACTION_LABEL = {v: k for k, v in ACTION_OF.items()}

TAB_NAMES = ["AI 对话", "代码编辑", "我的 AI 策略"]
TAB_CHAT, TAB_CODE, TAB_MINE = TAB_NAMES

EXAMPLE_PROMPTS = [
    "20 日均线上穿 60 日均线买入，下穿卖出，单笔满仓，只做多。",
    "RSI 低于 30 且价格在 20 日均线上方时买入，RSI 高于 60 或跌破 20 日均线时卖出。",
    "布林带下轨反弹策略：收盘价从下轨下方回到带内买入，触及上轨卖出，用 ATR 控制仓位。",
    "双均线加 MACD 过滤：快线上穿慢线且 MACD 柱状为正才买入，MACD 转负就卖出。",
]

IDEA_PROMPTS = [
    "我想做趋势跟踪，但总是假突破太多，有什么思路能过滤？",
    "我现在用均线金叉死叉，想加一点波动率或成交量维度的判断，怎么改？",
    "震荡市里均线策略反复挨打，有没有适合低波动行情的思路？",
]


def _err_detail(e: Exception) -> str:
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            data = resp.json()
            d = data.get("detail") if isinstance(data, dict) else None
            if isinstance(d, dict) and d.get("errors"):
                return "；".join(str(x) for x in d["errors"])
            if d:
                return d if isinstance(d, str) else str(d)
        except Exception:  # noqa: BLE001
            return (resp.text or str(e))[:300]
    return str(e)


def _init_state(api) -> None:
    ss = st.session_state
    ss.setdefault("ai_mode", "code")
    ss.setdefault("ai_action", ACTIONS[0])
    ss.setdefault("ai_session_id", None)
    ss.setdefault("ai_code", "")
    ss.setdefault("ai_code_sid", None)      # 已保存的 strategy_id
    ss.setdefault("ai_code_ver", 0)         # 编辑器控件版本号（换代码时强制刷新）
    ss.setdefault("ai_tab", TAB_CHAT)       # 当前标签页（可被程序切换）
    ss.setdefault("ai_msg", None)
    # 模型来源：deepseek（OpenAI 兼容配置）| workbuddy
    ss.setdefault("ai_provider", "deepseek")
    ss.setdefault("ai_model", None)         # workbuddy 时的具体模型 id（None=auto）
    # 当前版本（会话里最近一版）：聊天每生成一版就刷新，右侧面板靠它实时查看
    ss.setdefault("ai_cur_code", "")
    ss.setdefault("ai_cur_config", None)
    ss.setdefault("ai_cur_meta", {})
    ss.setdefault("ai_cur_sid", None)


def _set_code(code: str, sid: int | None = None) -> None:
    """把代码写进「代码编辑」并让编辑器下次渲染时重建。

    st.text_area 带 key 时值缓存在 session_state 里，只改 value= 不会刷新，
    因此用版本号换 key 强制重新初始化。
    """
    ss = st.session_state
    ss["ai_code"] = code or ""
    ss["ai_code_sid"] = sid
    ss["ai_code_ver"] = int(ss.get("ai_code_ver") or 0) + 1


def _toast(msg: str, icon: str = "✅") -> None:
    """跨 rerun 的提示：写进 session_state，下次渲染时弹出。"""
    st.session_state["ai_toast"] = (msg, icon)


def _market_ctx(api) -> dict:
    """给模型的行情上下文：有本地数据就带上标的，帮它定合理的窗口。"""
    try:
        stt = api.data_status()
        syms = [m["symbol"] for m in stt.get("symbols", []) if m.get("rows")]
    except Exception:  # noqa: BLE001
        syms = []
    return {"symbol": syms[0] if syms else "", "timeframe": "1d",
            "bars_count": 0} if syms else {}


def _workbuddy_available(api) -> dict | None:
    """WorkBuddy 是否可用（缓存一次）。返回 status dict 或 None。"""
    ss = st.session_state
    if "ai_wb_status" not in ss:
        try:
            ss["ai_wb_status"] = api.workbuddy_status()
        except Exception:  # noqa: BLE001
            ss["ai_wb_status"] = {"available": False}
    return ss["ai_wb_status"] if ss["ai_wb_status"].get("available") else None


def render(api) -> None:
    inject_theme()
    # 打开本页时主动滚回顶部：Streamlit 会保留上次的滚动位置，而本页下方有
    # 「当前版本 / 手动编辑 / 我的 AI 策略」等长区块，用户从别的页面进来常落在底部。
    # 只在「刚切到本页」这一次 rerun 触发，避免每轮 rerun（发消息、切视图）
    # 都把视图拽回顶部。_page_changed 由 streamlit_app.py 统一维护。
    if st.session_state.pop("_page_changed", False):
        scroll_to_top()
    st.markdown("## AI 策略工坊")
    st.caption("用中文描述交易思路，AI 直接写出可回测的策略；右侧实时查看当前版本，"
               "不满意就在对话框里继续聊，满意了一键保存到策略库。")

    try:
        status = api.llm_status()
    except Exception as e:  # noqa: BLE001
        st.error(f"无法连接后端服务：{_err_detail(e)}")
        return

    _init_state(api)
    ss = st.session_state

    toast = st.session_state.pop("ai_toast", None)
    if toast:
        st.toast(toast[0], icon=toast[1])

    wb = _workbuddy_available(api)
    has_llm = bool(status.get("configured"))
    if not has_llm and not wb:
        st.warning("尚未配置大模型 API Key，且未检测到可用的 WorkBuddy 登录态。"
                   "请到「系统设置 → 策略设置 → 大模型设置」填写 DeepSeek/OpenAI Key，"
                   "或登录 WorkBuddy 桌面端后使用其模型。")
        if st.button("前往大模型设置", type="primary"):
            st.session_state["nav_target"] = "策略设置"
            st.rerun()
        st.divider()
    elif not has_llm:
        st.info("未配置 DeepSeek/OpenAI Key，但检测到 WorkBuddy 登录态，"
                "可在下方把「模型来源」切到 WorkBuddy。")

    if has_llm:
        st.caption(f"默认模型：`{status.get('model')}`　接口：`{status.get('base_url')}`")
    if wb:
        st.caption(f"WorkBuddy：{wb.get('nickname') or '已登录'}　"
                   f"积分 {((wb.get('credits') or {}).get('total') or 0):,}　"
                   f"模型 {(wb.get('models') or []) and len(wb['models'])} 个")

    # 视图切换用 segmented_control 而不是 st.tabs：tabs 会把三个面板都渲染进 DOM，
    # 程序化切换（如「让 AI 改这段代码」要跳到对话页）不可靠。
    # 注意：ai_tab 是控件 key，控件实例化之后就不能再赋值，
    # 所以其它位置只写 ai_tab_next，在这里落地。
    nxt = ss.pop("ai_tab_next", None)
    if nxt in TAB_NAMES:
        ss["ai_tab"] = nxt
    if ss.get("ai_tab") not in TAB_NAMES:
        ss["ai_tab"] = TAB_NAMES[0]
    view = st.segmented_control("视图", list(TAB_NAMES), key="ai_tab",
                                label_visibility="collapsed") or TAB_NAMES[0]

    if view == TAB_CHAT:
        _render_chat(api, status, wb)
    elif view == TAB_CODE:
        _render_editor(api)
    else:
        _render_mine(api)


# ==================== 对话 ====================

def _render_model_source(api, status, wb) -> None:
    """模型来源下拉：DeepSeek/OpenAI 兼容 或 WorkBuddy（可选）。

    WorkBuddy 分支下再选具体模型（auto 或 32 个里的任意一个）。
    """
    ss = st.session_state
    has_llm = bool(status.get("configured"))

    labels = {}
    if has_llm:
        labels["deepseek"] = f"OpenAI 兼容（{status.get('model')}）"
    if wb:
        labels["workbuddy"] = "WorkBuddy（本机登录态）"

    if not labels:
        return
    # 若当前 provider 不可用（如没配 Key 但选了 deepseek），回退
    if ss["ai_provider"] not in labels:
        ss["ai_provider"] = "deepseek" if has_llm else "workbuddy"

    c1, c2 = st.columns([1.4, 1.4])
    ss["ai_provider"] = c1.selectbox("模型来源", list(labels),
                                     format_func=lambda k: labels[k],
                                     key="ai_provider_pick")

    if ss["ai_provider"] == "workbuddy":
        models = wb.get("models") or []
        opts = ["auto"] + [m.get("id") for m in models if m.get("id")]
        cur = ss.get("ai_model") or "auto"
        idx = opts.index(cur) if cur in opts else 0
        ss["ai_model"] = c2.selectbox("WorkBuddy 模型", opts, index=idx,
                                      key="ai_model_pick")
    else:
        ss["ai_model"] = None


def _render_chat(api, status, wb) -> None:
    ss = st.session_state

    # ---- 顶栏：生成模式 + 动作 + 会话 ----
    c1, c2 = st.columns([1.0, 1.0])
    ss["ai_mode"] = c1.radio("生成模式", list(MODE_OPTIONS.keys()),
                             format_func=lambda k: MODE_OPTIONS[k],
                             index=list(MODE_OPTIONS).index(ss["ai_mode"]),
                             horizontal=True, key="ai_mode_radio")

    # 动作 segmented_control 同样带 key：程序里「按这个写」「让 AI 改这版」只能写
    # ai_action_next，在这里落地。
    nxt_act = ss.pop("ai_action_next", None)
    if nxt_act in ACTION_LABEL:
        ss["ai_action"] = nxt_act
    if ss.get("ai_action") not in ACTIONS:
        ss["ai_action"] = ACTIONS[0]
    c2.segmented_control("这一步要做什么", ACTIONS, key="ai_action")
    action = ACTION_OF.get(ss.get("ai_action"), "write")

    # ---- 模型来源选择器 ----
    _render_model_source(api, status, wb)

    c3, c4 = st.columns([2.4, 1.0])
    sessions = []
    try:
        sessions = api.llm_sessions()
    except Exception:  # noqa: BLE001
        pass
    labels = {s["id"]: f"#{s['id']} {s.get('title') or '新对话'}"
                       f"（{s.get('message_count', 0)} 条）" for s in sessions}
    opts = [None] + list(labels)

    # 下拉框带 key 时优先用 session_state 里的值（不会跟随 value/index），
    # 所以「程序里换了会话」必须显式同步一次。
    if ss.pop("ai_sess_sync", False):
        ss["ai_sess_pick"] = (ss["ai_session_id"]
                              if ss["ai_session_id"] in labels else None)
    if ss.get("ai_sess_pick") not in opts:
        ss["ai_sess_pick"] = None

    pick = c3.selectbox("历史对话", opts,
                        format_func=lambda i: "（新对话）" if i is None else labels[i],
                        key="ai_sess_pick")
    if pick != ss["ai_session_id"]:
        ss["ai_session_id"] = pick
        # 换会话时把「当前版本」还原成那个会话里最近一版
        if pick:
            try:
                _sync_current(api.llm_session(pick).get("messages", []),
                              session_id=pick)
            except Exception:  # noqa: BLE001
                pass
        else:
            _clear_current()
        st.rerun()

    if c4.button("新建对话", use_container_width=True, key="ai_new_sess"):
        try:
            s = api.llm_session_new(mode=ss["ai_mode"])
            ss["ai_session_id"] = s.get("id")
            ss["ai_msg"] = None
            ss["ai_sess_sync"] = True
            _clear_current()
        except Exception as e:  # noqa: BLE001
            st.error(_err_detail(e))
        st.rerun()

    # ---- 主体：左对话 | 右当前版本 ----
    messages = []
    if ss["ai_session_id"]:
        try:
            messages = api.llm_session(ss["ai_session_id"]).get("messages", [])
        except Exception as e:  # noqa: BLE001
            st.error(f"读取会话失败：{_err_detail(e)}")
        else:
            # 页面刷新后从会话消息还原「当前版本」（第 N 版）
            _sync_current(messages)

    col_chat, col_ver = st.columns([1.1, 1.0])

    with col_chat:
        # 注意：不要用 st.container(height=...) 固定高度——那会让聊天区变成独立的
        # 滚动容器，鼠标悬停其上时滚轮被它吞掉，且滚到边界会产生「回弹」，
        # 必须先把内层退到顶/底才能带动外层页面滚动。这里只加边框、不设高度，
        # 让内容自然撑开，整页保持单一滚动条。
        with st.container(border=True):
            if not messages:
                st.info("还没有对话。可以直接点下面的示例，或自己描述策略。")
                cols = st.columns(2)
                for i, text in enumerate(EXAMPLE_PROMPTS):
                    if cols[i % 2].button(text, key=f"ai_ex_{i}", use_container_width=True):
                        st.session_state["ai_prefill"] = text
                        st.session_state["ai_action_next"] = "write"
                        st.rerun()
                st.divider()
                st.caption("没有思路？切到「💡 要建议」让 AI 给几个方向：")
                for i, text in enumerate(IDEA_PROMPTS):
                    if st.button(text, key=f"ai_idea_ex_{i}", use_container_width=True):
                        st.session_state["ai_prefill"] = text
                        st.session_state["ai_action_next"] = "ideas"
                        st.rerun()
            for m in messages:
                _render_message(m)

    with col_ver:
        _render_current(api)

    # ---- 输入区 ----
    hint = {
        "write": "描述你的新策略，AI 会直接写出可回测的代码",
        "revise": "告诉 AI 要改什么，它会基于当前版本重写（当前版本会一起发给它）",
        "ideas": "描述你的困惑或方向，AI 只给思路、不写代码",
    }[action]
    st.caption(f"🎯 {hint}")

    # 注意：不要用 st.chat_input —— 它会让整页变成「stAppScrollToBottomContainer」，
    # 一挂载 / 每次 rerun 都自动滚到最底部，上滚还会被强拉回底部（用户反馈的「回弹」+
    # 「进入页面先显示底部」就源于此）。改用 text_input + 发送按钮，整页保持普通滚动条，
    # 从顶部开始、上下自由滚动。
    prefill = st.session_state.pop("ai_prefill", "")
    if prefill:
        st.session_state["ai_chat_text"] = prefill
    c_in1, c_in2 = st.columns([5, 1])
    with c_in1:
        prompt = st.text_input(
            "策略需求",
            key="ai_chat_text",
            placeholder="描述你的策略，例如：5 日均线上穿 20 日均线买入，下穿卖出",
            label_visibility="collapsed",
        )
    with c_in2:
        send = st.button("发送 ▶", type="primary", use_container_width=True, key="ai_chat_send")
    if send and prompt:
        if ss.get("ai_provider") == "deepseek" and not status.get("configured"):
            st.error("请先配置 API Key（或把模型来源切到 WorkBuddy）")
        elif ss.get("ai_provider") == "workbuddy" and not wb:
            st.error("WorkBuddy 当前不可用，请登录 WorkBuddy 桌面端后重试")
        else:
            st.session_state["ai_chat_text"] = ""   # 清空输入框，避免下次渲染带回旧值
            _ask(api, prompt, action=action)


def _render_message(m: dict) -> None:
    role = m.get("role")
    if role == "user":
        with st.chat_message("user"):
            st.markdown(m.get("content") or "")
        return

    meta = m.get("meta") or {}
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(m.get("content") or "（未返回说明）")

        ideas = meta.get("ideas") or []
        if meta.get("intent") == "ideas" or ideas:
            _render_ideas(m, ideas)
            return

        ok = meta.get("valid")
        errors = meta.get("errors") or []
        if meta.get("error"):
            st.error("生成失败")
        elif ok:
            smoke = meta.get("smoke") or {}
            if smoke:
                st.success(f"代码已通过校验与冒烟回测"
                           f"（{smoke.get('signal_count', 0)} 个非零信号，"
                           f"权重 {smoke.get('min_weight', 0):.2f}~{smoke.get('max_weight', 0):.2f}）")
            else:
                st.success("配置已通过校验")
        elif errors:
            st.warning("生成结果未通过校验：\n" + "\n".join(f"- {e}" for e in errors[:8]))

        if m.get("code"):
            with st.expander(f"查看代码（{len(m['code'].splitlines())} 行）", expanded=False):
                st.code(m["code"], language="python")
                if st.button("载入代码编辑器", key=f"ai_load_{m['id']}"):
                    _set_code(m["code"], None)
                    st.session_state["ai_tab_next"] = TAB_CODE
                    _toast("已载入「代码编辑」，可手动修改")
                    st.rerun()
        if meta.get("config") and meta.get("mode") == "rule":
            with st.expander("查看规则配置 JSON"):
                st.json(meta["config"])


def _render_ideas(m: dict, ideas: list[dict]) -> None:
    """思路模式的每条想法一张卡片，带「按这个写」按钮。"""
    if not ideas:
        return
    for i, it in enumerate(ideas):
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {it.get('title') or '未命名思路'}**")
            st.caption(it.get("logic") or "")
            c1, c2 = st.columns([1, 1])
            c1.caption(f"适合：{it.get('scene') or '—'}")
            c2.caption(f"风险：{it.get('risk') or '—'}")
            if st.button("按这个思路写策略", key=f"ai_idea_{m['id']}_{i}",
                         use_container_width=True):
                st.session_state["ai_prefill"] = _idea_prompt(it)
                st.session_state["ai_action_next"] = "write"
                st.rerun()


def _idea_prompt(idea: dict) -> str:
    """把一条思路转成写代码的 prompt（前缀告诉模型别再发散）。"""
    lines = ["请按下面这个思路写出完整策略代码，不要发散、不要给备选方案：", "",
             f"思路名称：{idea.get('title') or ''}",
             f"核心逻辑：{idea.get('logic') or ''}"]
    if idea.get("scene"):
        lines.append(f"适合行情：{idea['scene']}")
    if idea.get("risk"):
        lines.append(f"主要风险与应对：{idea['risk']}")
    return "\n".join(lines)


def _ask(api, prompt: str, action: str = "write") -> None:
    ss = st.session_state
    mode = ss["ai_mode"]
    code = ss.get("ai_cur_code") if action == "revise" and mode == "code" else None
    config = ss.get("ai_cur_config") if action == "revise" and mode == "rule" else None
    provider = ss.get("ai_provider")
    model = ss.get("ai_model") if provider == "workbuddy" else None

    with st.spinner("AI 正在编写策略（通常 10~60 秒）..." if action != "ideas"
                    else "AI 正在想思路（通常 5~30 秒）..."):
        try:
            r = api.llm_chat(prompt, session_id=ss["ai_session_id"], mode=mode,
                             intent=action, code=code, config=config,
                             context=_market_ctx(api), provider=provider, model=model)
        except Exception as e:  # noqa: BLE001
            st.error(f"生成失败：{_err_detail(e)}")
            return

    ss["ai_session_id"] = r.get("session_id")
    ss["ai_sess_sync"] = True
    _apply_result(r)
    st.rerun()


def _apply_result(r: dict) -> None:
    """把一轮对话的结果落到「当前版本」，右侧面板随之刷新（实时查看）。"""
    ss = st.session_state
    result = r.get("result") or {}
    if result.get("code"):
        ss["ai_cur_code"] = result["code"]
        ss["ai_cur_config"] = None
        ss["ai_cur_meta"] = {"mode": ss.get("ai_mode", "code"),
                             "valid": bool(result.get("ok")),
                             "errors": result.get("errors") or [],
                             "smoke": result.get("smoke"),
                             "model": r.get("model"),
                             "explanation": (r.get("message") or {}).get("content") or ""}
    elif result.get("config"):
        ss["ai_cur_config"] = result["config"]
        ss["ai_cur_code"] = ""
        ss["ai_cur_meta"] = {"mode": "rule",
                             "valid": bool(result.get("ok")),
                             "errors": result.get("errors") or [],
                             "smoke": None,
                             "model": r.get("model"),
                             "explanation": (r.get("message") or {}).get("content") or ""}


def _sync_current(messages: list[dict], session_id: int | None = None) -> None:
    """从会话消息还原「当前版本」= 最近一条带代码/配置的 assistant 消息。"""
    ss = st.session_state
    code, config, meta = "", None, {}
    for m in reversed(messages or []):
        if m.get("role") != "assistant":
            continue
        md = m.get("meta") or {}
        if md.get("intent") == "ideas" or md.get("error"):
            continue
        if m.get("code"):
            code = m["code"]
            meta = md
            break
        if md.get("config"):
            config = md["config"]
            meta = md
            break
    ss["ai_cur_code"] = code
    ss["ai_cur_config"] = config
    ss["ai_cur_meta"] = meta
    if session_id is not None:
        try:
            sess = st.session_state.get("_ai_sess_cache") or {}
        except Exception:  # noqa: BLE001
            sess = {}
        ss["ai_cur_sid"] = (sess.get(session_id) if isinstance(sess, dict) else None)


def _clear_current() -> None:
    ss = st.session_state
    ss["ai_cur_code"] = ""
    ss["ai_cur_config"] = None
    ss["ai_cur_meta"] = {}
    ss["ai_cur_sid"] = None


def _version_no(messages: list[dict]) -> int:
    """第几版：数会话里非思路的 assistant 消息带代码/配置的条数。"""
    n = 0
    for m in messages or []:
        if m.get("role") != "assistant":
            continue
        meta = m.get("meta") or {}
        if meta.get("intent") == "ideas" or meta.get("error"):
            continue
        if m.get("code") or meta.get("config"):
            n += 1
    return n


# ==================== 当前版本（实时查看） ====================

def _render_current(api) -> None:
    """右侧面板：当前版本 + 校验状态 + 代码 + 三个动作。

    这就是「实时查看」：每轮对话结束 _apply_result 会更新 session_state，
    本面板随之刷新，不需要手动点「刷新」。
    """
    ss = st.session_state
    st.markdown("### 🧾 当前版本")
    meta = ss.get("ai_cur_meta") or {}
    code = ss.get("ai_cur_code") or ""
    config = ss.get("ai_cur_config")
    mode = meta.get("mode") or ss.get("ai_mode") or "code"

    if not code and not config:
        st.info("还没有生成任何版本。在左侧描述你的思路，或切到「💡 要建议」让 AI 给几个方向。")
        return

    valid = bool(meta.get("valid"))
    if valid:
        st.success("校验通过，可以保存或继续改")
    else:
        st.error("这一版还没通过校验，先让它改或自己动手")

    smoke = meta.get("smoke") or {}
    if smoke:
        st.caption(f"冒烟回测：{smoke.get('signal_count', 0)} 个非零信号，"
                   f"权重 {smoke.get('min_weight', 0):.2f}~{smoke.get('max_weight', 0):.2f}")
    errors = meta.get("errors") or []
    if errors:
        with st.expander("查看校验错误", expanded=True):
            st.markdown("\n".join(f"- {e}" for e in errors[:10]))
    if meta.get("model"):
        st.caption(f"模型 `{meta['model']}`")

    if code:
        st.caption(f"Python 代码 · {len(code.splitlines())} 行")
        with st.expander("查看当前版本代码", expanded=True):
            st.code(code, language="python")
    else:
        st.caption("声明式规则配置（JSON）")
        with st.expander("查看当前配置", expanded=True):
            st.json(config or {})

    st.divider()
    b1, b2 = st.columns([1, 1])
    if b1.button("✍️ 载入编辑器手改", use_container_width=True, key="ai_cur_edit"):
        if code:
            _set_code(code, ss.get("ai_cur_sid"))
        else:
            st.session_state["rule_edit"] = {"config": config or {}, "key": None}
            st.session_state["nav_target"] = "策略创建"
        st.session_state["ai_tab_next"] = TAB_CODE if code else None
        _toast("已载入编辑器，可手动修改")
        st.rerun()
    if b2.button("🔧 让 AI 改这版", use_container_width=True, key="ai_cur_revise"):
        st.session_state["ai_action_next"] = "revise"
        _toast("已切到「改这版」，在下面输入框说要改什么", icon="✍️")
        st.rerun()

    st.divider()
    st.markdown("**确认保存到策略库**")
    if st.button("✅ 一键保存（用 AI 起的名字，存为草稿）",
                 type="primary", use_container_width=True, key="ai_cur_save"):
        _quick_save(api)

    with st.expander("保存选项（重命名 / 描述 / 立即启用）"):
        _save_options(api, mode, code, config)


def _quick_save(api) -> None:
    """一键保存：用模型给出的名称/描述直接入库，保存为草稿（未启用）。"""
    ss = st.session_state
    mode = (ss.get("ai_cur_meta") or {}).get("mode") or ss.get("ai_mode") or "code"
    body: dict = {"mode": mode, "enabled": False,
                  "session_id": ss.get("ai_session_id")}
    if mode == "rule":
        config = ss.get("ai_cur_config")
        if not config:
            st.warning("当前没有可保存的配置")
            return
        body["config"] = config
    else:
        code = ss.get("ai_cur_code") or ""
        if not code.strip():
            st.warning("当前没有可保存的代码")
            return
        body["code"] = code
        info = ((ss.get("ai_cur_meta") or {}).get("smoke") or {}).get("inspect") or {}
        body["name"] = info.get("name") or "AI 策略"
        body["description"] = info.get("description") or ""
    sid = ss.get("ai_cur_sid")
    if sid:
        body["strategy_id"] = sid

    with st.spinner("保存并校验中..."):
        try:
            r = api.llm_save(**body)
        except Exception as e:  # noqa: BLE001
            st.error(f"保存失败：{_err_detail(e)}")
            return
    strat = r.get("strategy") or {}
    ss["ai_cur_sid"] = strat.get("id")
    ss["ai_code_sid"] = strat.get("id")
    _toast(f"已保存到策略库：{strat.get('name')}（key={strat.get('key')}）")
    st.rerun()


def _save_options(api, mode: str, code: str, config: dict | None) -> None:
    """给想改名字/描述/直接启用的用户用的表单保存。"""
    ss = st.session_state
    prev = None
    if ss.get("ai_cur_sid"):
        try:
            prev = api.llm_code(ss["ai_cur_sid"])
        except Exception:  # noqa: BLE001
            prev = None

    with st.form("ai_save_opts"):
        name = st.text_input("策略名称", value=(prev or {}).get("name") or "")
        desc = st.text_input("一句话描述", value=(prev or {}).get("description") or "")
        enabled = st.checkbox("保存后立即启用", value=False)
        ok = st.form_submit_button("确认保存", type="primary")
    if not ok:
        return

    body = {"mode": mode, "name": name, "description": desc, "enabled": enabled,
            "session_id": ss.get("ai_session_id")}
    if mode == "rule":
        if not config:
            st.warning("当前没有可保存的配置")
            return
        body["config"] = config
    else:
        body["code"] = code
    if ss.get("ai_cur_sid"):
        body["strategy_id"] = ss["ai_cur_sid"]

    with st.spinner("保存并校验中..."):
        try:
            r = api.llm_save(**body)
        except Exception as e:  # noqa: BLE001
            st.error(f"保存失败：{_err_detail(e)}")
            return
    strat = r.get("strategy") or {}
    ss["ai_cur_sid"] = strat.get("id")
    ss["ai_code_sid"] = strat.get("id")
    _toast(f"已保存：{strat.get('name')}（key={strat.get('key')}，#{strat.get('id')}）")
    st.rerun()


# ==================== 代码编辑 ====================

def _render_editor(api) -> None:
    ss = st.session_state
    st.markdown("### 手动编辑策略代码")
    st.caption("AI 生成的代码可以直接改。保存时后端会重新做静态安全检查 + 冒烟回测，"
               "不通过不会写入数据库。")

    if not ss["ai_code"]:
        st.info("还没有代码。先在「AI 对话」生成一版，或从「我的 AI 策略」里载入已有策略。")
        _load_existing(api)
        return

    sid_now = ss["ai_code_sid"]
    title = "（新建策略）" if not sid_now else f"（编辑已有策略 #{sid_now}）"
    st.caption(f"当前编辑：{title}")

    code = st.text_area(
        "策略源码（Python）", value=ss["ai_code"], height=460,
        key=f"ai_code_area_{ss['ai_code_ver']}",
        help="必须定义一个含 generate_signals 的类，返回 "
             "DataFrame[target_weight, reason, confidence]")

    c1, c2, c3 = st.columns([1, 1, 1])
    if c1.button("校验代码", key="ai_val"):
        with st.spinner("校验中..."):
            try:
                r = api.llm_validate(mode="code", code=code)
                info = r.get("inspect") or {}
                if r.get("ok"):
                    st.success(f"校验通过（类 {info.get('class_name')}，"
                               f"非零信号 {r.get('smoke', {}).get('signal_count', 0)} 个）")
                    if info.get("params_schema", {}).get("properties"):
                        st.caption("可调参数：" + "、".join(
                            f"`{k}`" for k in info["params_schema"]["properties"]))
                else:
                    st.error("校验未通过：\n" + "\n".join(f"- {e}" for e in r.get("errors", [])))
            except Exception as e:  # noqa: BLE001
                st.error(f"校验失败：{_err_detail(e)}")

    if c2.button("保存为策略", type="primary", key="ai_save"):
        ss["ai_code"] = code
        _save_dialog(api, mode="code", code=code)

    if c3.button("让 AI 改这段代码", key="ai_revise_btn"):
        # 切到对话页并把动作设成「改这版」，把编辑器里的代码作为当前版本发过去
        ss["ai_code"] = code
        ss["ai_cur_code"] = code
        ss["ai_cur_meta"] = {"mode": "code", "valid": True, "errors": [], "smoke": None}
        ss["ai_action_next"] = "revise"
        ss["ai_tab_next"] = TAB_CHAT
        st.rerun()

    st.divider()
    _load_existing(api)


def _load_existing(api) -> None:
    """从已保存的代码策略载入源码到编辑器。"""
    try:
        strategies = [s for s in api.strategies()
                      if s.get("strategy_type") == "code" and s.get("has_code")]
    except Exception:  # noqa: BLE001
        return
    if not strategies:
        return
    labels = {s["id"]: f"#{s['id']} {s.get('name')}（{s.get('code_lines')} 行）"
                       for s in strategies}
    c1, c2 = st.columns([3, 1])
    pick = c1.selectbox("载入已保存的代码策略", list(labels),
                        format_func=lambda i: labels[i], key="ai_load_pick")
    if c2.button("载入", key="ai_load_btn"):
        try:
            r = api.llm_code(pick)
            _set_code(r.get("code") or "", pick)
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(_err_detail(e))


def _save_dialog(api, mode: str, code: str = "", config: dict | None = None) -> None:
    ss = st.session_state
    prev_sid = ss.get("ai_code_sid")
    prev = None
    if prev_sid:
        try:
            prev = api.llm_code(prev_sid)
        except Exception:  # noqa: BLE001
            prev = None

    with st.form("ai_save_form"):
        name = st.text_input("策略名称", value=(prev or {}).get("name") or "")
        desc = st.text_input("一句话描述", value=(prev or {}).get("description") or "")
        enabled = st.checkbox("保存后立即启用", value=False)
        ok = st.form_submit_button("确认保存", type="primary")
    if not ok:
        return

    body = {"mode": mode, "name": name, "description": desc, "enabled": enabled,
            "session_id": ss.get("ai_session_id")}
    if mode == "code":
        body["code"] = code
        if prev_sid:
            body["strategy_id"] = prev_sid
    else:
        body["config"] = config
        if prev_sid:
            body["strategy_id"] = prev_sid
    with st.spinner("保存并校验中..."):
        try:
            r = api.llm_save(**body)
        except Exception as e:  # noqa: BLE001
            st.error(f"保存失败：{_err_detail(e)}")
            return
    strat = r.get("strategy") or {}
    ss["ai_code_sid"] = strat.get("id")
    ss["ai_cur_sid"] = strat.get("id")
    _toast(f"已保存：{strat.get('name')}（key={strat.get('key')}，#{strat.get('id')}）")
    st.rerun()


# ==================== 我的 AI 策略 ====================

def _render_mine(api) -> None:
    st.markdown("### 我的 AI 策略")
    try:
        strategies = [s for s in api.strategies()
                      if s.get("strategy_type") in ("code", "llm_generated")]
    except Exception as e:  # noqa: BLE001
        st.error(f"读取失败：{_err_detail(e)}")
        return

    if not strategies:
        st.info("还没有 AI 生成或手写的策略。去「AI 对话」生成一个吧。")
        return

    for s in strategies:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([4.4, 1.2, 1.2, 1.2])
            kind = "AI 代码策略" if s["strategy_type"] == "code" else "声明式规则策略"
            c1.markdown(f"**{s.get('name') or s['key']}**　`{s['key']}`")
            c1.caption(f"{kind}　{s.get('description') or '（无描述）'}")
            if s["strategy_type"] == "code":
                if s.get("code_ok"):
                    c1.caption(f"代码 {s.get('code_lines')} 行，校验通过")
                else:
                    c1.caption("代码校验未通过：" + "；".join(s.get("code_errors", [])[:3]))

            on = c2.toggle("启用", value=bool(s.get("enabled")), key=f"ai_en_{s['id']}")
            if on != bool(s.get("enabled")):
                api.strategy_toggle(s["id"])
                st.rerun()

            if s["strategy_type"] == "code":
                if c3.button("编辑代码", key=f"ai_edit_{s['id']}", use_container_width=True):
                    try:
                        r = api.llm_code(s["id"])
                        _set_code(r.get("code") or "", s["id"])
                        st.session_state["ai_tab_next"] = TAB_CODE
                        _toast("已载入「代码编辑」")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(_err_detail(e))
            else:
                if c3.button("表单编辑", key=f"ai_form_{s['id']}", use_container_width=True):
                    st.session_state["rule_edit"] = {"config": s.get("config") or {},
                                                     "key": s["key"]}
                    st.session_state["nav_target"] = "策略创建"
                    st.rerun()

            if c4.button("删除", key=f"ai_del_{s['id']}", use_container_width=True):
                try:
                    api.delete_strategy(s["id"])
                    if st.session_state.get("ai_code_sid") == s["id"]:
                        _set_code("", None)
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(_err_detail(e))

            with st.expander("参数与回测", expanded=False):
                cur = s.get("params") or {}
                vals = render_params_form(s.get("params_schema") or {}, cur,
                                          prefix=f"aip_{s['key']}")
                if vals and vals != cur:
                    if st.button("保存参数", key=f"ai_psave_{s['id']}"):
                        try:
                            api.strategy_update(s["id"], params=vals)
                            st.success("参数已保存")
                            st.rerun()
                        except Exception as e:  # noqa: BLE001
                            st.error(_err_detail(e))
                _quick_backtest(api, s)


def _quick_backtest(api, s: dict) -> None:
    try:
        stt = api.data_status()
        symbols = [m["symbol"] for m in stt.get("symbols", []) if m.get("rows")]
    except Exception:  # noqa: BLE001
        symbols = []
    if not symbols:
        st.info("尚无本地行情数据，请先在「数据管理」页添加标的")
        return
    c1, c2, c3 = st.columns([2, 1.4, 1.2])
    sym = c1.selectbox("标的", symbols, key=f"ai_bt_sym_{s['id']}")
    tf = c2.selectbox("周期", TIMEFRAMES, index=TIMEFRAMES.index("1d"),
                      format_func=tf_label, key=f"ai_bt_tf_{s['id']}")
    if c3.button("运行回测", type="primary", key=f"ai_bt_{s['id']}", use_container_width=True):
        with st.spinner("回测中..."):
            try:
                r = api.backtest_combo({"symbols": [sym], "timeframe": tf,
                                        "strategy_ids": [s["key"]],
                                        "engine": "event_driven"})
            except Exception as e:  # noqa: BLE001
                st.error(f"回测失败：{_err_detail(e)}")
                return
        m = r.get("metrics", {})
        metrics_row(m)
        st.caption(f"交易 {m.get('trade_count', 0)} 笔，总收益 "
                   f"{m.get('total_return', 0):.2%}，最大回撤 "
                   f"{m.get('max_drawdown', 0):.2%}，夏普 {m.get('sharpe', 0):.2f}")
        if m.get("trade_count", 0) == 0:
            st.warning("没有产生交易：条件可能太严，或方向反了。可以回到「AI 对话」让它调整。")
