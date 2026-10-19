"""策略设置页：大模型（LLM）设置 + WorkBuddy 连接，服务「AI 策略」页的自然语言生成。"""
from __future__ import annotations

import streamlit as st

from web.streamlit.components import inject_theme

LLM_MODEL_PRESETS = [
    "deepseek-chat", "deepseek-reasoner",
    "gpt-4o-mini", "gpt-4o", "qwen-plus", "glm-4-plus", "moonshot-v1-8k",
]

# API 协议选项（与后端 quant/llm/client.py PROTOCOLS 对应）
LLM_PROTOCOL_OPTIONS = {
    "openai": "OpenAI 兼容（DeepSeek/Qwen/Moonshot/GLM/OpenAI）",
    "anthropic": "Anthropic（Claude）",
    "gemini": "Google Gemini",
}

# 每种协议的默认接口地址 / 模型 / Key 提示（用户切换协议时给出引导）
LLM_PROTOCOL_HINTS = {
    "openai": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "key_help": "sk- 开头密钥；DeepSeek/OpenAI/Qwen/Moonshot/GLM 等兼容接口",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "model": "claude-3-5-sonnet-20241022",
        "key_help": "sk-ant- 开头密钥；接口地址无需加 /v1/messages",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com",
        "model": "gemini-1.5-pro",
        "key_help": "AIza 开头的 Google API Key",
    },
}


def render(api) -> None:
    inject_theme()
    st.markdown("## 策略设置")
    st.caption("配置「AI 策略」页用到的模型来源：DeepSeek/OpenAI 兼容接口，或复用本机 WorkBuddy 登录态。")

    _render_llm_settings(api)
    st.divider()
    _render_workbuddy_settings(api)


def _render_llm_settings(api) -> None:
    """大模型设置：Key / 接口地址 / 模型 / 采样参数，含连通性测试。"""
    st.markdown("### 大模型设置")
    st.caption("用于「AI 策略」页的自然语言生成策略。Key 只保存在本机数据库，不会外传。")

    try:
        cur = api.settings_llm()
    except Exception as e:  # noqa: BLE001
        st.warning(f"读取大模型设置失败：{e}")
        return

    configured = bool((cur.get("api_key") or "").strip())
    if configured:
        st.success(f"已配置 API Key（{cur.get('model')}）")
    else:
        st.info("尚未配置 API Key。可以在这里填写，或在项目根目录的 .env 中设置 LLM_API_KEY。")

    c1, c2 = st.columns([2, 1])
    api_key = c1.text_input("API Key", value=cur.get("api_key") or "", type="password",
                            key="llm_key",
                            help="DeepSeek 平台的 sk- 开头密钥；也可用 .env 的 LLM_API_KEY")

    # API 协议下拉：切换后联动默认接口地址 / 模型提示
    protocol = cur.get("api_protocol") or "openai"
    if protocol not in LLM_PROTOCOL_OPTIONS:
        protocol = "openai"
    proto_pick = c2.selectbox("API 协议", list(LLM_PROTOCOL_OPTIONS.keys()),
                              index=list(LLM_PROTOCOL_OPTIONS.keys()).index(protocol),
                              format_func=lambda k: LLM_PROTOCOL_OPTIONS[k],
                              key="llm_protocol")
    hint = LLM_PROTOCOL_HINTS[proto_pick]
    st.caption(f"协议：{LLM_PROTOCOL_OPTIONS[proto_pick]} · Key 提示：{hint['key_help']}")

    c1, c2 = st.columns([2, 1])
    base_url = c1.text_input("接口地址", value=cur.get("base_url") or hint["base_url"],
                             key="llm_base",
                             help="兼容所选协议的服务地址；切换协议后会给出对应的默认地址")

    c1, c2, c3 = st.columns([2, 1, 1])
    model = cur.get("model") or hint["model"]

    # 探测可用模型：点击后拉取接口地址的模型列表，注入下拉（仅 OpenAI 兼容协议支持）
    probed = st.session_state.get("llm_probed_models")
    if probed is None:
        probed = []
    opts = LLM_MODEL_PRESETS + ([m["id"] for m in probed
                                 if m.get("id") not in LLM_MODEL_PRESETS]) + ["自定义..."]
    idx = opts.index(model) if model in opts else len(opts) - 1
    pick = c1.selectbox("模型", opts, index=idx, key="llm_model_pick")
    if pick == "自定义...":
        model = c1.text_input("自定义模型名", value=model, key="llm_model_custom")
    else:
        model = pick
    max_retries = c2.number_input("失败重试次数", 0, 10, int(cur.get("max_retries") or 3),
                                  key="llm_retry")
    timeout = c3.number_input("超时（秒）", 5, 600, int(cur.get("timeout_seconds") or 60),
                              key="llm_timeout")

    # 探测可用模型按钮（仅 OpenAI 兼容协议有效；其它协议返回空）
    if proto_pick == "openai" and st.button(" 🔍 探测可用模型", key="llm_probe_btn"):
        with st.spinner("正在从接口地址拉取模型列表..."):
            try:
                r = api.llm_models()
                probed = r.get("models") or []
                st.session_state["llm_probed_models"] = probed
                if probed:
                    st.success(f"探测到 {len(probed)} 个模型，已加入上方模型下拉")
                else:
                    st.warning("该接口地址未返回模型列表（可能不支持 GET /models）")
            except Exception as e:  # noqa: BLE001
                st.error(f"探测失败：{_err_detail(e)}")
    if probed:
        st.caption(f"已探测模型：{', '.join(m['id'] for m in probed[:12])}"
                   + ("…" if len(probed) > 12 else ""))

    temperature = st.slider("生成温度（越低越稳定，代码生成建议 0.0~0.3）", 0.0, 1.5,
                            float(cur.get("temperature") or 0.2), 0.05, key="llm_temp")

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button(" 保存大模型设置", type="primary", key="llm_save_btn"):
        try:
            api.settings_llm_put({
                "api_key": (api_key or "").strip(),
                "base_url": (base_url or "").strip() or hint["base_url"],
                "api_protocol": proto_pick,
                "model": model, "max_retries": int(max_retries),
                "timeout_seconds": float(timeout), "temperature": float(temperature),
            })
            st.success("已保存")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"保存失败：{_err_detail(e)}")

    if b2.button(" 测试连接", key="llm_ping_btn"):
        with st.spinner("正在调用模型..."):
            try:
                r = api.llm_ping()
                st.success(f"连接正常（{r.get('model')}，{r.get('latency_ms')} ms）")
                if r.get("reply"):
                    st.caption(f"模型回复：{r['reply']}")
            except Exception as e:  # noqa: BLE001
                st.error(f"连接失败：{_err_detail(e)}")


def _err_detail(e: Exception) -> str:
    """把 httpx 的响应体里的 detail 抽出来，给用户看得懂的报错。"""
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            data = resp.json()
            if isinstance(data, dict) and data.get("detail"):
                d = data["detail"]
                return d if isinstance(d, str) else str(d)
        except Exception:  # noqa: BLE001
            return (resp.text or str(e))[:300]
    return str(e)


def _reasoning_label(m: dict) -> str:
    """把 WorkBuddy 模型的推理字段渲染成可读的「推理等级」文本。"""
    if not m.get("supports_reasoning"):
        return "无推理"
    levels = m.get("reasoning_levels")
    effort = m.get("reasoning_effort")
    if levels:
        return f"{effort or '?'}（可选：{'/'.join(levels)}）"
    return effort or "推理"


def _render_workbuddy_settings(api) -> None:
    """WorkBuddy 连接卡片：安装位置配置 + 账号 / 积分 / 模型列表（可选模型来源）。"""
    st.markdown("### WorkBuddy 连接")
    st.caption("复用本机已登录的 WorkBuddy 桌面端登录态，可把 WorkBuddy 的模型与积分作为「AI 策略」页的可选模型来源。")

    _render_workbuddy_path(api)

    if st.button(" 刷新状态", key="wb_refresh_btn"):
        st.session_state.pop("wb_status", None)

    status = st.session_state.get("wb_status")
    if status is None:
        with st.spinner("读取 WorkBuddy 登录态..."):
            try:
                status = api.workbuddy_status()
                st.session_state["wb_status"] = status
            except Exception as e:  # noqa: BLE001
                st.session_state["wb_status"] = {"available": False, "error": _err_detail(e)}
                status = st.session_state["wb_status"]

    if not status.get("available"):
        st.warning(f"WorkBuddy 当前不可用：{status.get('error', '未知原因')}")
        st.caption("请确认已安装并登录 WorkBuddy 桌面端。该来源为可选，未启用不影响 DeepSeek/OpenAI 兼容配置。")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("账号", status.get("nickname") or "已登录")
    c2.metric("区域", "国际版" if status.get("region") == "global" else "国内版")
    credits = status.get("credits") or {}
    c3.metric("积分总额", f"{credits.get('total', 0):,}")

    models = status.get("models") or []
    # 实时更新模型按钮：单独重取模型列表（含推理等级），不动账号/积分缓存
    if st.button(" 🔄 更新模型", key="wb_models_refresh_btn"):
        with st.spinner("更新 WorkBuddy 模型列表..."):
            try:
                r = api.workbuddy_models()
                fresh = r.get("models") or []
                if status is not None:
                    status["models"] = fresh
                    st.session_state["wb_status"] = status
                if fresh:
                    st.success(f"已更新，共 {len(fresh)} 个模型")
                else:
                    st.warning("未返回模型列表")
            except Exception as e:  # noqa: BLE001
                st.error(f"更新失败：{_err_detail(e)}")
    if models:
        with st.expander(f"可用模型（{len(models)} 个）", expanded=False):
            rows = [{"模型": m.get("name"), "ID": m.get("id"),
                     "推理等级": _reasoning_label(m),
                     "积分倍率": (m.get("credits") or "—"),
                     "默认": "✓" if m.get("is_default") else ""} for m in models]
            st.dataframe(rows, use_container_width=True, hide_index=True)

    accounts = (credits.get("accounts") or [])[:5]
    if accounts:
        st.caption("积分包（前 5 个）：" + "；".join(
            f"{a.get('name')} {a.get('remain')}/{a.get('size')}" for a in accounts))


def _render_workbuddy_path(api) -> None:
    """WorkBuddy 安装位置配置：用户可手动填写，系统校准。"""
    try:
        cfg = api.workbuddy_config()
    except Exception:  # noqa: BLE001
        cfg = {"custom_exe_path": "", "exe_path": ""}

    custom = cfg.get("custom_exe_path") or ""
    effective = cfg.get("exe_path") or ""

    st.markdown("**安装位置**")
    st.caption("默认自动探测 C 盘 WorkBuddy 官方安装路径；若你把它装到了别处，请手动填写 `WorkBuddy.exe` 的完整路径并校准。")

    col_path, col_btn = st.columns([4, 1])
    with col_path:
        new_path = st.text_input(
            "WorkBuddy.exe 完整路径（留空 = 自动探测）",
            value=custom,
            key="wb_exe_path_input",
            placeholder=r"例如：C:\Users\你的用户名\AppData\Local\Programs\WorkBuddy\WorkBuddy.exe",
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button(" 保存并校准", key="wb_calibrate_btn"):
            with st.spinner("校准中..."):
                try:
                    result = api.workbuddy_calibrate(new_path)
                    if result.get("ok"):
                        st.success(result.get("message", "校准成功"))
                        st.session_state.pop("wb_status", None)  # 让状态卡片重取
                    else:
                        st.error(result.get("message", "校准失败"))
                except Exception as e:  # noqa: BLE001
                    st.error(f"校准失败：{_err_detail(e)}")

    if effective:
        st.caption(f"当前生效：{effective}" + ("（自定义）" if custom else "（自动探测）"))
