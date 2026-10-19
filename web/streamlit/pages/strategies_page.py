"""策略管理页：启停开关、schema 驱动参数表单、调参自动保存。"""
from __future__ import annotations

import streamlit as st

from web.streamlit.components import (inject_theme, render_params_form,
                                      strategy_label)


def render(api) -> None:
    inject_theme()
    st.header("策略管理")
    st.caption("每个策略：启用开关 + 参数面板（按 schema 自动渲染，修改自动保存）")

    strategies = api.strategies()
    try:
        port_weights = (api.portfolio_get() or {}).get("weights", {})
    except Exception:  # noqa: BLE001
        port_weights = {}
    for s in strategies:
        with st.container(border=True):
            stype = s.get("strategy_type")
            is_rule = stype == "llm_generated"
            is_code = stype == "code"
            editable = is_rule or is_code
            referenced = s["key"] in port_weights
            c1, c2, c3, c4 = st.columns([4.4, 1, 1, 1])
            with c1:
                st.markdown(f"### {strategy_label(s)}  `{s['key']}`")
                st.caption(s.get("description", ""))
                if is_rule:
                    st.caption("类型：声明式规则策略（手动创建或 AI 生成）")
                elif is_code:
                    ok = s.get("code_ok")
                    st.caption(f"类型：Python 代码策略（{'校验通过' if ok else '校验未通过'}"
                               f"，{s.get('code_lines', 0)} 行）")
                if referenced:
                    st.caption("状态：**组合配置中使用中**（删除将自动从组合移除）")
            with c2:
                enabled = st.toggle("已启用" if s.get("enabled") else "已停用",
                                    value=bool(s.get("enabled")), key=f"en_{s['id']}")
                if enabled != bool(s.get("enabled")):
                    api.strategy_toggle(s["id"])
                    st.rerun()
            with c3:
                if editable:
                    if is_rule and st.button("编辑", key=f"ed_{s['id']}"):
                        st.session_state["rule_edit"] = {"config": s.get("config") or {},
                                                         "key": s["key"]}
                        st.session_state["nav_target"] = "策略创建"
                        st.rerun()
                    if is_code and st.button("改代码", key=f"edc_{s['id']}"):
                        try:
                            r = api.llm_code(s["id"])
                            st.session_state["ai_code"] = r.get("code") or ""
                            st.session_state["ai_code_sid"] = s["id"]
                            st.session_state["nav_target"] = "AI 策略"
                            st.rerun()
                        except Exception as e:  # noqa: BLE001
                            st.error(f"读取源码失败：{e}")
            with c4:
                if editable:
                    if st.button("删除", key=f"del_{s['id']}"):
                        st.session_state[f"confirm_del_{s['id']}"] = True
                        st.rerun()
                else:
                    st.button("删除", disabled=True, key=f"del_{s['id']}",
                              help="内置策略不可删除")

            if st.session_state.get(f"confirm_del_{s['id']}"):
                tip = (f"该策略正在组合中被使用，删除后将从组合中移除。" if referenced
                       else "删除后不可恢复。")
                st.warning(f"确认删除策略「{strategy_label(s)}」？{tip}")
                c_a, c_b = st.columns(2)
                if c_a.button("确认删除", key=f"cf_{s['id']}", type="primary"):
                    try:
                        r = api.delete_strategy(s["id"])
                        st.success(f"已删除「{strategy_label(s)}」"
                                   + ("，并已从组合中移除" if r.get("removed_from_portfolio") else ""))
                    except Exception as e:  # noqa: BLE001
                        st.error(f"删除失败：{e}")
                    st.session_state.pop(f"confirm_del_{s['id']}", None)
                    st.rerun()
                if c_b.button("取消", key=f"cn_{s['id']}"):
                    st.session_state.pop(f"confirm_del_{s['id']}", None)
                    st.rerun()

            with st.expander(" 参数调整", expanded=False):
                current = s.get("params") or {}
                if isinstance(current, dict):
                    # 忽略 None 字段（可选参数未设置时表单不渲染，避免误判差异）
                    current_clean = {k: v for k, v in current.items() if v is not None}
                else:
                    current_clean = {}
                values = render_params_form(s["params_schema"], current, prefix=f"p_{s['key']}")
                if values and values != current_clean:
                    api.strategy_update(s["id"], params=values)
                    st.caption(" 参数已自动保存")
            st.caption(f"权重：{s.get('weight', 100):.0f}%　（权重请在「组合配置」页调整）")
