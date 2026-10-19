"""数据设置页：数据源代理 + 存储与路径。

这两块都只影响「数据怎么下载、下到哪」，与交易/回测参数无关，因此从
「交易设置」里拆出来独立成页，位置排在交易设置之前。

页面内容：
1. 数据源代理 —— 逐渠道选择 跟随系统 / 强制直连 / 强制走代理
2. 存储与路径 —— 显示运行时真实路径，允许改数据目录（写 config/user.yaml）
"""
from __future__ import annotations

import streamlit as st

from web.streamlit.components import (PROXY_MODE_HINTS, PROXY_MODE_LABELS,
                                      inject_theme, to_dataframe)


def render(api) -> None:
    inject_theme()
    st.markdown("## 数据设置")
    st.caption("数据源代理与存储路径。两者都只影响数据下载，不影响交易与回测参数。")

    _render_proxy(api)
    st.divider()
    _render_storage(api)


def _render_proxy(api) -> None:
    """数据源代理：逐渠道选择 跟随系统 / 强制直连 / 强制走代理。"""
    st.markdown("### 数据源代理")
    st.caption(
        "国内站点通常直连更快，国外站点通常必须走代理——这里逐渠道独立设置。"
        "**修改后立即生效，无需重启。**"
    )
    try:
        data = api.proxy_get()
    except Exception as e:  # noqa: BLE001
        st.error(f"读取代理设置失败: {e}")
        return
    modes = data.get("modes", {})
    available = data.get("available_modes", ["auto", "direct", "proxy"])

    new_modes: dict[str, str] = {}
    c1, c2 = st.columns(2)
    for col, names in ((c1, list(modes.keys())[::2]), (c2, list(modes.keys())[1::2])):
        with col:
            for name in names:
                cur = modes[name]
                idx = available.index(cur) if cur in available else 0
                choice = st.selectbox(
                    name, available, index=idx, key=f"px_{name}",
                    format_func=lambda m, _l=PROXY_MODE_LABELS: _l.get(m, m))
                new_modes[name] = choice
                st.caption(PROXY_MODE_HINTS.get(choice, ""))

    if st.button("保存代理设置", type="primary", key="px_save"):
        try:
            api.proxy_put(new_modes)
            st.success("已保存并立即生效（下次下载按新代理策略走）")
        except Exception as e:  # noqa: BLE001
            st.error(f"保存失败: {e}")


def _fmt_bytes(n: int) -> str:
    v = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.1f} {unit}" if unit != "B" else f"{int(v)} B"
        v /= 1024.0
    return f"{v:.1f} TB"


def _render_storage(api) -> None:
    """存储与路径：显示运行时真实路径，允许改数据目录（写 config/user.yaml）。

    这是「让软件可分发」的关键一环——数据目录、数据库、行情缓存都不写死在代码里。
    """
    st.markdown("### 存储与路径")
    st.caption(
        "以下路径都由程序按「软件所在目录」自动推导，**不写死任何一台电脑的绝对路径**。"
        "把整个软件目录拷到别的机器即可直接使用。"
    )
    try:
        info = api.paths_get()
    except Exception as e:  # noqa: BLE001
        st.error(f"读取路径信息失败: {e}")
        return

    rows = [
        {"项目": "软件根目录", "路径": info.get("root", ""), "状态": "", "占用": ""},
        {"项目": "Python 解释器", "路径": info.get("python", ""), "状态": "", "占用": ""},
    ]
    for label, key in (("数据目录", "data_dir"), ("数据库", "db_path"), ("行情缓存", "market_dir")):
        p = info.get(key) or {}
        state = "✓ 存在" if p.get("exists") else "✗ 不存在"
        if p.get("exists") and not p.get("writable"):
            state += "（只读）"
        rows.append({
            "项目": label,
            "路径": p.get("path", ""),
            "状态": state,
            "占用": _fmt_bytes(p.get("size_bytes", 0)) if p.get("exists") else "",
        })
    st.dataframe(to_dataframe(rows), width="stretch", hide_index=True)

    source = info.get("data_dir_source", "")
    st.caption(f"数据目录来源：**{source}**　｜　覆盖配置文件：`{info.get('user_config', '')}`"
               + ("（已存在）" if info.get("user_config_exists") else "（未创建）"))

    with st.form("storage_form"):
        st.markdown("##### 数据目录")
        st.caption("留空 = 使用默认（软件目录下的 `data/`）。只接受**绝对路径**，"
                   "目录不存在会自动创建并做写入测试。")
        new_dir = st.text_input("数据目录（绝对路径）",
                                value=(info.get("data_dir") or {}).get("path", ""),
                                key="path_data_dir",
                                placeholder=str((info.get("defaults") or {}).get("data_dir", "")))
        submitted = st.form_submit_button("保存路径设置", type="primary")
    if submitted:
        try:
            res = api.paths_put(new_dir.strip() or None)
            if res.get("restart_required"):
                st.warning("已保存。**请重启后端服务**（关闭系统后重新 start.bat）"
                           "才会切换到新的数据目录；数据库与行情缓存不会自动迁移。")
            else:
                st.success("已保存")
        except Exception as e:  # noqa: BLE001
            st.error(f"保存失败: {e}")

    with st.expander("如何迁移已有数据到新目录？", expanded=False):
        st.markdown(
            "1. 关闭系统（侧边栏「关闭系统」）；\n"
            "2. 把旧 `data/` 目录整体（含 `quant.db`、`market/`、`.yf_cache/`）拷到新位置；\n"
            "3. 在上方填入新路径并保存；\n"
            "4. 重新启动，本页确认「数据目录」已指向新位置、数据库体积与迁移前一致。"
        )
