"""数据管理页：状态表（含数据起止）、添加/删除标的、时间段选择、下载全部历史、一键更新、自动刷新。

下载结果反馈持久化在 session_state，跨 rerun 保留显示（直到下次操作或手动清除）。
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from web.streamlit.components import (TIMEFRAMES, adjust_label, inject_theme,
                                      tf_label, to_dataframe)

# 下载全部历史的拉取起点（数据源返回其最早可用数据，如 A股可到上市初期）
EARLIEST_START = "2000-01-01"

PERIOD_HINT = (
    "支持 8 个周期（1m / 5m / 15m / 30m / 1h / 1d / 1w / 1M）。"
    "下载会按 baostock → akshare 的顺序自动尝试，某个源拿不到就换下一个；美股走 yfinance。"
    "各周期实际能拿到的深度取决于数据源当时的限制，可在下方「下载结果」里看到实际区间。"
)


def render(api) -> None:
    inject_theme()
    st.header("数据管理")
    st.info(PERIOD_HINT)

    auto = st.toggle("自动刷新", value=False, key="data_auto")
    interval = st.slider("刷新间隔（秒）", 1, 120, 5, key="data_iv")

    placeholder = st.empty()
    force = st.session_state.pop("health_force", False)

    def status_panel(force_refresh: bool = False) -> None:
        if force_refresh:
            with st.spinner("正在重新检测数据源（真实请求）..."):
                status = api.data_status(refresh=True)
            st.session_state.pop("health_checking", None)
        else:
            status = api.data_status()
        meta = status.get("symbols", [])
        if meta:
            df = to_dataframe(meta)
            display = pd.DataFrame({
                "标的": df["symbol"],
                "周期": df["timeframe"].map(tf_label),  # 1M → 月线 (Monthly)，不显示 1mo
                "数据源": df["source"],
                "行数": df["rows"],
                "数据起始": df["start_date"],
                "数据结束": df["end_date"],
            })
            st.dataframe(display, width="stretch", height=300)
            st.caption("「数据起始 → 数据结束」= 本地实际存储区间：分钟数据通常只覆盖最近几天，日线覆盖数年")
        else:
            st.info("尚无行情数据，请在下方「添加新标的」或「一键更新数据」")
        _render_health(status)
        with st.expander("下载渠道能力表（哪个源支持哪个周期）", expanded=False):
            _render_channels(api)
    # 自动刷新开启时首次渲染放循环里，避免顶部空白
    if not auto:
        with placeholder.container():
            status_panel(force)

    # ---- 数据源代理与存储路径（已迁到独立的「系统设置 → 数据设置」页，此处仅留入口提示）----
    st.caption("数据源代理（逐渠道直连/走代理）与存储路径已移到 **系统设置 → 数据设置** 页。")

    # ---- 添加新标的 ----
    st.divider()
    st.markdown("#### 添加新标的")
    st.caption("输入 symbol（国内如 600519 / 600519.SH / 000001.SZ / 510300.SH；国外如 AAPL / TSLA / SPY），自动推断资产类型与下载渠道")
    c1, c2, c3 = st.columns([2, 2, 1])
    new_symbol = c1.text_input("symbol", "", key="new_sym",
                               placeholder="如 600519.SH、000001.SZ、AAPL、TSLA")
    new_tf = c2.multiselect("周期", TIMEFRAMES, default=["1d"], key="new_tf",
                            format_func=tf_label)
    new_adj = c3.selectbox("A股复权", ["qfq", "hfq", "none"], index=0,
                           format_func=adjust_label, key="new_adj")
    dc1, dc2 = st.columns(2)
    new_start = dc1.text_input("开始日期（YYYY-MM-DD，留空 = 数据源最大范围）", "",
                               key="new_start")
    new_end = dc2.text_input("结束日期（YYYY-MM-DD，留空 = 今天）", "", key="new_end")
    if st.button("添加并更新", type="primary", key="new_add"):
        sym = new_symbol.strip().upper()
        if not sym:
            st.warning("请输入 symbol")
        else:
            try:
                api.add_symbol(sym)
                res = api.data_update(
                    symbols=[sym], timeframes=new_tf or None,
                    adjust=new_adj, incremental=False,
                    start_date=new_start.strip() or None, end_date=new_end.strip() or None)
            except Exception as e:  # noqa: BLE001
                st.error(f"拉取失败: {e}")
                st.stop()
            st.session_state["data_feedback"] = _build_feedback(
                res.get("results", []), new_start.strip() or None)
            st.rerun()

    # ---- 删除标的 ----
    st.divider()
    st.markdown("#### 删除标的")
    try:
        meta = api.data_status().get("symbols", [])
        existing = sorted({m["symbol"] for m in meta})
    except Exception:  # noqa: BLE001
        existing = []
    if existing:
        del_sym = st.selectbox("选择要删除的标的（同时删除已存行情数据）", existing, key="del_sym")
        if st.button("删除标的", key="del_btn"):
            r = api.delete_symbol(del_sym)
            st.success(f"已删除 {del_sym}（清理数据文件 {r.get('data_files_removed', 0)} 个）")
            st.rerun()
    else:
        st.caption("（暂无可删除的标的）")

    # ---- 一键更新 ----
    st.divider()
    st.markdown("#### 一键更新数据（全部启用标的）")
    c1, c2, c3 = st.columns(3)
    upd_tfs = c1.multiselect("周期", TIMEFRAMES, default=["1d"], key="upd_tf",
                             format_func=tf_label)
    upd_adj = c2.selectbox("A股复权", ["qfq", "hfq", "none"], index=0,
                           format_func=adjust_label, key="upd_adj")
    upd_full = c3.checkbox("全量重刷（清空后重拉）", key="upd_full")
    uc1, uc2 = st.columns(2)
    upd_start = uc1.text_input("开始日期（YYYY-MM-DD，留空 = 自动）", "", key="upd_start")
    upd_end = uc2.text_input("结束日期（YYYY-MM-DD，留空 = 今天）", "", key="upd_end")
    if st.button("一键更新数据", type="primary", key="upd_btn"):
        with st.spinner("正在更新行情数据..."):
            res = api.data_update(
                timeframes=upd_tfs or None, adjust=upd_adj, incremental=not upd_full,
                start_date=upd_start.strip() or None, end_date=upd_end.strip() or None)
        st.session_state["data_feedback"] = _build_feedback(
            res.get("results", []), upd_start.strip() or None)
        st.rerun()

    # ---- 下载全部历史 ----
    st.divider()
    st.markdown("#### 下载全部历史")
    st.caption(f"对选定标的 × 周期，从最早可用日期（{EARLIEST_START} 起，数据源限制内）下载到最新")
    hc1, hc2 = st.columns(2)
    hist_syms = hc1.multiselect("标的（默认全部已有数据）", existing or ["（暂无标的）"],
                                default=existing, key="hist_sym")
    hist_tfs = hc2.multiselect("周期", TIMEFRAMES, default=["1d"], key="hist_tf",
                               format_func=tf_label)
    hist_adj = st.selectbox("A股复权", ["qfq", "hfq", "none"], index=0,
                            format_func=adjust_label, key="hist_adj")
    if st.button("下载全部历史", type="primary", key="hist_btn"):
        syms = [s for s in hist_syms if s != "（暂无标的）"] or None
        with st.spinner("正在从最早可用日期下载全部历史..."):
            res = api.data_update(symbols=syms, timeframes=hist_tfs or None,
                                  adjust=hist_adj, incremental=False,
                                  start_date=EARLIEST_START, end_date=None)
        st.session_state["data_feedback"] = _build_feedback(
            res.get("results", []), EARLIEST_START)
        st.rerun()

    # ---- 下载结果反馈（持久化，直到下次操作或清除） ----
    _render_feedback()

    if auto:
        # 轮询 API + st.empty() 局部刷新（自动循环不强制；"重新检测"按钮优先强制实时检测）
        first = True
        while True:
            with placeholder.container():
                status_panel(force if first else False)
            first = False
            time.sleep(interval)
            st.rerun()


def _latency_label(ms: float | None) -> str:
    """延迟分级：<500 正常 / 500-2000 偏高 / >2000 异常 / None 超时。"""
    if ms is None:
        return "超时"
    if ms < 500:
        return "延迟正常"
    if ms <= 2000:
        return "延迟偏高"
    return "延迟异常"


def _latency_color(label: str) -> str:
    return {"延迟正常": "color: #34d399; font-weight: bold",
            "延迟偏高": "color: #fbbf24; font-weight: bold",
            "延迟异常": "color: #f87171; font-weight: bold",
            "超时": "color: #7d90b3"}.get(label, "")


def _health_status(v: dict) -> str:
    """健康状态文字：正常 / 部分可用 / 超时 / 异常。

    「部分可用」用于多通道数据源（akshare）：第一条通道挂了但有兜底通道接住时，
    只报「正常」会掩盖某条通道不可用的事实，只报「异常」又与实际能否下载矛盾。
    """
    if v.get("healthy"):
        chans = v.get("channels") or []
        if chans and not all(c.get("healthy") for c in chans):
            return "部分可用"
        return "正常"
    return "超时" if "超时" in (v.get("error") or "") else "异常"


def _channels_brief(v: dict) -> str:
    """多通道数据源的逐通道摘要，形如「东方财富 ✗ ｜ 腾讯 ✓ 640ms ｜ 新浪 ✓ 1359ms」。

    全部信息都塞进这一列，不再单独挂一行 caption。
    熔断中的通道只标「暂不可用」，**不显示剩余秒数**——那个数字每秒都在变，
    写进表格既没有任何操作价值，还会让人以为是个固定配置项。
    """
    chans = v.get("channels") or []
    if not chans:
        return ""
    parts = []
    for c in chans:
        if c.get("healthy"):
            seg = f"{c.get('name')} ✓ {c.get('latency_ms') or 0:.0f}ms"
        else:
            seg = f"{c.get('name')} ✗"
            br = c.get("breaker")
            if br and br != "ok":
                seg += "（暂不可用）"
        parts.append(seg)
    return " ｜ ".join(parts)


def _render_health(status: dict) -> None:
    """健康检查面板：真实延迟 + 分级着色 + 通道明细 + 检查时间/缓存标注 + 重新检测按钮。"""
    health = status.get("health", {})
    if not health:
        return
    rows = []
    for k, v in health.items():
        lat = v.get("latency_ms")
        hs = _health_status(v)
        checked = v.get("checked_at")
        ts_str = "—"
        if checked:
            try:
                ts_str = time.strftime("%H:%M:%S", time.localtime(checked))
            except Exception:  # noqa: BLE001
                ts_str = "—"
            if v.get("cached"):
                age = v.get("cached_age_seconds") or 0
                ts_str += f"（已缓存 {age:.0f} 秒）"
        rows.append({
            "数据源": k,
            "实际通道": v.get("source") or "—",
            "通道明细": _channels_brief(v) or "—",
            "健康状态": hs,
            "本次检查延迟": f"{lat:.0f} ms" if lat is not None else "—",
            "延迟分级": _latency_label(lat),
            "上次检查时间": ts_str,
            "错误": v.get("error") or "",
        })
    hdf = to_dataframe(rows)
    hc1, hc2 = st.columns([4.4, 1])
    with hc1:
        st.caption(f"数据源健康检查（共 {len(hdf)} 个；延迟为真实请求耗时，缓存命中会标注「已缓存 X 秒」）")
    with hc2:
        checking = st.session_state.get("health_checking", False)
        if checking:
            st.button("检测中...", disabled=True, use_container_width=True, key="hk_btn")
        elif st.button("重新检测", use_container_width=True, key="hk_btn"):
            st.session_state["health_checking"] = True
            st.session_state["health_force"] = True
            st.rerun()
    st.dataframe(
        hdf.style.map(_latency_color, subset=["延迟分级"]),
        width="stretch",
    )


def _build_feedback(results: list[dict], requested_start: str | None) -> dict:
    """把更新结果转成持久化反馈行：状态 / 实际区间 / 部分数据提示 / 失败原因。"""
    rows = []
    for r in results:
        ok = r.get("ok")
        row = {"标的": r.get("symbol"), "周期": tf_label(r.get("timeframe", "")),
               "状态": "成功" if ok else "失败"}
        if ok:
            row["行数"] = r.get("rows")
            row["下载渠道"] = r.get("source") or "—"
            start, end = r.get("start"), r.get("end")
            row["实际区间"] = f"{start} → {end}"
            notes: list[str] = []
            if requested_start and start:
                try:
                    req = pd.Timestamp(requested_start)
                    act = pd.Timestamp(start)
                    if act > req:
                        notes.append(f"数据源仅提供 {start} 起的部分数据（请求自 {requested_start}）")
                except Exception:  # noqa: BLE001
                    pass
            msg = r.get("message") or ""
            if "已跳过" in msg:  # 前置渠道不支持该周期时的说明
                notes.append(msg.split("（已跳过：", 1)[-1].rstrip("）"))
            row["提示"] = "；".join(n for n in notes if n)
        else:
            row["行数"] = 0
            row["下载渠道"] = "—"
            row["实际区间"] = ""
            row["提示"] = f"失败原因：{r.get('error') or r.get('message') or '未知'}"
        rows.append(row)
    return {"rows": rows}


# 代理模式常量的真身在 web/streamlit/components.py，这里保留别名仅为向后兼容
from web.streamlit.components import PROXY_MODE_HINTS, PROXY_MODE_LABELS  # noqa: E402,F401


def _render_channels(api) -> None:
    try:
        data = api.data_channels()
    except Exception as e:  # noqa: BLE001
        st.caption(f"（渠道信息不可用：{e}）")
        return
    rows = []
    tfs = data.get("timeframes", [])
    for ch in data.get("channels", []):
        row = {"渠道": ch.get("display_name") or ch.get("name"),
               "标识": ch.get("name"),
               "支持资产": "、".join(ch.get("asset_types") or [])}
        for tf in tfs:
            info = (ch.get("timeframes") or {}).get(tf) or {}
            row[tf] = "✓" if info.get("supported") else "—"
        rows.append(row)
    if rows:
        st.dataframe(to_dataframe(rows), width="stretch")
        st.caption("✓ = 支持；— = 不支持。下载顺序：baostock（主）→ akshare（备选，内部自动降级）；"
                   "美股走 yfinance")

    # akshare 内部三条腿：外层只显示一项 akshare，但各腿支持的周期并不相同。
    subs = next((c.get("sub_channels") or [] for c in data.get("channels", [])
                 if c.get("name") == "akshare"), [])
    if subs:
        st.markdown("###### akshare 内部通道")
        sub_rows = []
        for sc in subs:
            row = {"通道": sc.get("display_name") or sc.get("name"),
                   "当前状态": _channel_state_label(sc.get("state"))}
            for tf in tfs:
                row[tf] = "✓" if (sc.get("timeframes") or {}).get(tf) else "—"
            sub_rows.append(row)
        st.dataframe(to_dataframe(sub_rows), width="stretch")


def _channel_state_label(state) -> str:
    """把 channel_status() 的 ``ok`` / ``cooling <n>s`` 转成人话。

    不显示剩余秒数：那是个不断变动的实现细节，对用户没有任何操作价值。
    """
    if not state or state == "unknown":
        return "—"
    if state == "ok":
        return "可用"
    return "暂不可用"



def _render_feedback() -> None:
    fb = st.session_state.get("data_feedback")
    if not fb:
        return
    rows = fb.get("rows") or []
    df = to_dataframe(rows)
    ok_n = int((df["状态"] == "成功").sum()) if len(df) else 0
    st.divider()
    st.markdown("##### 最近一次下载结果")
    st.caption(f"成功 {ok_n}/{len(df)}。实际区间与数据源限制见下表；本反馈保留显示直到下次操作或清除")
    if len(df):
        st.dataframe(df, width="stretch")
    if st.button("清除结果反馈", key="fb_clear"):
        st.session_state["data_feedback"] = None
        st.rerun()
