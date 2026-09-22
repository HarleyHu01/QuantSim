"""数据管理页：状态表（含数据起止）、添加/删除标的、时间段选择、下载全部历史、一键更新、自动刷新。

下载结果反馈持久化在 session_state，跨 rerun 保留显示（直到下次操作或手动清除）。
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from web.streamlit.components import (TIMEFRAMES, adjust_label, tf_label,
                                      to_dataframe)

# 下载全部历史的拉取起点（数据源返回其最早可用数据，如 A股可到上市初期）
EARLIEST_START = "2000-01-01"

PERIOD_HINT = (
    "周期说明：分钟级数据（1m/5m/15m/30m/1h）免费数据源通常只提供最近 7–30 天；"
    "日线/周线/月线可获得数年历史；A股分钟数据（腾讯/新浪通道）通常只有最近几个交易日。"
    "如需更早的分钟数据，可指定「开始日期」尝试补拉（数据源限制内）。"
)


def render(api) -> None:
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

    # 自动刷新开启时首次渲染放循环里，避免顶部空白
    if not auto:
        with placeholder.container():
            status_panel(force)

    # ---- 添加新标的 ----
    st.divider()
    st.markdown("#### 添加新标的")
    st.caption("输入 symbol（如 TSLA / NVDA / 600519 / BTC/USDT），自动推断资产类型并拉取数据")
    c1, c2, c3 = st.columns([2, 2, 1])
    new_symbol = c1.text_input("symbol", "", key="new_sym",
                               placeholder="如 TSLA、NVDA、600519、BTC/USDT")
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
    return {"延迟正常": "color: #16a34a; font-weight: bold",
            "延迟偏高": "color: #d97706; font-weight: bold",
            "延迟异常": "color: #dc2626; font-weight: bold",
            "超时": "color: #6b7280"}.get(label, "")


def _render_health(status: dict) -> None:
    """健康检查面板：真实延迟 + 分级着色 + 检查时间/缓存标注 + 重新检测按钮。"""
    health = status.get("health", {})
    if not health:
        return
    rows = []
    for k, v in health.items():
        lat = v.get("latency_ms")
        healthy = v.get("healthy")
        hs = "正常" if healthy else ("超时" if "超时" in (v.get("error") or "") else "异常")
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
            start, end = r.get("start"), r.get("end")
            row["实际区间"] = f"{start} → {end}"
            if requested_start and start:
                try:
                    req = pd.Timestamp(requested_start)
                    act = pd.Timestamp(start)
                    row["提示"] = (f"数据源仅提供 {start} 起的部分数据（请求自 {requested_start}）"
                                   if act > req else "")
                except Exception:  # noqa: BLE001
                    row["提示"] = ""
            else:
                row["提示"] = ""
        else:
            row["行数"] = 0
            row["实际区间"] = ""
            row["提示"] = f"失败原因：{r.get('error') or r.get('message') or '未知'}"
        rows.append(row)
    return {"rows": rows}


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
