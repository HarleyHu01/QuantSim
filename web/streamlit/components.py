"""Streamlit 通用组件：schema 驱动参数表单、指标卡片、ECharts 图表。"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M"]
MODE_LABELS = {"signal_merge": "信号加权 (signal_merge)", "capital_split": "资金分配 (capital_split)",
               "equal": "等权 (equal)"}
ENGINE_LABELS = {"event_driven": "事件驱动回测 (event_driven)", "vectorized": "向量化回测 (vectorized)"}
EXEC_LABELS = {"next_open": "下一根开盘成交", "same_close": "当根收盘成交"}
ADJUST_LABELS = {"qfq": "前复权 (qfq)", "hfq": "后复权 (hfq)", "none": "不复权 (none)"}
METRIC_LABELS = {
    "total_return": "总收益率", "annual_return": "年化收益率", "max_drawdown": "最大回撤",
    "sharpe": "夏普比率", "sortino": "索提诺比率", "calmar": "卡玛比率",
    "win_rate": "胜率", "profit_factor": "盈亏比", "trade_count": "交易次数",
    "avg_trade_pnl": "平均盈亏", "avg_exposure": "平均暴露度", "benchmark_return": "基准收益",
    "volatility": "年化波动率", "elapsed_seconds": "耗时（秒）",
    # 对比指标
    "buyhold_return": "买入持有收益", "buyhold_annual": "买入持有年化",
    "excess_return_vs_buyhold": "超额收益(vs买入持有)",
    "benchmark_annual": "基准年化", "benchmark_max_drawdown": "基准最大回撤",
    "beta": "Beta", "alpha": "Alpha(年化)", "information_ratio": "信息比率",
}
PAPER_STATUS_LABELS = {"running": "运行中", "stopped": "已停止"}

# 时间周期中英文映射（注意 1M 在文件名中是 1mo，UI 展示必须还原为 1M/月线）
TF_LABELS = {
    "1m": "1分钟 (1 Minute)",
    "5m": "5分钟 (5 Minutes)",
    "15m": "15分钟 (15 Minutes)",
    "30m": "30分钟 (30 Minutes)",
    "1h": "1小时 (1 Hour)",
    "2h": "2小时 (2 Hours)",
    "4h": "4小时 (4 Hours)",
    "1d": "日线 (Daily)",
    "1w": "周线 (Weekly)",
    "1M": "月线 (Monthly)",
}


def tf_label(tf: str) -> str:
    return TF_LABELS.get(tf, tf)


def strategy_label(s: dict) -> str:
    """策略显示名：中英文全称（display_name），缺省回退 name/key。"""
    return s.get("display_name") or s.get("name") or s.get("key", "")


def adjust_label(a: str) -> str:
    return ADJUST_LABELS.get(a, a)


def metric_label(k: str) -> str:
    return METRIC_LABELS.get(k, k)


def echarts(option: dict, height: int = 380) -> None:
    """内嵌 ECharts（CDN），option 为 ECharts 配置。"""
    opt_json = json.dumps(option, ensure_ascii=False)
    html = f"""
    <div id="chart" style="width:100%;height:{height}px;"></div>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <script>
      var c = echarts.init(document.getElementById('chart'));
      c.setOption({opt_json});
      window.addEventListener('resize', function() {{ c.resize(); }});
    </script>
    """
    st.components.v1.html(html, height=height + 8, scrolling=False)


# ================= 策略参数表单（按 schema 渲染） =================

def render_params_form(schema: dict, current: dict | None, prefix: str = "p") -> dict:
    """按 params_schema 自动渲染控件，返回当前值字典。

    - bool → checkbox；integer/number 带范围 → slider（范围大则 number_input）
    - string 带 enum → selectbox；str → text_input
    """
    props = (schema or {}).get("properties", {})
    current = current or {}
    values: dict = {}
    for name, spec in props.items():
        title = spec.get("title", name)
        default = spec.get("default")
        val = current.get(name, default)
        ptype = spec.get("type")
        key = f"{prefix}_{name}"
        lo, hi = spec.get("minimum"), spec.get("maximum")

        # 可空字段（int|None 等 → anyOf 含 null）：解析实际类型
        nullable = False
        if ptype is None and isinstance(spec.get("anyOf"), list):
            non_null = [t for t in spec["anyOf"] if t.get("type") != "null"]
            if non_null:
                nullable = True
                ptype = non_null[0].get("type")
                lo = non_null[0].get("minimum", lo)
                hi = non_null[0].get("maximum", hi)
                default = non_null[0].get("default", default)
        if val is None and nullable:
            continue  # 可选字段未设置：跳过，避免 str(None) 污染配置

        if ptype == "boolean":
            values[name] = st.checkbox(title, value=bool(val), key=key)
        elif ptype == "integer":
            if lo is not None and hi is not None and (hi - lo) <= 500:
                values[name] = st.slider(title, int(lo), int(hi), int(val), key=key)
            else:
                values[name] = int(st.number_input(
                    title, value=int(val), min_value=int(lo) if lo is not None else None,
                    max_value=int(hi) if hi is not None else None, step=1, key=key))
        elif ptype == "number":
            if lo is not None and hi is not None:
                step = 0.01 if (hi - lo) <= 2 else 0.1
                values[name] = st.slider(title, float(lo), float(hi), float(val), step, key=key)
            else:
                values[name] = st.number_input(title, value=float(val), step=0.01,
                                               format="%.4f", key=key)
        elif ptype == "string":
            enum = spec.get("enum")
            if enum:
                idx = enum.index(val) if val in enum else 0
                values[name] = st.selectbox(title, enum, index=idx, key=key)
            else:
                values[name] = st.text_input(title, value=str(val), key=key)
        else:
            values[name] = st.text_input(title, value=str(val), key=key)
    return values


# ================= 指标卡片 =================

def metrics_row(metrics: dict) -> None:
    def pct(v, digits=2):
        return "—" if v is None else f"{v * 100:.{digits}f}%"

    c = st.columns(6)
    c[0].metric("总收益率", pct(metrics.get("total_return")))
    c[1].metric("年化收益率", pct(metrics.get("annual_return")))
    c[2].metric("最大回撤", pct(metrics.get("max_drawdown")))
    c[3].metric("夏普比率", f"{metrics.get('sharpe'):.3f}" if metrics.get("sharpe") is not None else "—")
    c[4].metric("胜率", pct(metrics.get("win_rate")))
    c[5].metric("盈亏比", f"{metrics.get('profit_factor'):.2f}" if metrics.get("profit_factor") is not None else "—")
    c2 = st.columns(6)
    c2[0].metric("交易次数", int(metrics.get("trade_count", 0)))
    c2[1].metric("索提诺比率", f"{metrics.get('sortino'):.3f}" if metrics.get("sortino") is not None else "—")
    c2[2].metric("卡玛比率", f"{metrics.get('calmar'):.3f}" if metrics.get("calmar") is not None else "—")
    c2[3].metric("暴露度", pct(metrics.get("avg_exposure")))
    c2[4].metric("年化波动", pct(metrics.get("volatility")))
    c2[5].metric("基准收益", pct(metrics.get("benchmark_return")))
    # 第三行：对比指标（策略 vs 买入持有 vs 基准）
    c3 = st.columns(6)
    c3[0].metric("超额收益(vs买入持有)", pct(metrics.get("excess_return_vs_buyhold")))
    c3[1].metric("买入持有年化", pct(metrics.get("buyhold_annual")))
    c3[2].metric("基准最大回撤", pct(metrics.get("benchmark_max_drawdown")))
    c3[3].metric("Beta", f"{metrics.get('beta'):.3f}" if metrics.get("beta") is not None else "—")
    c3[4].metric("Alpha(年化)", pct(metrics.get("alpha")))
    c3[5].metric("信息比率", f"{metrics.get('information_ratio'):.3f}" if metrics.get("information_ratio") is not None else "—")
    if metrics.get("elapsed_seconds") is not None:
        st.caption(f"耗时 {metrics['elapsed_seconds']}s")


# ================= ECharts 图表 =================

def _ts_list(rows: list[dict], col: str = "ts") -> list[str]:
    return [str(r.get(col, ""))[:16] for r in rows]


def comparison_option(eq_rows: list[dict]) -> dict:
    """归一化收益对比（初始=100%）：策略收益 / 买入持有 / 基准指数 / 标的股价。

    各序列以其首值归一化，直观对比"策略是否跑赢无脑持有/大盘"。
    """
    x = _ts_list(eq_rows)
    lines = [
        ("策略收益", "equity", "#1f77b4", "solid", 2),
        ("买入持有", "buyhold", "#2ca02c", "dashed", 1),
        ("基准指数", "benchmark", "#ff7f0e", "dashed", 1),
        ("标的股价", "symbol_price", "#9467bd", "dotted", 1),
    ]
    series, legend = [], []
    for name, key, color, ltype, width in lines:
        vals = [r.get(key) for r in eq_rows]
        first = next((v for v in vals if v is not None and v), None)
        if first is None:
            continue  # 旧回测无该列 → 跳过该线
        data = [None if v is None else round(v / first * 100, 3) for v in vals]
        series.append({"name": name, "type": "line", "data": data, "showSymbol": False,
                       "color": color,
                       "lineStyle": {"width": width, "type": ltype}})
        legend.append(name)
    return {
        "title": {"text": "收益对比（归一化，初始=100%）", "left": "center",
                  "textStyle": {"fontSize": 13}},
        "tooltip": {"trigger": "axis", "valueFormatter": "{value}%"},
        "legend": {"data": legend, "top": 26},
        "grid": {"left": 60, "right": 20, "top": 60, "bottom": 30},
        "xAxis": {"type": "category", "data": x, "axisLabel": {"fontSize": 10}},
        "yAxis": {"type": "value", "scale": True, "axisLabel": {"fontSize": 10,
                                                                "formatter": "{value}%"}},
        "series": series,
    }


def drawdown_option(eq_rows: list[dict]) -> dict:
    """回撤对比（%）：策略回撤 vs 基准回撤。"""
    x = _ts_list(eq_rows)
    dd = [round((r.get("drawdown") or 0) * 100, 3) for r in eq_rows]
    # 基准回撤：1 − b/cummax(b)
    bench_dd, peak = [], None
    for r in eq_rows:
        b = r.get("benchmark")
        if b is None:
            bench_dd.append(None)
            continue
        peak = b if peak is None else max(peak, b)
        bench_dd.append(round((1.0 - b / peak) * 100, 3) if peak else 0.0)
    return {
        "title": {"text": "回撤对比（%）", "left": "center", "textStyle": {"fontSize": 13}},
        "tooltip": {"trigger": "axis", "valueFormatter": "{value}%"},
        "legend": {"data": ["策略回撤", "基准回撤"], "top": 26},
        "grid": {"left": 60, "right": 20, "top": 60, "bottom": 30},
        "xAxis": {"type": "category", "data": x, "axisLabel": {"fontSize": 10}},
        "yAxis": {"type": "value", "axisLabel": {"fontSize": 10, "formatter": "{value}%"}},
        "series": [
            {"name": "策略回撤", "type": "line", "data": dd, "showSymbol": False,
             "lineStyle": {"color": "#d62728", "width": 2},
             "areaStyle": {"color": "rgba(214,39,40,0.15)"}},
            {"name": "基准回撤", "type": "line", "data": bench_dd, "showSymbol": False,
             "lineStyle": {"color": "#ff7f0e", "width": 1, "type": "dashed"},
             "areaStyle": {"color": "rgba(255,127,14,0.08)"}},
        ],
    }


def monthly_heatmap_option(eq_rows: list[dict], series_key: str = "equity") -> dict:
    """月度收益热力图（%）：权益曲线按月复合收益的年 × 月矩阵。"""
    ts = pd.to_datetime([r.get("ts") for r in eq_rows], errors="coerce")
    vals = pd.Series([r.get(series_key) for r in eq_rows], index=ts, dtype="float64").dropna()
    if len(vals) < 3:
        return {}
    norm = vals / vals.iloc[0]
    rets = norm.pct_change().dropna()
    monthly = rets.resample("ME").apply(lambda s: (1.0 + s).prod() - 1.0) * 100.0
    if monthly.empty:
        return {}
    m = monthly.to_frame("ret")
    m["year"], m["month"] = m.index.year, m.index.month
    years = sorted(m["year"].unique())
    yidx = {y: i for i, y in enumerate(years)}
    data = [[yidx[int(row.year)], int(row.month) - 1, round(float(row.ret), 2)]
            for row in m.itertuples()]
    vmax = max(abs(float(v)) for v in m["ret"]) or 1.0
    return {
        "title": {"text": "策略月度收益（%）", "left": "center", "textStyle": {"fontSize": 13}},
        "tooltip": {"position": "top",
                    "formatter": "function(p){return p.value[2] + '%';}"},
        "grid": {"left": 40, "right": 20, "top": 60, "bottom": 60},
        "xAxis": {"type": "category", "data": [f"{i}月" for i in range(1, 13)],
                  "splitArea": {"show": True}, "axisLabel": {"fontSize": 10}},
        "yAxis": {"type": "category", "data": [str(y) for y in years],
                  "splitArea": {"show": True}, "axisLabel": {"fontSize": 10}},
        "visualMap": {"min": -vmax, "max": vmax, "calculable": True, "orient": "horizontal",
                      "left": "center", "bottom": 0, "text": ["高", "低"],
                      "inRange": {"color": ["#14b143", "#f5f5f5", "#ef232a"]}},
        "series": [{"type": "heatmap", "data": data,
                    "label": {"show": True, "fontSize": 10, "formatter": "{c}%"}}],
    }


def kline_option(bars: list[dict], trades: list[dict], symbol: str) -> dict:
    """K 线 + 成交量 + 买卖点标记。"""
    if not bars:
        return {}
    dates = [str(b.get("datetime", ""))[:16] for b in bars]
    kdata = [[b.get("open"), b.get("close"), b.get("low"), b.get("high")] for b in bars]
    vols = [{"value": b.get("volume") or 0, "itemStyle": {"color": "#ef232a" if (b.get("close") or 0) >= (b.get("open") or 0) else "#14b143"}} for b in bars]

    def markers(side: str, color: str):
        """买卖点：红色上箭头=买入（标在最低价下方），绿色图钉=卖出（标在最高价上方）。"""
        pts = []
        for t in trades:
            if t.get("side") != side:
                continue
            ts = str(t.get("timestamp", ""))[:16]
            if ts in dates:
                bar = bars[dates.index(ts)]
                y = bar.get("low") * 0.995 if side == "buy" else bar.get("high") * 1.005
                pts.append({
                    "coord": [dates.index(ts), y],
                    "value": f"{'买入' if side == 'buy' else '卖出'} {t.get('qty'):.0f}@{t.get('price'):.2f}",
                    "itemStyle": {"color": color, "borderColor": "#fff", "borderWidth": 1},
                    "label": {"show": True, "formatter": "B" if side == "buy" else "S",
                              "position": "top", "fontWeight": "bold", "color": color},
                })
        return pts

    return {
        "title": {"text": f"{symbol} K线 + 买卖点 + 成交量", "left": "center", "textStyle": {"fontSize": 13}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": {"data": ["K线", "买入", "卖出", "成交量"], "top": 26},
        "axisPointer": {"link": [{"xAxisIndex": "all"}]},
        "grid": [{"left": 60, "right": 20, "top": 60, "height": "46%"},
                 {"left": 60, "right": 20, "top": "70%", "height": "16%"}],
        "xAxis": [
            {"type": "category", "data": dates, "axisLabel": {"fontSize": 9}},
            {"type": "category", "gridIndex": 1, "data": dates, "axisLabel": {"show": False}},
        ],
        "yAxis": [
            {"scale": True, "axisLabel": {"fontSize": 9}},
            {"gridIndex": 1, "axisLabel": {"fontSize": 9}},
        ],
        "dataZoom": [{"type": "inside", "xAxisIndex": [0, 1], "start": 0, "end": 100}],
        "series": [
            {"name": "K线", "type": "candlestick", "data": kdata,
             "itemStyle": {"color": "#ef232a", "color0": "#14b143",
                           "borderColor": "#ef232a", "borderColor0": "#14b143"}},
            {"name": "买入", "type": "scatter", "data": markers("buy", "#ef232a"),
             "symbol": "triangle", "symbolSize": 16, "z": 20},
            {"name": "卖出", "type": "scatter", "data": markers("sell", "#14b143"),
             "symbol": "pin", "symbolSize": 18, "z": 20},
            {"name": "成交量", "type": "bar", "xAxisIndex": 1, "yAxisIndex": 1, "data": vols},
        ],
    }


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()
