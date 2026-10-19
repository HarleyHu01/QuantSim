"""模型预测页：单模型试预测 + 多模型加权集成。

这个页面回答两个问题：

1. **某一个模型怎么看？** —— 选 Kronos 单跑，画历史 + 预测曲线。
2. **综合起来该不该动手？** —— 多模型加权投票，再过一组硬性门控
   （幅度下限 / 波动率归一 / 一致性 / 分歧度 / 最少模型数），
   最后给一个仓位。空仓时也会列出每条门控的实测值，说明为什么不动。

定位是**研究工具**：不写库、不下单。确认要用了再去「回测」页跑历史，
或建模拟盘账户跑实时。
"""
from __future__ import annotations

import streamlit as st

from web.streamlit.components import (TIMEFRAMES, echarts,
                                      forecast_option, inject_theme, tf_label)

_INSTALL_CMDS = """# 安装 Kronos / 深度学习依赖（CPU 版 torch，约 200MB）
pip install -r requirements-ml.txt

# 国内网络先设 HuggingFace 镜像
set HF_ENDPOINT=https://hf-mirror.com

# 想要 GPU 加速，卸掉 CPU 版再装 CUDA 版
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124"""

_EXTERNAL_CMDS = {
    "chronos": "pip install chronos-forecasting",
    "timesfm": "pip install timesfm",
    "moirai": "pip install uni2ts",
}

#: kind → 中文说明（UI 分组用）
_KIND_LABELS = {
    "dl": "深度学习（金融专用）",
    "stat": "统计基线（零依赖）",
    "external": "通用时序基础模型（需额外安装）",
}


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


def _symbols(api) -> list[str]:
    try:
        stt = api.data_status()
    except Exception:  # noqa: BLE001
        return []
    return [m["symbol"] for m in stt.get("symbols", []) if m.get("rows")]


# ==================== 单模型 ====================

def _render_kronos_setup(status: dict) -> None:
    if not status.get("enabled", True):
        st.warning("Kronos 已在配置中关闭。到 config/default.yaml 把 ml.kronos.enabled 改回 true 即可启用。")
        return
    missing = status.get("missing") or []
    st.warning("Kronos 运行所需依赖尚未安装：" + f"`{'`, `'.join(missing)}`")
    st.code(_INSTALL_CMDS, language="bash")
    st.caption("不安装也不影响其它功能——经典策略、回测、模拟盘、AI 策略生成都不依赖它。")


def _render_single(api, status: dict, symbols: list[str]) -> None:
    spec = status.get("spec") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("模型", str(spec.get("model", "")).split("/")[-1])
    c2.metric("设备", status.get("device") or "—")
    c3.metric("参数量", f"{status['params_million']:.1f}M" if status.get("params_million") else "未加载")
    c4.metric("加载耗时", f"{status['load_seconds']:.1f}s" if status.get("load_seconds") else "—")

    if not status.get("loaded"):
        st.caption("模型尚未加载。首次加载含权重下载，约 4~50 秒；加载后进程内复用，之后每次预测约 0.7 秒。")
        if st.button("加载模型", type="primary", key="kr_load"):
            with st.spinner("加载中（首次需下载权重，请稍候）..."):
                try:
                    r = api.ml_kronos_load()
                except Exception as e:  # noqa: BLE001
                    st.error(f"加载失败：{_err_detail(e)}")
                    return
            if r.get("ok"):
                st.success(f"加载成功：{r.get('device')}，{r.get('load_seconds')}s")
                st.rerun()
            else:
                st.error(r.get("message") or "加载失败")
    else:
        st.caption(f"模型已加载（{status.get('load_seconds')}s），可直接预测。")

    st.divider()
    st.markdown("#### 单模型试预测")
    f1, f2, f3 = st.columns([2, 1.4, 1])
    sym = f1.selectbox("标的", symbols, key="kr_sym")
    tf = f2.selectbox("周期", TIMEFRAMES,
                      index=TIMEFRAMES.index("5m") if "5m" in TIMEFRAMES else 0,
                      format_func=tf_label, key="kr_tf")
    choices = status.get("model_choices") or ["Kronos-small"]
    # 默认选中配置里指定的模型（而不是列表第一个），否则会莫名其妙去加载 mini
    cur_model = str(spec.get("model") or "").split("/")[-1]
    if cur_model not in choices:
        cur_model = choices[0]
    model = f3.selectbox("模型", choices, index=choices.index(cur_model),
                         key="kr_model",
                         help="与 config 的 ml.kronos.model 不一致时会另加载一份权重（首次较慢）")

    max_ctx = (status.get("max_context_by_model") or {}).get(model, 512)
    g1, g2, g3 = st.columns(3)
    bars = g1.number_input("上下文长度（根）", 32, int(max_ctx), min(240, int(max_ctx)),
                           step=32, key="kr_bars",
                           help=f"{model} 的上下文上限是 {max_ctx} 根，超过会被截断")
    pred_len = g2.number_input("预测长度（根）", 1, 120, int(spec.get("pred_len") or 12),
                               step=1, key="kr_plen")
    samples = g3.number_input("采样次数", 1, 8, int(spec.get("sample_count") or 1), step=1,
                              key="kr_sc", help=">1 时取多路径均值，更稳但耗时线性增加")

    if st.button("开始预测", type="primary", key="kr_go"):
        with st.spinner(f"预测中（{bars} 根上下文 → {pred_len} 根）..."):
            try:
                r = api.ml_kronos_predict(sym, timeframe=tf, bars=int(bars),
                                          pred_len=int(pred_len), model=model,
                                          sample_count=int(samples))
            except Exception as e:  # noqa: BLE001
                st.error(f"预测失败：{_err_detail(e)}")
                st.session_state.pop("kr_result", None)
                return
        st.session_state["kr_result"] = r

    r = st.session_state.get("kr_result")
    if not r:
        return

    last_close = r.get("last_close") or 0.0
    exp_ret = r.get("expected_return") or 0.0
    fcast = r.get("forecast") or []
    pred_close = fcast[-1].get("close") if fcast else None
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最新收盘", f"{last_close:.2f}", help=f"最后一根 K 线：{r.get('last_bar')}")
    # 标签说的是「预测长度根之后」，不是「上下文长度根之后」
    m2.metric(f"{len(fcast)} 根后预测收盘",
              f"{pred_close:.2f}" if pred_close else "—",
              delta=f"{exp_ret:+.2%}" if pred_close else None)
    m3.metric("预测幅度", f"{exp_ret:+.2%}")
    m4.metric("耗时", f"{r.get('elapsed_seconds')}s")

    st.caption(f"{r.get('symbol')} · {tf_label(r.get('timeframe', ''))} · "
               f"{r.get('context_bars')} 根上下文 → 预测 {len(fcast)} 根 · "
               f"设备 {r.get('device')}")

    option = forecast_option(r)
    if option:
        echarts(option, height=460)

    with st.expander("预测明细", expanded=False):
        st.dataframe(
            [
                {"时间": str(x.get("t"))[:16], "开": x.get("open"), "高": x.get("high"),
                 "低": x.get("low"), "收": x.get("close"), "量": x.get("volume")}
                for x in fcast
            ],
            use_container_width=True, hide_index=True,
        )


# ==================== 多模型集成 ====================

_GATE_LABELS = {
    "min_models": "有效模型数",
    "min_abs_return": "最小预测幅度",
    "min_return_sigma": "波动率倍数下限",
    "min_agreement": "一致性",
    "max_dispersion": "分歧度",
    "direction": "方向",
    "override": "强制空仓",
}


def _fmt_gate_value(key: str, v) -> str:
    """门控值按语义格式化。

    注意 ``min_return_sigma`` 的 ``need`` 在 gates 里已经被乘过波动率，
    存的是**幅度比率**（不是倍数），所以它和 ``got`` 一样按百分数显示；
    倍数本身在配置里（界面上那个「波动率倍数下限」输入框）。
    """
    if v is None:
        return "—"
    if not isinstance(v, (int, float)):
        return str(v)
    if key == "min_models":
        return f"{int(v)}"
    if key == "max_dispersion":
        return f"{v:.4f}"
    if key in ("min_abs_return", "min_return_sigma", "min_agreement"):
        return f"{v:.2%}"
    return f"{v:.4f}"



def _gate_rows(result: dict) -> list[dict]:
    """把门控结果整理成表格行（含阈值与实测值，便于判断差多少）。

    后端 ``gates`` 是 ``{key: {need, got, pass, sigma?, allow_short?}}``，
    这里统一摊平成「门控 / 实测 / 要求 / 结果」四列。
    """
    rows = []
    for key, g in (result.get("gates") or {}).items():
        if not isinstance(g, dict):
            rows.append({"门控": _GATE_LABELS.get(key, key), "实测": str(g),
                         "要求": "—", "结果": "—"})
            continue
        got = g.get("got")
        need = g.get("need")
        if key == "direction":
            allowed = g.get("allow_short")
            rows.append({
                "门控": _GATE_LABELS.get(key, key),
                "实测": "允许做空" if allowed else "仅做多",
                "要求": "预测方向在允许范围内",
                "结果": "通过" if g.get("pass") else "不通过",
            })
            continue
        rows.append({
            "门控": _GATE_LABELS.get(key, key),
            "实测": _fmt_gate_value(key, got),
            "要求": "≥ " + _fmt_gate_value(key, need) if key != "max_dispersion"
                    else "≤ " + _fmt_gate_value(key, need),
            "结果": "通过" if g.get("pass") else ("不通过" if g.get("pass") is False else "跳过"),
        })
    return rows



def _render_ensemble(api, symbols: list[str]) -> None:
    try:
        cfg_resp = api.ml_ensemble_config()
    except Exception as e:  # noqa: BLE001
        st.error(f"读取集成配置失败：{_err_detail(e)}")
        return

    cfg = cfg_resp.get("config") or {}
    models = cfg_resp.get("models") or []
    weights = dict(cfg.get("weights") or {})

    st.caption(
        "多模型加权投票 + 硬性门控。默认只启用 Kronos 与几个统计基线——"
        "统计基线不是凑数的，随机游走是零假设：任何模型跑不赢它，就说明预测没有信息量。"
    )

    # ---- 模型清单 ----
    by_kind: dict[str, list[dict]] = {}
    for m in models:
        by_kind.setdefault(m.get("kind") or "stat", []).append(m)

    with st.expander("参与集成的模型", expanded=False):
        for kind in ("dl", "stat", "external"):
            group = by_kind.get(kind) or []
            if not group:
                continue
            st.markdown(f"**{_KIND_LABELS.get(kind, kind)}**")
            st.dataframe(
                [{
                    "标识": m.get("key"),
                    "名称": m.get("label"),
                    "可用": "✅" if m.get("available") else "❌ 未安装",
                    "权重": weights.get(m.get("key"), 0.0),
                    "说明": m.get("detail") or m.get("error") or "",
                } for m in group],
                use_container_width=True, hide_index=True,
            )
        uninstalled = [m for m in models if not m.get("available")]
        if uninstalled:
            st.caption("未安装的模型不会参与投票。要启用它们：")
            for m in uninstalled:
                cmd = _EXTERNAL_CMDS.get(m.get("key"))
                if cmd:
                    st.code(f"{cmd}   # {m.get('label')}", language="bash")

    st.divider()

    # ---- 参数 ----
    st.markdown("#### 权重与门控")
    st.caption("权重为 0 的模型不参与投票（也就不耗时）。门控是「宁可不做」的保险，任一不过就空仓。")

    cols = st.columns(3)
    new_weights: dict[str, float] = {}
    for i, m in enumerate(models):
        key = m.get("key")
        if not key:
            continue
        c = cols[i % 3]
        new_weights[key] = c.number_input(
            m.get("label") or key, 0.0, 10.0,
            float(weights.get(key, 0.0)), step=0.1,
            key=f"ens_w_{key}",
            disabled=not m.get("available"),
            help=(m.get("detail") or "") if not m.get("available") else None,
        )

    e1, e2, e3 = st.columns(3)
    min_abs = e1.number_input("最小预测幅度", 0.0, 0.5,
                              float(cfg.get("min_abs_return") or 0.0), step=0.001,
                              format="%.4f", key="ens_min_abs",
                              help="预测涨跌幅绝对值低于此值就直接空仓（绝对阈值，配合下面的波动率门控使用）")
    min_sig = e2.number_input("波动率倍数下限", 0.0, 10.0,
                              float(cfg.get("min_return_sigma") or 0.0), step=0.1,
                              key="ens_min_sig",
                              help="预测幅度至少要达到「近 60 根波动率 × 该倍数」。这是跨周期都能成立的门控")
    min_agr = e3.number_input("最少一致性", 0.0, 1.0,
                              float(cfg.get("min_agreement") or 0.0), step=0.05,
                              key="ens_min_agr",
                              help="同向模型的权重占比。0.6 表示至少 60% 的权重方向一致")

    e4, e5, e6 = st.columns(3)
    max_dis = e4.number_input("最大分歧度", 0.0, 20.0,
                              float(cfg.get("max_dispersion") or 0.0), step=0.5,
                              key="ens_max_dis",
                              help="各模型预测的加权标准差（相对波动率倍数）。分歧太大说明模型自己都没底")
    min_models = e5.number_input("最少有效模型数", 1, 9,
                                 int(cfg.get("min_models") or 1), step=1, key="ens_min_models",
                                 help="成功给出预测的模型数（权重>0 且无报错）少于此值就空仓")
    position = e6.number_input("目标仓位", 0.0, 1.0,
                               float(cfg.get("position") or 1.0), step=0.05, key="ens_pos",
                               help="全部门控通过后分配给该策略的资金比例")

    s1, s2 = st.columns([1, 3])
    allow_short = s1.checkbox("允许做空", value=bool(cfg.get("allow_short")), key="ens_short")
    if s2.button("保存集成配置", key="ens_save"):
        try:
            api.ml_ensemble_config_put(
                weights=new_weights, min_abs_return=float(min_abs),
                min_return_sigma=float(min_sig), min_agreement=float(min_agr),
                max_dispersion=float(max_dis), min_models=int(min_models),
                position=float(position), allow_short=bool(allow_short),
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"保存失败：{_err_detail(e)}")
        else:
            st.success("已保存。策略「多模型集成策略」下次回测会使用这套配置。")

    st.divider()

    # ---- 试预测 ----
    st.markdown("#### 集成试预测")
    p1, p2, p3 = st.columns([2, 1.4, 1])
    sym = p1.selectbox("标的", symbols, key="ens_sym")
    tf = p2.selectbox("周期", TIMEFRAMES,
                      index=TIMEFRAMES.index("5m") if "5m" in TIMEFRAMES else 0,
                      format_func=tf_label, key="ens_tf")
    bars = p3.number_input("上下文（根）", 40, 4000, 300, step=20, key="ens_bars")

    a1, a2 = st.columns([1, 3])
    if a1.button("用历史表现估权重", key="ens_auto",
                 help="滚动回看一段历史，按方向命中率和相关性给建议权重。不是严格的样本外评估，当初始猜测用"):
        with st.spinner("评估中..."):
            try:
                aw = api.ml_auto_weights(sym, timeframe=tf, eval_bars=240, step=20,
                                         context=240,
                                         keys=[k for k, v in new_weights.items() if v > 0])
            except Exception as e:  # noqa: BLE001
                st.error(f"评估失败：{_err_detail(e)}")
            else:
                st.session_state["ens_auto"] = aw

    aw = st.session_state.get("ens_auto")
    if aw:
        if not aw.get("ok"):
            st.warning(aw.get("error") or "样本不足，无法评估")
        else:
            detail = aw.get("detail") or {}
            st.dataframe(
                [{
                    "模型": k,
                    "方向命中率": f"{(v.get('hit') or 0):.0%}",
                    "相关性 IC": f"{(v.get('ic') or 0):+.3f}",
                    "得分": f"{(v.get('score') or 0):.2f}",
                    "建议权重": f"{(aw.get('weights') or {}).get(k, 0):.2f}",
                } for k, v in detail.items()],
                use_container_width=True, hide_index=True,
            )
            st.caption(f"评估点 {aw.get('eval_points')} 个 · 耗时 {aw.get('elapsed_seconds')}s · "
                       "得分 = max(0,IC) × max(0, 2×命中率-1) × 2。"
                       "负相关或命中率低于 50% 的模型直接给 0，而不是反向使用（负权重不稳定）。")

    if a2.button("开始集成预测", type="primary", key="ens_go"):
        with st.spinner("各模型投票中（首次会加载 Kronos 权重）..."):
            try:
                r = api.ml_ensemble_predict(sym, timeframe=tf, bars=int(bars),
                                            weights={k: v for k, v in new_weights.items() if v > 0})
            except Exception as e:  # noqa: BLE001
                st.error(f"预测失败：{_err_detail(e)}")
                st.session_state.pop("ens_result", None)
                return
        st.session_state["ens_result"] = r

    r = st.session_state.get("ens_result")
    if not r:
        return

    weight = r.get("target_weight") or 0.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("目标仓位", f"{weight:.2f}",
              delta="做多" if weight > 0 else ("做空" if weight < 0 else "空仓"),
              delta_color="normal" if weight != 0 else "off")
    m2.metric("加权预测幅度", f"{(r.get('expected_return') or 0):+.2%}")
    m3.metric("一致性 / 分歧", f"{(r.get('agreement') or 0):.0%} / "
                              f"{(r.get('dispersion') or 0):.4f}")
    m4.metric("耗时", f"{r.get('elapsed_seconds')}s")

    st.info(r.get("reason") or "")

    st.markdown("**各模型投票**")
    st.dataframe(
        [{
            "模型": v.get("label") or v.get("key"),
            "权重": v.get("weight"),
            "预测幅度": (f"{v['expected_return']:+.3%}"
                        if v.get("expected_return") is not None else "—"),
            "状态": "✅" if v.get("ok") else f"❌ {v.get('error') or '失败'}",
        } for v in (r.get("votes") or [])],
        use_container_width=True, hide_index=True,
    )

    rows = _gate_rows(r)
    if rows:
        st.markdown("**硬性门控**")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("任一门控不通过就空仓。空仓时看这张表能看出「差多少」，比只给一个仓位数字有用。")

    st.caption(f"{r.get('symbol')} · {tf_label(r.get('timeframe', ''))} · "
               f"{r.get('context_bars')} 根上下文 · 最新收盘 {r.get('last_close')}")

    st.divider()
    st.markdown("#### 把集成变成策略")
    st.caption(
        "策略「多模型集成策略」用的就是这里保存的权重与门控。回测时会在每根 K 线上"
        "重新投票（受 rebalance_every 控制频率，默认每 20 根一次），因此比单模型慢。"
    )
    if st.button("去回测这个策略", key="ens_bt_go"):
        st.session_state["nav_target"] = "策略回测"
        st.rerun()


# ==================== 入口 ====================

def render(api) -> None:
    inject_theme()
    st.markdown("## 模型预测")
    st.caption("单模型看细节、多模型看共识。这里只做研究，不写库、不下单。")

    view = st.segmented_control("模式", ["单模型（Kronos）", "多模型集成"],
                                key="mp_view", label_visibility="collapsed") or "单模型（Kronos）"

    symbols = _symbols(api)
    if not symbols:
        st.info("尚无本地行情数据，请先到「数据管理」页添加标的并下载。")
        return

    if view == "单模型（Kronos）":
        try:
            status = api.ml_kronos_status()
        except Exception as e:  # noqa: BLE001
            st.error(f"无法连接后端服务：{_err_detail(e)}")
            return
        if not status.get("available") or not status.get("enabled", True):
            _render_kronos_setup(status)
            return
        _render_single(api, status, symbols)
    else:
        _render_ensemble(api, symbols)
