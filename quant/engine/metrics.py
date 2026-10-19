"""回测指标计算。

输入 equity DataFrame（列含 equity/cash/position_value/exposure/drawdown/benchmark，
可选 buyhold/symbol_price——等权买入持有权益与首标的收盘价）与成交列表。

对比指标约定：
- buyhold：等权买入持有全部交易标的的权益曲线（初始资金 × 各标的归一化 close 的均值）
- excess_return_vs_buyhold：策略总收益 − 买入持有总收益
- alpha：年化超额收益 = 策略年化 − beta × 基准年化（rf=0）
- beta：策略日收益对基准日收益的回归斜率（cov/var）
- information_ratio：策略与基准日收益差值的年化夏普
- benchmark_max_drawdown：基准曲线的最大回撤
"""
from __future__ import annotations

import math

import pandas as pd

from quant.core.models import Trade

# 各周期一年 bar 数（近似：1d=252 交易日，1w=52，1M=12；分钟级按美股 6.5h 交易日估算）
BARS_PER_YEAR = {
    "1m": 252 * 390, "5m": 252 * 78, "15m": 252 * 26, "30m": 252 * 13,
    "1h": 252 * 6.5,
    "1d": 252, "1w": 52, "1M": 12,
}


def periods_for(timeframe: str) -> int:
    return BARS_PER_YEAR.get(timeframe, 252)


def _series_annual_return(s: pd.Series, periods_per_year: int) -> float | None:
    """序列（金额/价格）的总收益 → 年化收益。"""
    if len(s) < 2 or s.iloc[0] == 0:
        return None
    total = float(s.iloc[-1] / s.iloc[0] - 1.0)
    if total <= -1.0:
        return None
    return float((1.0 + total) ** (periods_per_year / (len(s) - 1)) - 1.0)


def compute_metrics(
    equity: pd.DataFrame,
    trades: list[Trade],
    periods_per_year: int = 252,
    rf: float = 0.0,
) -> dict[str, float | None]:
    """从权益曲线与成交列表计算指标。"""
    if equity is None or equity.empty or len(equity) < 2:
        return {"error": "权益曲线数据不足"}

    eq = equity["equity"]
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    n = len(eq) - 1

    annual = None
    if total > -1.0 and n > 0:
        annual = float((1.0 + total) ** (periods_per_year / n) - 1.0)

    max_dd = float((1.0 - eq / eq.cummax()).max()) if "drawdown" not in equity.columns \
        else float(equity["drawdown"].max())

    rets = eq.pct_change().dropna()
    std = float(rets.std()) if len(rets) > 1 else 0.0
    sharpe = None
    if std > 0:
        sharpe = float((rets.mean() - rf / periods_per_year) / std * math.sqrt(periods_per_year))

    downside = rets[rets < 0]
    sortino = None
    if len(downside) > 1 and float(downside.std()) > 0:
        sortino = float(rets.mean() / downside.std() * math.sqrt(periods_per_year))

    calmar = None
    if annual is not None and max_dd and max_dd > 0:
        calmar = float(annual / max_dd)

    vol = None
    if std > 0:
        vol = float(std * math.sqrt(periods_per_year))

    closed = [t for t in trades if t.pnl != 0.0]
    win_rate = None
    if closed:
        win_rate = float(sum(1 for t in closed if t.pnl > 0) / len(closed))

    gross_profit = sum(t.pnl for t in closed if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in closed if t.pnl < 0))
    profit_factor = None
    if gross_loss > 0:
        profit_factor = float(gross_profit / gross_loss)

    avg_trade = float(sum(t.pnl for t in trades) / len(trades)) if trades else None

    avg_exposure = None
    if "exposure" in equity.columns:
        avg_exposure = float(equity["exposure"].mean())

    # ---- 买入持有（等权，来自 equity["buyhold"]）----
    buyhold_return = buyhold_annual = None
    if "buyhold" in equity.columns:
        bh = equity["buyhold"].dropna()
        if len(bh) >= 2 and bh.iloc[0] > 0:
            buyhold_return = float(bh.iloc[-1] / bh.iloc[0] - 1.0)
            buyhold_annual = _series_annual_return(bh, periods_per_year)
    excess_return_vs_buyhold = total - buyhold_return if buyhold_return is not None else None

    # ---- 基准（benchmark 金额序列）----
    benchmark_return = benchmark_annual = benchmark_max_drawdown = None
    beta = alpha = information_ratio = None
    if "benchmark" in equity.columns and equity["benchmark"].notna().any():
        b = equity["benchmark"].dropna()
        if len(b) >= 2 and b.iloc[0] != 0:
            benchmark_return = float(b.iloc[-1] / b.iloc[0] - 1.0)
            benchmark_annual = _series_annual_return(b, periods_per_year)
            benchmark_max_drawdown = float((1.0 - b / b.cummax()).max())
            # Beta / Alpha / 信息比率：策略收益 vs 基准收益（共同区间）
            brets = b.pct_change().dropna()
            j = pd.concat([rets, brets], axis=1, keys=["s", "b"]).dropna()
            if len(j) >= 3 and float(j["b"].var()) > 0:
                beta = float(j["s"].cov(j["b"]) / j["b"].var())
                if annual is not None and benchmark_annual is not None:
                    alpha = float(annual - beta * benchmark_annual)  # rf=0
                diff = j["s"] - j["b"]
                if float(diff.std()) > 0:
                    information_ratio = float(
                        diff.mean() / diff.std() * math.sqrt(periods_per_year))

    return {
        "total_return": round(total, 6),
        "annual_return": round(annual, 6) if annual is not None else None,
        "max_drawdown": round(max_dd, 6),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "sortino": round(sortino, 4) if sortino is not None else None,
        "calmar": round(calmar, 4) if calmar is not None else None,
        "volatility": round(vol, 6) if vol is not None else None,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "trade_count": len(trades),
        "avg_trade_pnl": round(avg_trade, 4) if avg_trade is not None else None,
        "avg_exposure": round(avg_exposure, 4) if avg_exposure is not None else None,
        # 对比指标
        "benchmark_return": round(benchmark_return, 6) if benchmark_return is not None else None,
        "benchmark_annual": round(benchmark_annual, 6) if benchmark_annual is not None else None,
        "benchmark_max_drawdown": round(benchmark_max_drawdown, 6)
        if benchmark_max_drawdown is not None else None,
        "buyhold_return": round(buyhold_return, 6) if buyhold_return is not None else None,
        "buyhold_annual": round(buyhold_annual, 6) if buyhold_annual is not None else None,
        "excess_return_vs_buyhold": round(excess_return_vs_buyhold, 6)
        if excess_return_vs_buyhold is not None else None,
        "beta": round(beta, 4) if beta is not None else None,
        "alpha": round(alpha, 6) if alpha is not None else None,
        "information_ratio": round(information_ratio, 4) if information_ratio is not None else None,
        "periods_per_year": periods_per_year,
    }
