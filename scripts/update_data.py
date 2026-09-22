"""命令行一键更新行情数据。

用法示例:
  python scripts/update_data.py                        # 更新全部启用标的 × 默认周期
  python scripts/update_data.py -s AAPL,SPY -tf 1d     # 指定标的与周期
  python scripts/update_data.py -s BTC/USDT -tf 1h,4h  # 加密多周期
  python scripts/update_data.py --adjust none --full   # 不复权 + 全量重刷（清空旧数据）
  python scripts/update_data.py -s AAPL -tf 1d --start 2020-01-01   # 从指定日期补历史
  python scripts/update_data.py -s AAPL -tf 1d --start 2024-01-01 --end 2024-06-30  # 指定时间段
  python scripts/update_data.py -s AAPL -tf 1m --start 2026-08-14   # 分钟级数据（数据源限制近期）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.core.config import AppConfig  # noqa: E402
from quant.data.manager import DataManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="QuantSim 一键更新行情数据")
    ap.add_argument("-s", "--symbols", help="逗号分隔的标的；不传则更新全部启用标的")
    ap.add_argument("-tf", "--timeframes", help="逗号分隔的周期（1m/5m/15m/30m/1h/2h/4h/1d/1w/1M）；默认取配置")
    ap.add_argument("--adjust", choices=["qfq", "hfq", "none"], default=None, help="A股复权方式")
    ap.add_argument("--full", action="store_true",
                    help="全量重刷：清空本地该标的/周期的数据后重拉（复权方式改变时使用）")
    ap.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD，从该日期补拉历史（backfill）")
    ap.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）；与 --start 一起指定时间段")
    args = ap.parse_args()

    config = AppConfig.load()
    dm = DataManager.create(config)
    symbols = args.symbols.split(",") if args.symbols else None
    tfs = args.timeframes.split(",") if args.timeframes else None
    results = dm.update(
        symbols=symbols, timeframes=tfs,
        incremental=(not args.full), adjust=args.adjust,
        full_reload=args.full, start_date=args.start, end_date=args.end,
    )

    print(f"\n{'symbol':<14}{'tf':<6}{'source':<10}{'rows':>8}  {'覆盖区间':<42}{'状态'}")
    print("-" * 104)
    for r in results:
        interval = f"{r.start} → {r.end}" if r.ok else (r.error or "")
        status = "OK" if r.ok else "FAIL"
        print(f"{r.symbol:<14}{r.timeframe:<6}{(r.source or '-'):<10}{r.rows:>8}  {interval:<42}{status}")
    ok = sum(1 for r in results if r.ok)
    print("-" * 104)
    print(f"完成: {ok}/{len(results)} 成功")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
