"""命令行模拟盘：start（前台常驻）/ stop / status。

用法:
  python scripts/run_paper.py status
  python scripts/run_paper.py start --account 1 --symbols AAPL --timeframe 1d --strategy dual_ma
  python scripts/run_paper.py start --symbols BTC/USDT --timeframe 1m --strategy dual_ma --poll 65
  python scripts/run_paper.py stop --account 1

start 为前台常驻进程（Ctrl+C 或另一进程 stop 后优雅退出）；
也可用 API 服务（uvicorn）驱动同一套 PaperTradingEngine，二者选一。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # noqa: E402

from quant.core.config import AppConfig  # noqa: E402
from quant.data.manager import DataManager  # noqa: E402
from quant.engine.paper import PaperTradingEngine  # noqa: E402
from quant.storage.db import init_db  # noqa: E402
from quant.storage.repositories import PaperRepo, SymbolsRepo  # noqa: E402

_stop_event = asyncio.Event()


async def _run_forever(engine: PaperTradingEngine, account_id: int) -> None:
    """常驻循环：每 5s 检查账户状态，被 stop 命令置 stopped 后退出。"""
    while not _stop_event.is_set():
        await asyncio.sleep(5)
        account = engine.repo.get_account(account_id)
        if account is None or account["status"] != "running":
            print(f"账户 {account_id} 状态={account['status'] if account else '不存在'}，模拟盘退出")
            engine.stop(account_id)
            break


def cmd_start(args) -> int:
    config = AppConfig.load()
    init_db(config.effective_db_path)
    dm = DataManager.create(config)
    scheduler = AsyncIOScheduler()
    scheduler.start()
    engine = PaperTradingEngine(config, dm, scheduler, PaperRepo(config.effective_db_path),
                                SymbolsRepo(config.effective_db_path))

    if args.account is None:
        settings = json.loads(args.settings) if args.settings else {}
        capital = args.capital or settings.get("initial_capital", 1_000_000.0)
        account_id = engine.repo.create_account(args.name, float(capital), settings)
        print(f"新建账户 {account_id}（资金 {capital}）")
    else:
        account_id = args.account

    st = engine.start(
        account_id, args.symbols.split(","), args.timeframe,
        args.strategy.split(","),
        params=json.loads(args.params) if args.params else None,
        settings=json.loads(args.settings) if args.settings else None,
        portfolio=json.loads(args.portfolio) if args.portfolio else None,
        poll_seconds=args.poll,
    )
    print(json.dumps({"account_id": account_id, "status": st["status"], "jobs": st["jobs"]},
                     ensure_ascii=False, indent=2))
    print("模拟盘运行中（Ctrl+C 停止，或另一终端执行 stop）...")

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop_event.set)
        except NotImplementedError:  # Windows
            pass
    try:
        asyncio.run(_run_forever(engine, account_id))
    except KeyboardInterrupt:
        pass
    engine.stop(account_id)
    scheduler.shutdown(wait=False)
    return 0


def cmd_stop(args) -> int:
    config = AppConfig.load()
    init_db(config.effective_db_path)
    engine = PaperTradingEngine(config, None, AsyncIOScheduler(),
                                PaperRepo(config.effective_db_path),
                                SymbolsRepo(config.effective_db_path))
    # 无常驻进程时 stop 只置状态（常驻进程轮询到后自行退出）
    repo = engine.repo
    account = repo.get_account(args.account)
    if account is None:
        print(f"账户不存在: {args.account}")
        return 1
    repo.set_status(args.account, "stopped")
    print(f"账户 {args.account} 已标记 stopped（若由本 CLI start，进程将在 5 秒内退出）")
    return 0


def cmd_status(args) -> int:
    config = AppConfig.load()
    init_db(config.effective_db_path)
    repo = PaperRepo(config.effective_db_path)
    accounts = repo.list_accounts()
    if args.account is None:
        for a in accounts:
            eq = repo.list_equity(a["id"], limit=1)
            print(f"  #{a['id']:<4} {a['name']:<12} 状态={a['status']:<8} "
                  f"资金={a['initial_capital']:>12.0f} 最新净值={eq[0]['equity'] if eq else '-'}")
        return 0
    a = repo.get_account(args.account)
    if a is None:
        print(f"账户不存在: {args.account}")
        return 1
    print(json.dumps(a, ensure_ascii=False, indent=2))
    print("持仓:", json.dumps(repo.list_positions(args.account), ensure_ascii=False))
    print("最近成交:", json.dumps(repo.list_trades(args.account, limit=5), ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="QuantSim 命令行模拟盘")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="前台常驻运行模拟盘")
    p_start.add_argument("--account", type=int, default=None)
    p_start.add_argument("--name", default="模拟盘")
    p_start.add_argument("--capital", type=float, default=None)
    p_start.add_argument("-s", "--symbols", required=True)
    p_start.add_argument("-tf", "--timeframe", default="1d")
    p_start.add_argument("-st", "--strategy", required=True, help="逗号分隔策略 key")
    p_start.add_argument("--params", default=None, help='策略参数 JSON')
    p_start.add_argument("--settings", default=None, help='交易设置 JSON')
    p_start.add_argument("--portfolio", default=None, help='组合配置 JSON')
    p_start.add_argument("--poll", type=int, default=None, help="轮询间隔秒数（覆盖默认）")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="停止模拟盘")
    p_stop.add_argument("--account", type=int, required=True)
    p_stop.set_defaults(func=cmd_stop)

    p_status = sub.add_parser("status", help="查询状态")
    p_status.add_argument("--account", type=int, default=None)
    p_status.set_defaults(func=cmd_status)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
