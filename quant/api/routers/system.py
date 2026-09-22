"""系统控制 API：一键关闭（停止模拟盘账户 → 保存配置 → 关闭后端进程）。

- 停止所有运行中的模拟盘账户（PaperTradingEngine.stop，落库）
- 配置均为即时持久化（settings/portfolio/strategies 每次保存即写库），无需额外落盘
- 后台线程先尝试优雅退出（SIGINT，uvicorn 触发 lifespan shutdown），超时后兜底强制退出
"""
from __future__ import annotations

import os
import signal
import threading
import time

from fastapi import APIRouter

from quant.api.state import app_state

router = APIRouter(prefix="/system", tags=["system"])


def _terminate() -> None:
    """关闭 uvicorn 进程：优先优雅退出，1.5s 未退出则强制。"""
    time.sleep(0.5)
    try:
        signal.raise_signal(signal.SIGINT)  # uvicorn 优雅退出（POSIX 生效）
    except Exception:  # noqa: BLE001
        pass
    time.sleep(1.5)
    try:
        os.kill(os.getpid(), signal.SIGTERM)  # Windows：TerminateProcess
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)  # 兜底


@router.post("/shutdown")
def shutdown() -> dict:
    """一键关闭系统：停模拟盘账户（落库）→ 返回后由后台线程关闭进程。"""
    stopped: list[int] = []
    eng = app_state.get("paper_engine")
    if eng is not None:
        for aid in list(getattr(eng, "_state", {}).keys()):
            try:
                eng.stop(aid)
                stopped.append(aid)
            except Exception:  # noqa: BLE001
                pass
    if app_state.get("shutdown_real", True):
        threading.Thread(target=_terminate, daemon=True).start()
    return {"ok": True, "stopped_accounts": stopped, "message": "系统正在关闭..."}
