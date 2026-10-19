"""WorkBuddy 接入 API：登录态 / 积分 / 模型列表。

路由（全部挂 /api/workbuddy）：
- GET  /workbuddy/status  登录态 + 积分 + 模型列表（一张卡片展示）
- GET  /workbuddy/models  模型列表
- POST /workbuddy/ping    连通性测试（发一条极短对话）
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from quant.llm import workbuddy as wb

router = APIRouter(prefix="/workbuddy", tags=["workbuddy"])


class WorkBuddyConfigIn(BaseModel):
    exe_path: str = ""


@router.get("/status")
def wb_status() -> dict[str, Any]:
    """前端初始化用：是否可用、账号、积分、模型列表。"""
    return wb.status()


@router.get("/config")
def wb_get_config() -> dict[str, Any]:
    """返回当前安装位置配置（自定义路径 + 实际生效路径）。"""
    return {
        "custom_exe_path": wb.get_exe_path(),
        "exe_path": wb.find_workbuddy_exe() or "",
    }


@router.post("/config")
def wb_set_config(body: WorkBuddyConfigIn) -> dict[str, Any]:
    """保存用户自定义的 WorkBuddy 安装位置（空串 = 清除，回退自动探测）。"""
    wb.set_exe_path(body.exe_path)
    return {
        "ok": True,
        "custom_exe_path": wb.get_exe_path(),
        "exe_path": wb.find_workbuddy_exe() or "",
    }


@router.post("/calibrate")
def wb_calibrate(body: WorkBuddyConfigIn) -> dict[str, Any]:
    """校准安装位置：用指定（或当前）路径跑一次取密钥，验证是否可用。"""
    return wb.calibrate(body.exe_path or None)


@router.get("/models")
def wb_models() -> dict[str, Any]:
    try:
        return {"models": wb.WorkBuddyClient().models()}
    except wb.WorkBuddyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/ping")
def wb_ping() -> dict[str, Any]:
    try:
        client = wb.WorkBuddyClient()
        # 用轻量模型，避免 auto 路由到推理模型返回空 content
        r = client.chat([{"role": "user", "content": "回复一个字：好"}],
                        model="glm-5.3-flash", max_tokens=16)
        return {"ok": True, "model": r["model"], "reply": r["content"].strip()[:50]}
    except wb.WorkBuddyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
