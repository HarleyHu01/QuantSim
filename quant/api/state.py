"""应用共享状态（避免 routers 与 app 循环导入）。"""
from __future__ import annotations

from typing import Any

app_state: dict[str, Any] = {}
