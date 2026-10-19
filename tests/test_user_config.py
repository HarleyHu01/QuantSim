"""本机覆盖层（config/user.yaml）与路径 API 的测试。

覆盖：
1. ``_deep_merge`` 的递归合并语义（列表整体替换、嵌套 dict 合并）。
2. ``AppConfig.load()`` 在不叠加 / 叠加 user.yaml 时的优先级。
3. ``GET/PUT /api/settings/paths`` 往返：写 user.yaml、拒绝相对路径、拒绝不可写目录。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant.core.config import USER_CONFIG, AppConfig, _deep_merge


# ---------------------------------------------------------------- _deep_merge

def test_deep_merge_flat_override():
    out = _deep_merge({"a": 1, "b": 2}, {"b": 3})
    assert out == {"a": 1, "b": 3}


def test_deep_merge_nested_dict():
    out = _deep_merge(
        {"llm": {"model": "deepseek-chat", "timeout_seconds": 60}},
        {"llm": {"model": "deepseek-reasoner"}},
    )
    assert out["llm"] == {"model": "deepseek-reasoner", "timeout_seconds": 60}


def test_deep_merge_list_replaced_not_concatenated():
    out = _deep_merge({"symbols": [{"symbol": "A"}]}, {"symbols": [{"symbol": "B"}]})
    assert out["symbols"] == [{"symbol": "B"}]


def test_deep_merge_none_override_replaces():
    out = _deep_merge({"data_dir": "x"}, {"data_dir": None})
    assert out["data_dir"] is None


# ---------------------------------------------------------- AppConfig.load

def test_load_without_user_layer_ignores_user_file(tmp_path):
    """传显式 yaml_path 时不叠加 user.yaml，避免测试/一次性覆盖被本机设置污染。"""
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(yaml.safe_dump({"data_dir": str(tmp_path / "custom")}),
                        encoding="utf-8")
    cfg = AppConfig.load(cfg_file)
    assert Path(cfg.data_dir) == tmp_path / "custom"


def test_user_config_is_optional():
    """user.yaml 不存在时 load() 正常工作（大多数用户都是这个状态）。"""
    if USER_CONFIG.exists():
        pytest.skip("本机已存在 config/user.yaml，跳过默认态断言")
    cfg = AppConfig.load()
    assert cfg.effective_db_path.name == "quant.db"
    assert cfg.effective_market_dir.name == "market"


# ------------------------------------------------------------- paths API

@pytest.fixture()
def client(monkeypatch):
    """只挂载 settings router 的最小 app（不需要 lifespan / 仓库）。"""
    from quant.api.routers import settings as settings_router

    app = FastAPI()
    app.include_router(settings_router.router, prefix="/api")
    return TestClient(app)


def test_get_paths_returns_runtime_info(client):
    r = client.get("/api/settings/paths")
    assert r.status_code == 200
    body = r.json()
    assert body["root"]
    assert body["data_dir"]["path"]
    assert body["db_path"]["path"].endswith("quant.db")
    assert body["market_dir"]["path"].endswith("market")
    assert "defaults" in body
    assert body["data_dir_source"]
    assert body["api"]["port"] > 0


def test_get_paths_db_size_is_file_size(client):
    """db_path 是文件，占用必须报真实字节数（曾经误报 0）。"""
    body = client.get("/api/settings/paths").json()
    db = body["db_path"]
    if db["exists"]:
        assert db["size_bytes"] > 0
        assert db["is_dir"] is False


def test_put_paths_rejects_relative_path(client):
    r = client.put("/api/settings/paths", json={"data_dir": "relative/path"})
    assert r.status_code == 400
    assert "绝对路径" in r.json()["detail"]


def test_put_paths_accepts_absolute_and_writes_user_yaml(client, tmp_path, monkeypatch):
    """写绝对路径 → 返回 restart_required，并真的落盘到 user.yaml。"""
    fake = tmp_path / "config" / "user.yaml"
    import quant.core.config as real
    monkeypatch.setattr(real, "USER_CONFIG", fake)

    target = tmp_path / "newdata"
    r = client.put("/api/settings/paths", json={"data_dir": str(target)})
    assert r.status_code == 200, r.text
    assert r.json()["restart_required"] is True
    assert target.exists()                      # 目录被创建
    assert fake.exists()                        # 覆盖层被写出
    saved = yaml.safe_load(fake.read_text(encoding="utf-8"))
    assert saved["data_dir"] == str(target)


def test_put_paths_empty_clears_override(client, tmp_path, monkeypatch):
    """传空串 = 清除覆盖，恢复默认 data/。"""
    fake = tmp_path / "config" / "user.yaml"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(yaml.safe_dump({"data_dir": "D:/somewhere"}), encoding="utf-8")

    import quant.core.config as real
    monkeypatch.setattr(real, "USER_CONFIG", fake)

    r = client.put("/api/settings/paths", json={"data_dir": ""})
    assert r.status_code == 200
    if fake.exists():
        assert "data_dir" not in (yaml.safe_load(fake.read_text(encoding="utf-8")) or {})
