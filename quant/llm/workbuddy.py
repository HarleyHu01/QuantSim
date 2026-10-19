"""WorkBuddy 模型 Provider：复用本机已登录的 WorkBuddy 桌面 App 登录态，
直连 WorkBuddy 私有上游 API，把 WorkBuddy 的模型和积分「搬」进 QuantSim。

背景：dsh-workbuddy-connect 插件（github.com/corrinehu/dsh-workbuddy-connect）
用同样的方式接入 DSH。这里按该插件的协议独立实现，仅复用其逆向结论：

1. 凭据文件：%LOCALAPPDATA%\\CodeBuddyExtension\\Data\\Public\\auth\\workbuddy-desktop.info
   顶层含 auth（accessToken/refreshToken/domain/…）与 account（uid/uin/…）。
2. WorkBuddy 5.6+ 把 auth.accessToken / auth.refreshToken 加密成
   ``{$wbEncrypted: 1, envelope: <base64>}``；envelope 解码为
   ``{suite, keyId, nonce, authTag, ciphertext}``，AES-256-GCM 加密。
3. 密钥：运行 WorkBuddy 自己的 Electron 二进制（ELECTRON_RUN_AS_NODE=1）执行
   ``process._linkedBinding("electron_browser_workbuddy_storage").loggerGet()``
   得到 ``{version:1, atRestSecretKey:<base64 32B>}``；
   protectorKey = sha256(atRestSecretKey 字符串的 UTF-8 字节)。
4. AAD 见 ``_build_aad``（转写自插件 buildAuthenticatedContextAad）。
5. 上游端点：国内 chat=copilot.tencent.com、billing=codebuddy.cn；
   国际 chat/billing=workbuddy.ai。region 由 auth.domain 判定。
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import struct
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import httpx

log = logging.getLogger(__name__)

# ---- 端点与常量 ----
CN_CHAT_BASE = "https://copilot.tencent.com"
CN_BILLING_BASE = "https://www.codebuddy.cn"
GLOBAL_BASE = "https://www.workbuddy.ai"

AUTH_FILE = os.path.expandvars(
    r"%LOCALAPPDATA%\CodeBuddyExtension\Data\Public\auth\workbuddy-desktop.info")

CLIENT_UA = "CLI/2.63.2 CodeBuddy/2.63.2"
IDE_VERSION = "5.8.3"

# kv_store 里存放「用户自定义 WorkBuddy 安装位置」的键
EXE_PATH_KEY = "workbuddy.exe_path"

# 常见安装位置（按优先级探测 WorkBuddy.exe，找不到时置空 → status 报「未检测到」）
# 说明：这里全部用环境变量/盘符展开，**不放任何某台机器专属的绝对路径**；
# 非常规安装位置请由用户在 UI 里填（写 kv_store 的 workbuddy.exe_path），
# 或用环境变量 WORKBUDDY_EXE 覆盖。
def _drive_candidates() -> list[str]:
    """在所有本地固定盘符下探测常见安装目录（不写死任何具体盘符）。

    很多用户会把 WorkBuddy 装到 D:\\\\Insert\\\\WorkBuddy 这类非系统盘目录，
    只探 %LOCALAPPDATA% 必然报「未检测到」。这里遍历系统报告的固定驱动器，
    按下面几个常见相对路径找；找不到也不会报错。
    """
    out: list[str] = []
    try:
        import string

        for letter in string.ascii_uppercase:
            root = f"{letter}:\\"
            if not os.path.isdir(root):
                continue
            for rel in (r"WorkBuddy\WorkBuddy.exe",
                        r"Programs\WorkBuddy\WorkBuddy.exe",
                        r"Program Files\WorkBuddy\WorkBuddy.exe",
                        r"Insert\WorkBuddy\WorkBuddy.exe",
                        r"Apps\WorkBuddy\WorkBuddy.exe",
                        r"Software\WorkBuddy\WorkBuddy.exe"):
                out.append(os.path.join(root, rel))
    except Exception:  # noqa: BLE001  pragma: no cover
        pass
    return out


_EXE_CANDIDATES = [
    # 1) 环境变量显式指定（最高优先，便于绿色版/便携部署）
    os.path.expandvars(r"%WORKBUDDY_EXE%") if os.getenv("WORKBUDDY_EXE") else "",
    # 2) 当前用户的默认安装位置（官方安装器默认路径）
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\WorkBuddy\WorkBuddy.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\workbuddy\workbuddy.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\WorkBuddy\WorkBuddy.exe"),
    # 3) 系统级安装（管理员安装时）
    os.path.expandvars(r"%PROGRAMFILES%\WorkBuddy\WorkBuddy.exe"),
    os.path.expandvars(r"%PROGRAMFILES(X86)%\WorkBuddy\WorkBuddy.exe"),
] + _drive_candidates()


def _kv() -> "Any":
    """懒加载 KVRepo（与 api 层同一套 kv_store 表）。"""
    from quant.storage.repositories import KVRepo
    from quant.core.config import AppConfig

    return KVRepo(AppConfig.load().effective_db_path)


def get_exe_path() -> str:
    """返回用户自定义的 WorkBuddy 安装位置（可能为空串）。"""
    v = _kv().get(EXE_PATH_KEY, "")
    return v if isinstance(v, str) else ""


def set_exe_path(path: str) -> None:
    """保存用户自定义的 WorkBuddy 安装位置（空串 = 清除，回退自动探测）。"""
    p = (path or "").strip().strip('"')
    if p:
        _kv().put(EXE_PATH_KEY, p)
    else:
        _kv().delete(EXE_PATH_KEY)


class WorkBuddyError(RuntimeError):
    """WorkBuddy 调用失败（消息为可直接展示给用户的中文）。"""


# ====================== 凭据读取与解密 ======================

def find_workbuddy_exe() -> str | None:
    """返回本机 WorkBuddy 可执行文件路径；找不到返回 None。

    优先级：用户自定义路径（kv_store 的 ``workbuddy.exe_path``）> 常见安装位置。
    """
    custom = get_exe_path()
    if custom and Path(custom).exists():
        return custom
    for p in _EXE_CANDIDATES:
        if p and Path(p).exists():
            return p
    return None


def fetch_at_rest_secret(exe_path: str | None = None) -> str:
    """运行 WorkBuddy 自己的 Electron 取 atRestSecretKey（base64 32 字节）。

    通过 ``ELECTRON_RUN_AS_NODE=1`` 让 Electron 进入纯 Node 模式，
    执行其私有 ``electron_browser_workbuddy_storage`` binding 的 loggerGet()。
    """
    exe = exe_path or find_workbuddy_exe()
    if not exe or not Path(exe).exists():
        raise WorkBuddyError("未检测到 WorkBuddy 桌面客户端（WorkBuddy.exe）")
    script = ('process.stdout.write(String(process._linkedBinding'
              '("electron_browser_workbuddy_storage").loggerGet()))')
    env = dict(os.environ)
    env["ELECTRON_RUN_AS_NODE"] = "1"
    try:
        proc = subprocess.run(
            [exe, "-e", script], capture_output=True, timeout=30,
            env=env, check=False, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as e:
        raise WorkBuddyError(f"无法启动 WorkBuddy 客户端：{e}") from e
    except subprocess.TimeoutExpired as e:
        raise WorkBuddyError("获取 WorkBuddy 密钥超时（30s）") from e
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not out:
        err = (proc.stderr or "").strip()
        raise WorkBuddyError(f"获取 WorkBuddy 密钥失败（exit {proc.returncode}）"
                             + (f"：{err[:200]}" if err else ""))
    try:
        payload = json.loads(out)
    except json.JSONDecodeError as e:
        raise WorkBuddyError("WorkBuddy 密钥返回格式异常") from e
    secret = payload.get("atRestSecretKey") if isinstance(payload, dict) else None
    if not isinstance(secret, str) or len(base64.b64decode(secret)) != 32:
        raise WorkBuddyError("WorkBuddy 密钥内容非法")
    return secret


def _build_aad(key_id: str, suite: int) -> bytes:
    """转写自 WorkBuddy buildAuthenticatedContextAad。"""

    def _lp(v: str) -> bytes:
        b = v.encode("utf-8")
        return struct.pack(">I", len(b)) + b

    return (b"WB-AAD\x00" + b"\x01" + _lp("WBEV1") + _lp("sym-v1")
            + struct.pack(">I", suite) + _lp(key_id) + b"\x02\x00\x00")


def _decrypt_field(secret: str, wrapped: dict) -> str:
    """解密单个 {$wbEncrypted:1, envelope} 字段。"""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    env_b64 = wrapped.get("envelope")
    if not env_b64:
        raise WorkBuddyError("凭据信封缺失")
    try:
        env = json.loads(base64.b64decode(env_b64).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise WorkBuddyError("凭据信封解码失败") from e
    key = hashlib.sha256(secret.encode("utf-8")).digest()
    nonce = base64.b64decode(env["nonce"])
    ct = base64.b64decode(env["ciphertext"]) + base64.b64decode(env["authTag"])
    aad = _build_aad(env["keyId"], int(env["suite"]))
    try:
        return AESGCM(key).decrypt(nonce, ct, aad).decode("utf-8")
    except Exception as e:  # noqa: BLE001
        raise WorkBuddyError("凭据解密失败（AES-GCM 校验不通过）") from e


def _read_auth_doc() -> dict:
    if not os.path.exists(AUTH_FILE):
        raise WorkBuddyError("未找到 WorkBuddy 登录凭据（请先登录 WorkBuddy 桌面客户端）")
    try:
        with open(AUTH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        raise WorkBuddyError(f"读取 WorkBuddy 凭据失败：{e}") from e


@dataclass
class WorkBuddyCredential:
    """解密后的 WorkBuddy 登录态。"""
    access_token: str
    refresh_token: str
    uid: str = ""
    domain: str = ""
    nickname: str = ""
    enterprise_id: str = ""
    expires_at: str = ""

    @property
    def region(self) -> str:
        d = (self.domain or "").lower()
        return "global" if (d == "workbuddy.ai" or d.endswith(".workbuddy.ai")) else "cn"

    @property
    def chat_base(self) -> str:
        return GLOBAL_BASE if self.region == "global" else CN_CHAT_BASE

    @property
    def billing_base(self) -> str:
        return GLOBAL_BASE if self.region == "global" else CN_BILLING_BASE


def load_credential(secret: str | None = None) -> WorkBuddyCredential:
    """读取并解密本机 WorkBuddy 凭据。

    secret 用于缓存；首次调用会运行 Electron 取密钥（约 1~2s）。
    """
    doc = _read_auth_doc()
    auth = doc.get("auth") or {}
    account = doc.get("account") or {}
    secret_holder: list[str | None] = [secret]

    def _field(name: str) -> str:
        v = auth.get(name)
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, dict) and v.get("$wbEncrypted") == 1:
            if secret_holder[0] is None:
                secret_holder[0] = fetch_at_rest_secret()
            return _decrypt_field(secret_holder[0], v)
        return ""

    def _decrypt_any(v):
        """account 里昵称等字段也可能是加密信封，统一解。"""
        if isinstance(v, dict) and v.get("$wbEncrypted") == 1:
            if secret_holder[0] is None:
                secret_holder[0] = fetch_at_rest_secret()
            return _decrypt_field(secret_holder[0], v)
        return v

    return WorkBuddyCredential(
        access_token=_field("accessToken"),
        refresh_token=_field("refreshToken"),
        uid=str(_decrypt_any(account.get("uid")) or ""),
        domain=str(auth.get("domain") or ""),
        nickname=str(_decrypt_any(account.get("nickname")) or ""),
        enterprise_id=str(_decrypt_any(account.get("enterpriseId")) or ""),
        expires_at=str(auth.get("expiresAt") or ""),
    )


# ====================== 上游客户端 ======================

def _common_headers(cred: WorkBuddyCredential, base: str) -> dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": base,
        "Referer": base + "/",
        "User-Agent": CLIENT_UA,
    }


def _identity_headers(cred: WorkBuddyCredential) -> dict[str, str]:
    h = {"Authorization": f"Bearer {cred.access_token}"}
    if cred.uid:
        h["X-User-Id"] = cred.uid
    if cred.enterprise_id:
        h["X-Enterprise-Id"] = cred.enterprise_id
        h["X-Tenant-Id"] = cred.enterprise_id
    if cred.domain:
        h["X-Domain"] = cred.domain
    return h


class WorkBuddyClient:
    """直连 WorkBuddy 上游 API 的最小客户端（chat 流式 / 模型目录 / 积分）。"""

    def __init__(self, credential: WorkBuddyCredential | None = None,
                 timeout: float = 120.0):
        self.credential = credential or load_credential()
        self.timeout = timeout

    # ---- 模型目录 ----
    def models(self) -> list[dict[str, Any]]:
        base = self.credential.chat_base
        headers = {**_common_headers(self.credential, base),
                   **_identity_headers(self.credential)}
        url = f"{base}/console/enterprises/personal/models"
        try:
            with httpx.Client(timeout=self.timeout, verify=False) as c:
                r = c.get(url, headers=headers)
        except httpx.HTTPError as e:
            raise WorkBuddyError(f"获取 WorkBuddy 模型列表失败（网络）：{e}") from e
        if r.status_code != 200:
            raise WorkBuddyError(f"获取模型列表失败（HTTP {r.status_code}）")
        try:
            data = r.json()
        except Exception as e:  # noqa: BLE001
            raise WorkBuddyError("模型列表返回格式异常") from e
        return self._parse_models(data)

    @staticmethod
    def _parse_models(data: Any) -> list[dict[str, Any]]:
        # 实测国内版：data.models[]（id/name/credits/descriptionZh/maxInputTokens/…）
        try:
            inner = data
            for key in ("data", "Response", "Data"):
                if isinstance(inner, dict) and key in inner:
                    inner = inner[key]
        except Exception:  # noqa: BLE001
            inner = data
        items: Any = []
        if isinstance(inner, dict):
            items = inner.get("models") or inner.get("ModelList") or []
        if not isinstance(items, list):
            return []
        out = []
        for m in items:
            if not isinstance(m, dict):
                continue
            mid = m.get("id") or m.get("model") or m.get("ModelId") or ""
            if not mid:
                continue
            name = m.get("name") or m.get("ModelName") or str(mid)
            reasoning = m.get("reasoning") if isinstance(m.get("reasoning"), dict) else {}
            out.append({
                "id": str(mid),
                "name": str(name),
                "credits": m.get("credits"),  # 形如 "x0.05" 或 None（auto）
                "description": m.get("descriptionZh") or m.get("description") or "",
                "max_input_tokens": m.get("maxInputTokens"),
                "max_output_tokens": m.get("maxOutputTokens"),
                "is_default": bool(m.get("isDefault")),
                "supports_reasoning": bool(m.get("supportsReasoning")),
                "only_reasoning": bool(m.get("onlyReasoning")),
                # 推理等级：默认档 + 可选档（含 high/max 等）
                "reasoning_effort": reasoning.get("defaultEffort")
                                   or reasoning.get("effort") or None,
                "reasoning_levels": reasoning.get("supportedEfforts")
                                    or ([] if not reasoning else None),
            })
        return out

    # ---- 积分余额 ----
    def credits(self) -> dict[str, Any]:
        base = self.credential.billing_base
        headers = {**_common_headers(self.credential, base),
                   **_identity_headers(self.credential),
                   "Accept": "application/json",
                   "Content-Type": "application/json"}
        url = f"{base}/v2/billing/meter/get-user-resource"
        now = datetime.now()

        def _fmt(d: datetime) -> str:
            return d.strftime("%Y-%m-%d %H:%M:%S")

        body = {
            "PageNumber": 1, "PageSize": 100, "ProductCode": "p_tcaca",
            "Status": [0, 3],
            "PackageEndTimeRangeBegin": _fmt(now),
            "PackageEndTimeRangeEnd": _fmt(now + timedelta(days=36865)),
        }
        try:
            with httpx.Client(timeout=self.timeout, verify=False) as c:
                r = c.post(url, headers=headers, json=body)
        except httpx.HTTPError as e:
            raise WorkBuddyError(f"获取积分余额失败（网络）：{e}") from e
        if r.status_code != 200:
            raise WorkBuddyError(f"获取积分余额失败（HTTP {r.status_code}）")
        try:
            data = r.json()
        except Exception as e:  # noqa: BLE001
            raise WorkBuddyError("积分余额返回格式异常") from e
        return self._parse_credits(data)

    @staticmethod
    def _parse_credits(data: Any) -> dict[str, Any]:
        inner = data
        for key in ("data", "Response", "Data"):
            if isinstance(inner, dict) and key in inner:
                inner = inner[key]
        if not isinstance(inner, dict):
            return {"total": 0, "accounts": []}
        accounts = inner.get("Accounts") or []
        total = inner.get("TotalDosage") or 0
        out = []
        for a in accounts or []:
            if not isinstance(a, dict):
                continue
            remain = a.get("CapacityRemain")
            size = a.get("CapacitySize")
            out.append({
                "name": a.get("PackageName") or a.get("ProductName") or "",
                "remain": remain,
                "size": size,
                "used": a.get("CapacityUsed"),
                "expire": a.get("CycleEndTime") or "",
                "unit": a.get("CapacityUnit") or "credits",
            })
        return {"total": total, "accounts": out}

    # ---- chat（SSE 流式）----
    def chat_stream(self, messages: list[dict[str, str]], model: str,
                    max_tokens: int | None = None,
                    temperature: float | None = None,
                    json_mode: bool = False) -> Iterator[dict[str, Any]]:
        """流式对话，逐段 yield 统一格式的 chunk。

        每段 chunk = {"delta": str, "reasoning": str, "done": bool,
                       "usage": dict|None, "model": str}。
        """
        base = self.credential.chat_base
        headers = {**_common_headers(self.credential, base),
                   **_identity_headers(self.credential),
                   "X-IDE-Type": "WorkBuddy",
                   "X-IDE-Name": "WorkBuddy",
                   "X-IDE-Version": IDE_VERSION,
                   "X-Product": "SaaS",
                   "X-Conversation-ID": str(uuid.uuid4()),
                   "X-Conversation-Request-ID": str(uuid.uuid4()),
                   "X-Conversation-Message-ID": str(uuid.uuid4())}
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        url = f"{base}/v2/chat/completions"
        try:
            with httpx.Client(timeout=self.timeout, verify=False) as c:
                with c.stream("POST", url, headers=headers, json=payload) as r:
                    if r.status_code != 200:
                        raise WorkBuddyError(
                            self._chat_http_error(r.status_code, r.read().decode("utf-8", "replace")))
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        for item in self._parse_chunk(chunk, model):
                            yield item
        except httpx.HTTPError as e:
            raise WorkBuddyError(f"对话请求失败（网络）：{e}") from e

    @staticmethod
    def _parse_chunk(chunk: dict, default_model: str) -> Iterator[dict[str, Any]]:
        choices = chunk.get("choices") or []
        if not choices:
            return
        delta = (choices[0].get("delta") or {}) if isinstance(choices[0], dict) else {}
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or ""
        finish = (choices[0].get("finish_reason") if isinstance(choices[0], dict) else None)
        usage = chunk.get("usage")
        yield {
            "delta": content,
            "reasoning": reasoning,
            "done": finish is not None and finish != "",
            "usage": usage,
            "model": chunk.get("model") or default_model,
        }

    def chat(self, messages: list[dict[str, str]], model: str,
             max_tokens: int | None = None, temperature: float | None = None,
             json_mode: bool = False) -> dict[str, Any]:
        """非流式对话：聚合所有 chunk 返回 {content, model, usage}。"""
        content, usage, model_used = "", None, model
        for ch in self.chat_stream(messages, model, max_tokens, temperature, json_mode):
            content += ch["delta"]
            if ch["usage"]:
                usage = ch["usage"]
            model_used = ch["model"]
        return {"content": content, "model": model_used, "usage": usage or {}}

    @staticmethod
    def _chat_http_error(status: int, body: str) -> str:
        if status == 401:
            return "WorkBuddy 登录已过期，请重新登录 WorkBuddy 桌面客户端"
        if status == 402:
            return "WorkBuddy 积分不足，请充值或切换模型"
        if status == 429:
            return "WorkBuddy 请求过于频繁（限流），请稍后再试"
        hint = body[:300]
        return f"WorkBuddy 调用失败（HTTP {status}）" + (f"：{hint}" if hint else "")


# ====================== 门面：对外统一入口 ======================

def get_client() -> WorkBuddyClient:
    return WorkBuddyClient()


def status() -> dict[str, Any]:
    """组合登录态 + 积分 + 模型列表，供 UI 一张卡片展示。"""
    exe = find_workbuddy_exe()
    out: dict[str, Any] = {
        "available": False,
        "exe_found": exe is not None,
        "exe_path": exe or "",
        "custom_exe_path": get_exe_path(),
        "auth_file": os.path.exists(AUTH_FILE),
        "nickname": "",
        "uid": "",
        "domain": "",
        "region": "",
        "credits": None,
        "models": [],
        "error": None,
    }
    try:
        cred = load_credential()
    except WorkBuddyError as e:
        out["error"] = str(e)
        return out
    out["available"] = bool(cred.access_token)
    out["nickname"] = cred.nickname
    out["uid"] = cred.uid
    out["domain"] = cred.domain
    out["region"] = cred.region
    if not cred.access_token:
        out["error"] = "凭据为空（未登录或 token 解析失败）"
        return out
    client = WorkBuddyClient(cred)
    try:
        out["credits"] = client.credits()
    except WorkBuddyError as e:
        out["error"] = f"积分读取失败：{e}"
    try:
        out["models"] = client.models()
    except WorkBuddyError as e:
        if not out["error"]:
            out["error"] = f"模型列表读取失败：{e}"
    return out


def calibrate(exe_path: str | None = None) -> dict[str, Any]:
    """校验指定（或当前）WorkBuddy 安装位置是否能取出密钥。

    返回 {ok, exe_path, exists, message}，用于前端「校准安装位置」。
    """
    path = (exe_path or get_exe_path() or find_workbuddy_exe()) or ""
    if not path:
        return {"ok": False, "exe_path": "", "exists": False,
                "message": "未找到 WorkBuddy 安装位置，请手动填写 WorkBuddy.exe 的完整路径"}
    if not Path(path).exists():
        return {"ok": False, "exe_path": path, "exists": False,
                "message": f"路径不存在：{path}"}
    try:
        fetch_at_rest_secret(path)
    except WorkBuddyError as e:
        return {"ok": False, "exe_path": path, "exists": True,
                "message": f"能启动但取密钥失败：{e}"}
    return {"ok": True, "exe_path": path, "exists": True,
            "message": "校准成功：已能读取 WorkBuddy 登录密钥"}
