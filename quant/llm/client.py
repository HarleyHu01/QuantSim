"""OpenAI 兼容 Chat Completions 客户端（DeepSeek 等）。

设计要点（对应 docs/ARCHITECTURE.md §10）：
- httpx 直连，不引入官方 SDK
- 连接/读超时可配置（默认 60s）
- 指数退避重试（默认 3 次）：429 / 5xx / 超时 / 网络错误重试；401 直接失败并提示检查 key
- 所有错误转成中文可读消息，供 UI 直接展示
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from quant.core.config import AppConfig, LLMConfig

log = logging.getLogger(__name__)

# 支持的 API 协议：key -> 展示名（UI 下拉与客户端共用）
PROTOCOLS = {
    "openai": "OpenAI 兼容（DeepSeek/Qwen/Moonshot/GLM/OpenAI）",
    "anthropic": "Anthropic（Claude）",
    "gemini": "Google Gemini",
}

# 可重试的 HTTP 状态码
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

ERROR_HINTS = {
    401: "API Key 无效或未配置，请在「交易设置 → 大模型设置」中填写正确的 Key。",
    402: "账户余额不足，请前往服务商充值。",
    403: "无权访问该模型，请检查 Key 的权限或换用其它模型。",
    404: "接口地址或模型名不存在，请检查 Base URL（形如 https://api.deepseek.com）与模型名。",
    429: "请求过于频繁（限流），已自动重试，请稍后再试。",
}


class LLMError(RuntimeError):
    """大模型调用失败（消息为可直接展示给用户的中文）。"""

    def __init__(self, message: str, *, status: int | None = None,
                 retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class LLMResult:
    content: str
    model: str
    usage: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"content": self.content, "model": self.model, "usage": self.usage}


def resolve_llm_config(db_path=None, settings_repo=None) -> LLMConfig:
    """配置优先级：环境变量/默认（AppConfig） < 数据库保存的 UI 设置。

    数据库里 api_key 为空时不覆盖默认值，避免「UI 存了空串把 .env 的 key 冲掉」。
    """
    cfg = AppConfig.load().llm
    saved: dict[str, Any] = {}
    try:
        if settings_repo is not None:
            saved = settings_repo.get_llm() or {}
        elif db_path is not None:
            from quant.storage.repositories import SettingsRepo

            saved = SettingsRepo(db_path).get_llm() or {}
    except Exception as e:  # noqa: BLE001 读设置失败不影响默认配置
        log.warning("读取 LLM 设置失败，使用默认配置: %s", e)
        saved = {}
    merged = cfg.model_dump()
    for k, v in (saved or {}).items():
        if k not in merged or v is None:
            continue
        if k == "api_key" and not str(v).strip():
            continue
        merged[k] = v
    return LLMConfig(**merged)


class LLMClient:
    """支持 OpenAI 兼容 / Anthropic / Gemini 三种协议的 Chat 客户端。"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.protocol = (config.api_protocol or "openai").strip().lower()

    # ---- 对外接口 ----
    def chat(self, messages: list[dict[str, str]], *, temperature: float | None = None,
             json_mode: bool = False, max_tokens: int | None = None) -> LLMResult:
        """发起一次对话补全，返回内容字符串（三种协议归一化）。"""
        if not (self.config.api_key or "").strip():
            raise LLMError(ERROR_HINTS[401], status=401)
        temp = self.config.temperature if temperature is None else float(temperature)

        if self.protocol == "anthropic":
            return self._chat_anthropic(messages, temp, json_mode, max_tokens)
        if self.protocol == "gemini":
            return self._chat_gemini(messages, temp, json_mode, max_tokens)
        return self._chat_openai(messages, temp, json_mode, max_tokens)

    def ping(self) -> dict[str, Any]:
        """连通性检查：发一条极短对话，返回 {ok, model, latency_ms, reply}。"""
        t0 = time.perf_counter()
        r = self.chat([{"role": "user", "content": "回复一个字：好"}], max_tokens=16)
        return {"ok": True, "model": r.model,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "reply": r.content.strip()[:50]}

    def list_models(self) -> list[dict[str, Any]]:
        """探测该接口地址下可用的模型列表。

        OpenAI 兼容走 GET /models；Anthropic/Gemini 无公开列表端点时返回空列表。
        返回 [{id, owned_by, created}]；失败抛 LLMError（中文消息）。
        """
        if self.protocol != "openai":
            # Anthropic / Gemini 没有稳定的 GET /models 公共端点，返回空让 UI 走预置+自定义
            return []
        base = (self.config.base_url or "").rstrip("/")
        if base.endswith("/chat/completions"):
            base = base.rsplit("/chat/completions", 1)[0]
        if not base.endswith("/v1") and "deepseek" not in base:
            base = f"{base}/v1"
        url = f"{base}/models"
        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                resp = client.get(url, headers=self._headers())
        except httpx.HTTPError as e:
            raise LLMError(f"探测模型列表失败（网络）：{e}") from e
        if resp.status_code != 200:
            hint = ERROR_HINTS.get(resp.status_code,
                                   f"探测模型列表失败（HTTP {resp.status_code}）")
            raise LLMError(hint, status=resp.status_code)
        try:
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            raise LLMError("模型列表返回格式异常") from e
        items: Any = data
        if isinstance(data, dict):
            items = data.get("data") or data.get("models") or []
        if not isinstance(items, list):
            return []
        out = []
        for m in items:
            if isinstance(m, str):
                out.append({"id": m, "owned_by": "", "created": None})
            elif isinstance(m, dict):
                mid = m.get("id") or m.get("model") or ""
                if not mid:
                    continue
                out.append({"id": str(mid),
                            "owned_by": m.get("owned_by") or m.get("ownedBy") or "",
                            "created": m.get("created")})
        return out

    # ---- 协议实现 ----
    def _chat_openai(self, messages: list[dict[str, str]], temp: float,
                     json_mode: bool, max_tokens: int | None) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temp,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        data = self._post_json(self._url_openai(), payload, self._headers())
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"大模型返回格式异常: {e}") from e
        return LLMResult(content=content, model=data.get("model") or self.config.model,
                         usage=data.get("usage") or {})

    def _chat_anthropic(self, messages: list[dict[str, str]], temp: float,
                        json_mode: bool, max_tokens: int | None) -> LLMResult:
        # Anthropic Messages API：system 需从 messages 里拆出；content 必须是字符串
        system = ""
        body_messages: list[dict[str, str]] = []
        for m in messages:
            if m.get("role") == "system":
                system += (m.get("content") or "") + "\n"
            else:
                body_messages.append({"role": m.get("role", "user"),
                                      "content": m.get("content") or ""})
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": body_messages,
            "max_tokens": int(max_tokens or 2048),
            "temperature": temp,
        }
        if system.strip():
            payload["system"] = system.strip()
        data = self._post_json(self._url_anthropic(), payload, self._headers())
        try:
            blocks = data["content"]
            content = "".join(
                (b.get("text") or "") for b in blocks
                if isinstance(b, dict) and b.get("type") == "text")
        except (KeyError, TypeError) as e:
            raise LLMError(f"Anthropic 返回格式异常: {e}") from e
        usage = data.get("usage") or {}
        return LLMResult(content=content, model=data.get("model") or self.config.model,
                         usage=usage)

    def _chat_gemini(self, messages: list[dict[str, str]], temp: float,
                     json_mode: bool, max_tokens: int | None) -> LLMResult:
        # Gemini generateContent：contents 用 role=user/model；system 用 systemInstruction
        system = ""
        contents: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                system += (m.get("content") or "") + "\n"
                continue
            # Gemini 只有 user/model 两种角色，assistant → model
            g_role = "model" if role == "assistant" else "user"
            contents.append({"role": g_role,
                             "parts": [{"text": m.get("content") or ""}]})
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temp},
        }
        if system.strip():
            payload["systemInstruction"] = {"parts": [{"text": system.strip()}]}
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = int(max_tokens)
        data = self._post_json(self._url_gemini(), payload, self._headers())
        try:
            candidates = data.get("candidates") or []
            content = "".join(
                (p.get("text") or "")
                for cand in candidates
                for p in (cand.get("content") or {}).get("parts", [])
                if isinstance(p, dict))
        except (KeyError, TypeError) as e:
            raise LLMError(f"Gemini 返回格式异常: {e}") from e
        usage = data.get("usageMetadata") or {}
        return LLMResult(content=content, model=self.config.model, usage=usage)

    # ---- 端点 / 头 ----
    def _url(self) -> str:
        """兼容旧测试的入口：默认 OpenAI 端点。"""
        if self.protocol == "anthropic":
            return self._url_anthropic()
        if self.protocol == "gemini":
            return self._url_gemini()
        return self._url_openai()

    def _url_openai(self) -> str:
        base = (self.config.base_url or "").rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if not base.endswith("/v1") and "deepseek" not in base:
            base = f"{base}/v1"
        return f"{base}/chat/completions"

    def _url_anthropic(self) -> str:
        base = (self.config.base_url or "").rstrip("/")
        if base.endswith("/v1/messages"):
            return base
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/messages"

    def _url_gemini(self) -> str:
        base = (self.config.base_url or "").rstrip("/")
        if base.endswith(":generateContent"):
            return base
        model = self.config.model
        # 容错：base_url 已含 /models/{model} 时不再重复拼
        if f"/models/{model}" in base:
            return f"{base}:generateContent"
        if base.endswith("/v1beta"):
            return f"{base}/models/{model}:generateContent"
        return f"{base}/v1beta/models/{model}:generateContent"

    def _headers(self) -> dict[str, str]:
        if self.protocol == "anthropic":
            return {
                "x-api-key": self.config.api_key.strip(),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        if self.protocol == "gemini":
            return {
                "x-goog-api-key": self.config.api_key.strip(),
                "Content-Type": "application/json",
            }
        return {
            "Authorization": f"Bearer {self.config.api_key.strip()}",
            "Content-Type": "application/json",
        }

    def _post(self, _path: str, payload: dict[str, Any]) -> dict[str, Any]:
        # 兼容旧签名：_path 忽略，用 _url()（默认 openai）
        return self._post_json(self._url(), payload, self._headers())

    def _post_json(self, url: str, payload: dict[str, Any],
                   headers: dict[str, str]) -> dict[str, Any]:
        attempts = max(1, int(self.config.max_retries))
        last_err: Exception | None = None
        for i in range(attempts):
            try:
                with httpx.Client(timeout=self.config.timeout_seconds) as client:
                    resp = client.post(url, json=payload, headers=headers)
                if resp.status_code == 200:
                    return resp.json()
                err = self._http_error(resp)
                if not err.retryable or i == attempts - 1:
                    raise err
                last_err = err
            except LLMError as e:
                if not e.retryable or i == attempts - 1:
                    raise
                last_err = e
            except httpx.TimeoutException:
                last_err = LLMError(
                    f"请求超时（{self.config.timeout_seconds:.0f}s），已重试 {i + 1} 次。"
                    "可在设置中调大超时时间或换用更快的模型。", retryable=True)
                if i == attempts - 1:
                    raise last_err from None
            except httpx.HTTPError as e:
                last_err = LLMError(f"网络连接失败: {e}。请检查网络 / 代理 / Base URL。",
                                    retryable=True)
                if i == attempts - 1:
                    raise last_err from e
            # 指数退避：0.8s, 1.6s, 3.2s …（上限 8s）
            time.sleep(min(8.0, 0.8 * (2 ** i)))
        raise last_err or LLMError("大模型调用失败（未知原因）")

    def _http_error(self, resp: httpx.Response) -> LLMError:
        status = resp.status_code
        detail = ""
        try:
            body = resp.json()
            detail = str((body.get("error") or {}).get("message") or body)[:300]
        except Exception:  # noqa: BLE001
            detail = (resp.text or "")[:300]
        hint = ERROR_HINTS.get(status)
        if hint is None:
            if 500 <= status < 600:
                hint = f"服务端错误（HTTP {status}），已自动重试。"
            else:
                hint = f"调用失败（HTTP {status}）。"
        msg = f"{hint}（HTTP {status}）" + (f"\n服务端返回：{detail}" if detail else "")
        return LLMError(msg, status=status, retryable=status in _RETRY_STATUS)


def client_from_settings(settings_repo=None, db_path=None) -> LLMClient:
    """按当前设置构造客户端（每次调用读取，UI 改完设置即时生效）。"""
    return LLMClient(resolve_llm_config(db_path=db_path, settings_repo=settings_repo))
