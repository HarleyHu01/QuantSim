"""大模型接入层（OpenAI 兼容 Chat Completions，httpx 直连，无需 SDK）。"""

from quant.llm.client import LLMClient, LLMError, resolve_llm_config

__all__ = ["LLMClient", "LLMError", "resolve_llm_config"]
