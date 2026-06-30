from typing import Any, List, Optional

import aiohttp

from .ollama_auth import build_ollama_headers
from .types import ChatMessage, LLMConfig, LLMResponse, TokenUsage


class OllamaClient:
    """Asynchronous client for Ollama's native /api endpoints.

    Accepts arbitrary ``**kwargs`` (e.g. ``tools=[...]``) and forwards them into
    the request body. Tools MUST already be JSON tool *schemas* (dicts), not raw
    Python callables — see src/llm/tool_schema.py.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        request.update(kwargs)

        base_url = (self.config.base_url or "http://localhost:11434").rstrip("/")
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        headers = build_ollama_headers(self.config.api_key)

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.post(f"{base_url}/api/chat", json=request) as response:
                if response.status >= 400:
                    detail = (await response.text()).strip()
                    if len(detail) > 1000:
                        detail = detail[:1000]
                    raise RuntimeError(
                        f"Ollama request failed with status {response.status}: {detail}"
                    )
                payload = await response.json()

        message = payload.get("message") or {}
        prompt_tokens = int(payload.get("prompt_eval_count") or 0)
        completion_tokens = int(payload.get("eval_count") or 0)
        return LLMResponse(
            content=message.get("content") or "",
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            raw=payload,
        )
