from typing import Any, List, Optional

from openai import AsyncOpenAI

from .pricing import estimate_cost_usd
from .types import ChatMessage, LLMConfig, LLMResponse, TokenUsage


class OpenAICompatibleClient:
    """Client for OpenAI and OpenAI-compatible /v1 chat and embedding APIs."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = AsyncOpenAI(
            # Local OpenAI-compatible servers (vLLM) ignore the key; the
            # "local-dev-key" fallback is dev-only and is rejected at startup in
            # production by src.startup.validate_runtime_config (audit M10).
            api_key=config.api_key or "local-dev-key",
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        request = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        response = await self._client.chat.completions.create(**request)
        content = response.choices[0].message.content or ""
        usage = self._usage_from_response(response)
        return LLMResponse(content=content, usage=usage, raw=self._raw_response(response))

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = await self._client.embeddings.create(
            model=self.config.embedding_model or self.config.model,
            input=texts,
        )
        return [list(item.embedding) for item in response.data]

    async def embed_query(self, text: str) -> List[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    def _usage_from_response(self, response: Any) -> TokenUsage:
        raw_usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(raw_usage, "completion_tokens", 0) or 0)
        total_tokens = getattr(raw_usage, "total_tokens", None)
        if total_tokens is not None:
            total_tokens = int(total_tokens)

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimate_cost_usd(
                self.config.model,
                prompt_tokens,
                completion_tokens,
            ),
        )

    @staticmethod
    def _raw_response(response: Any) -> Optional[dict]:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "dict"):
            return response.dict()
        return None
