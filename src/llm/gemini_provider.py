from typing import List, Optional

import google.generativeai as genai

from src.llm.types import ChatMessage, LLMConfig, LLMResponse, TokenUsage


class GeminiClient:
    def __init__(self, config: LLMConfig):
        if not config.api_key:
            raise RuntimeError("GEMINI API key is required for GeminiClient")
        self.config = config
        self.embedding_model = config.embedding_model or "models/text-embedding-004"
        genai.configure(api_key=config.api_key)
        self._model = genai.GenerativeModel(config.model)

    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.8,
        tools: Optional[list] = None,
    ) -> LLMResponse:
        if tools:
            raise NotImplementedError("Gemini tool-calling adapter is not implemented in NomNom yet")
        prompt = "\n\n".join(f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages)
        response = await self._model.generate_content_async(
            prompt,
            generation_config={"temperature": temperature},
        )
        text = getattr(response, "text", "") or ""
        return LLMResponse(content=text, usage=TokenUsage(), raw={"provider": "gemini"})

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            result = await genai.embed_content_async(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document",
            )
            vectors.append(result["embedding"])
        return vectors

    async def embed_text(self, text: str) -> List[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_query(self, text: str) -> List[float]:
        result = await genai.embed_content_async(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]
