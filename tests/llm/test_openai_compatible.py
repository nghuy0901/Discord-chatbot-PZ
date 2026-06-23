from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.openai_compatible import OpenAICompatibleClient
from src.llm.types import LLMConfig


@pytest.mark.asyncio
async def test_chat_returns_content_and_usage():
    config = LLMConfig(
        provider="openai_compatible",
        model="nomnom-test-model",
        api_key="test-key",
        base_url="http://localhost:8001/v1",
    )
    client = OpenAICompatibleClient(config)

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="hello"))]
    fake_response.usage = MagicMock(prompt_tokens=3, completion_tokens=4, total_tokens=7)
    client._client.chat.completions.create = AsyncMock(return_value=fake_response)

    response = await client.chat([{"role": "user", "content": "hi"}], temperature=0.1)

    assert response.content == "hello"
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 7


@pytest.mark.asyncio
async def test_embed_returns_vectors():
    config = LLMConfig(
        provider="openai_compatible",
        model="nomnom-test-model",
        embedding_model="text-embedding-3-small",
        api_key="test-key",
        base_url="http://localhost:8001/v1",
    )
    client = OpenAICompatibleClient(config)

    fake_embedding = MagicMock(embedding=[0.1, 0.2, 0.3])
    fake_response = MagicMock(data=[fake_embedding])
    client._client.embeddings.create = AsyncMock(return_value=fake_response)

    vectors = await client.embed_texts(["abc"])

    assert vectors == [[0.1, 0.2, 0.3]]
