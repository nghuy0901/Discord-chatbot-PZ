import pytest

from src.llm.gemini_provider import GeminiClient
from src.llm.types import LLMConfig


def test_gemini_requires_api_key():
    config = LLMConfig(provider="gemini", model="gemini-1.5-flash", api_key=None)
    with pytest.raises(RuntimeError, match="GEMINI"):
        GeminiClient(config)


def test_gemini_default_embedding_model_is_documented():
    config = LLMConfig(
        provider="gemini",
        model="gemini-1.5-flash",
        api_key="test-key",
        embedding_model=None,
    )
    client = GeminiClient(config)
    assert client.embedding_model == "models/text-embedding-004"
