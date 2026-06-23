import pytest

from src.llm.embedding_factory import build_embedding_config
from src.llm.factory import build_llm_config


def test_openai_compatible_config_for_vllm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_MODEL", "vllm-local")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_API_KEY", "local-key")

    config = build_llm_config()

    assert config.provider == "openai_compatible"
    assert config.model == "vllm-local"
    assert config.base_url == "http://localhost:8000/v1"


def test_embedding_config_is_independent_from_chat_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-1.5-flash")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "local-key")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")

    config = build_embedding_config()

    assert config.provider == "openai_compatible"
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.base_url == "http://localhost:8000/v1"


def test_factory_rejects_missing_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
        build_llm_config()
