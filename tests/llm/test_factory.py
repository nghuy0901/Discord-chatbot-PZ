import pytest

from src.llm.embedding_factory import build_embedding_config
from src.llm.factory import build_llm_config, get_chat_client


def test_ollama_cloud_config_and_client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3-coder-next:cloud")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    monkeypatch.setenv("OLLAMA_API_KEY", "cloud-key")

    config = build_llm_config()
    assert config.provider == "ollama"
    assert config.model == "qwen3-coder-next:cloud"
    assert config.base_url == "https://ollama.com"

    from src.llm.ollama_client import OllamaClient

    assert isinstance(get_chat_client(), OllamaClient)


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


def test_ollama_config_uses_compatibility_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3-coder-next:cloud")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test-key")

    config = build_llm_config()

    assert config.provider == "ollama"
    assert config.model == "qwen3-coder-next:cloud"
    assert config.base_url == "https://ollama.com"
    assert config.api_key == "ollama-test-key"


def test_ollama_cloud_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OLLAMA_API_KEY"):
        build_llm_config()
