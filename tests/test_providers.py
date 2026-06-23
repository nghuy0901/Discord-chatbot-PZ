import pytest

from src.providers import ModelInfo, ProviderManager, ProviderType


def test_supported_provider_types_do_not_include_free_provider():
    assert ProviderType.OPENAI.value == "openai"
    assert ProviderType.GEMINI.value == "gemini"
    assert ProviderType.OPENAI_COMPATIBLE.value == "openai_compatible"
    assert "free" not in {provider.value for provider in ProviderType}


def test_model_info_creation():
    model = ModelInfo(
        name="test-model",
        provider=ProviderType.OPENAI,
        description="Test model",
        supports_vision=True,
        supports_image_generation=False,
    )

    assert model.name == "test-model"
    assert model.provider == ProviderType.OPENAI
    assert model.description == "Test model"
    assert model.supports_vision is True
    assert model.supports_image_generation is False


def test_provider_manager_requires_configured_provider(monkeypatch):
    for key in (
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "OPENAI_KEY",
        "GEMINI_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        ProviderManager()


def test_provider_manager_initializes_openai_from_llm_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    manager = ProviderManager()

    assert manager.current_provider == ProviderType.OPENAI
    assert manager.get_available_providers() == [ProviderType.OPENAI]
