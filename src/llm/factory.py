import os

from src.llm.openai_compatible import OpenAICompatibleClient
from src.llm.ollama_auth import validate_ollama_auth
from src.llm.types import LLMConfig


SUPPORTED_PROVIDERS = {"openai", "gemini", "openai_compatible", "ollama"}


def build_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        raise RuntimeError(
            "LLM_PROVIDER is required: openai|gemini|openai_compatible|ollama"
        )
    provider = provider.lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise RuntimeError(f"Unsupported LLM_PROVIDER={provider}")

    if provider == "ollama":
        model = os.getenv("LLM_MODEL") or os.getenv(
            "OLLAMA_MODEL", "qwen3-coder-next:cloud"
        )
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OLLAMA_API_KEY")
        base_url = os.getenv("LLM_BASE_URL") or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        validate_ollama_auth(base_url, api_key)
        return LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=int(
                os.getenv("LLM_TIMEOUT_SECONDS", os.getenv("OLLAMA_TIMEOUT", "120"))
            ),
        )

    return LLMConfig(
        provider=provider,
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
    )


def get_chat_client():
    config = build_llm_config()
    if config.provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleClient(config)
    if config.provider == "gemini":
        from src.llm.gemini_provider import GeminiClient

        return GeminiClient(config)
    if config.provider == "ollama":
        from src.llm.ollama_client import OllamaClient

        return OllamaClient(config)
    raise RuntimeError(f"Unsupported provider: {config.provider}")
