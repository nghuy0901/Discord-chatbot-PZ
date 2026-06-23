import os

from src.llm.openai_compatible import OpenAICompatibleClient
from src.llm.types import LLMConfig


SUPPORTED_EMBEDDING_PROVIDERS = {"openai", "openai_compatible", "gemini"}


def build_embedding_config() -> LLMConfig:
    provider = os.getenv("EMBEDDING_PROVIDER")
    if not provider:
        raise RuntimeError("EMBEDDING_PROVIDER is required: openai|openai_compatible|gemini")
    provider = provider.lower().strip()
    if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise RuntimeError(f"Unsupported EMBEDDING_PROVIDER={provider}")

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    return LLMConfig(
        provider=provider,
        model=embedding_model,
        embedding_model=embedding_model,
        api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
        timeout_seconds=int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "120")),
    )


def get_embedding_client():
    config = build_embedding_config()
    if config.provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleClient(config)
    if config.provider == "gemini":
        from src.llm.gemini_provider import GeminiClient

        return GeminiClient(config)
    raise RuntimeError(f"Unsupported embedding provider: {config.provider}")


def get_langchain_embeddings():
    config = build_embedding_config()
    if config.provider in {"openai", "openai_compatible"}:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=config.embedding_model or config.model,
            api_key=config.api_key or "not-needed-for-local",
            base_url=config.base_url,
        )
    if config.provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=config.embedding_model or "models/text-embedding-004",
            google_api_key=config.api_key,
        )
    raise RuntimeError(f"Unsupported embedding provider: {config.provider}")
