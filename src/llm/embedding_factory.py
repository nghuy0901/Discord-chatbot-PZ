import os

from src.llm.openai_compatible import OpenAICompatibleClient
from src.llm.types import LLMConfig


SUPPORTED_EMBEDDING_PROVIDERS = {"openai", "openai_compatible", "gemini"}


def embedding_input_mode(provider: str | None = None) -> str:
    """Return the text representation sent to the embedding endpoint.

    ``OpenAIEmbeddings`` normally pre-tokenizes text with OpenAI's tiktoken
    tokenizer to support automatic long-text splitting.  That is correct for
    OpenAI embedding models, but wrong for a non-OpenAI compatible endpoint
    such as vLLM serving BGE-M3: its token IDs belong to BGE's tokenizer.
    Chunks in this project are deliberately below the model context limit, so
    raw text is both safe and required for the local compatible path.
    """
    resolved = (provider or os.getenv("EMBEDDING_PROVIDER", "")).lower().strip()
    raw_text_enabled = os.getenv(
        "EMBEDDING_OPENAI_COMPAT_RAW_TEXT", "true"
    ).lower() == "true"
    if resolved == "openai_compatible" and raw_text_enabled:
        return "raw_text_v1"
    return "provider_native_v1"


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

        kwargs = {
            "model": config.embedding_model or config.model,
            "api_key": config.api_key or "not-needed-for-local",
            "base_url": config.base_url,
        }
        if embedding_input_mode(config.provider) == "raw_text_v1":
            # Do not turn raw strings into cl100k token IDs before sending them
            # to BGE-M3/vLLM.  See ``embedding_input_mode`` above.
            kwargs.update(
                {
                    "tiktoken_enabled": False,
                    "check_embedding_ctx_length": False,
                }
            )
        elif config.provider == "openai":
            # `text-embedding-3-*` supports shortening at request time.  This
            # keeps the database schema and actual returned vector dimension in
            # sync when moving from the free local profile to managed OpenAI.
            dimension = int(os.getenv("EMBEDDING_DIMENSION", "0") or 0)
            if dimension > 0:
                kwargs["dimensions"] = dimension

        return OpenAIEmbeddings(
            **kwargs,
        )
    if config.provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=config.embedding_model or "models/text-embedding-004",
            google_api_key=config.api_key,
        )
    raise RuntimeError(f"Unsupported embedding provider: {config.provider}")
