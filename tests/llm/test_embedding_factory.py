import sys
from types import SimpleNamespace


def _install_fake_langchain(monkeypatch, captured):
    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(OpenAIEmbeddings=FakeOpenAIEmbeddings),
    )


def test_openai_compatible_embeddings_send_raw_text(monkeypatch):
    from src.llm.embedding_factory import get_langchain_embeddings

    captured = {}
    _install_fake_langchain(monkeypatch, captured)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_OPENAI_COMPAT_RAW_TEXT", "true")

    get_langchain_embeddings()

    assert captured["tiktoken_enabled"] is False
    assert captured["check_embedding_ctx_length"] is False


def test_openai_embeddings_receive_configured_dimension(monkeypatch):
    from src.llm.embedding_factory import get_langchain_embeddings

    captured = {}
    _install_fake_langchain(monkeypatch, captured)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")

    get_langchain_embeddings()

    assert captured["dimensions"] == 1024
    assert "tiktoken_enabled" not in captured
