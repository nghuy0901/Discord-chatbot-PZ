import pytest

from rag.embedding_registry import EmbeddingSignature, validate_embedding_signature


def test_embedding_signature_detects_model_change():
    current = EmbeddingSignature(
        provider="openai_compatible",
        model="BAAI/bge-m3",
        dimension=1024,
        collection_version="v1",
    )
    stored = EmbeddingSignature(
        provider="openai_compatible",
        model="text-embedding-3-small",
        dimension=1536,
        collection_version="v1",
    )
    with pytest.raises(RuntimeError, match="Embedding collection mismatch"):
        validate_embedding_signature(current=current, stored=stored)
