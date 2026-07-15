from dataclasses import dataclass
import os

from src.llm.embedding_factory import embedding_input_mode


@dataclass(frozen=True)
class EmbeddingSignature:
    provider: str
    model: str
    dimension: int
    collection_version: str
    input_mode: str = "provider_native_v1"


def current_embedding_signature() -> EmbeddingSignature:
    return EmbeddingSignature(
        provider=os.getenv("EMBEDDING_PROVIDER", ""),
        model=os.getenv("EMBEDDING_MODEL", ""),
        dimension=int(os.getenv("EMBEDDING_DIMENSION", "0")),
        collection_version=os.getenv("EMBEDDING_COLLECTION_VERSION", "v1"),
        input_mode=embedding_input_mode(),
    )


def validate_embedding_signature(current: EmbeddingSignature, stored: EmbeddingSignature) -> None:
    if current != stored:
        raise RuntimeError(
            "Embedding collection mismatch. Reindex required before serving RAG queries. "
            f"current={current}, stored={stored}"
        )
