from dataclasses import dataclass
import os


@dataclass(frozen=True)
class EmbeddingSignature:
    provider: str
    model: str
    dimension: int
    collection_version: str


def current_embedding_signature() -> EmbeddingSignature:
    return EmbeddingSignature(
        provider=os.getenv("EMBEDDING_PROVIDER", ""),
        model=os.getenv("EMBEDDING_MODEL", ""),
        dimension=int(os.getenv("EMBEDDING_DIMENSION", "0")),
        collection_version=os.getenv("EMBEDDING_COLLECTION_VERSION", "v1"),
    )


def validate_embedding_signature(current: EmbeddingSignature, stored: EmbeddingSignature) -> None:
    if current != stored:
        raise RuntimeError(
            "Embedding collection mismatch. Reindex required before serving RAG queries. "
            f"current={current}, stored={stored}"
        )
