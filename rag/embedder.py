"""
Provider-neutral embedding helpers for the RAG pipeline.
"""

import logging
import os
from typing import List, Optional

from dotenv import load_dotenv

from src.llm.embedding_factory import get_embedding_client, get_langchain_embeddings

load_dotenv()
logger = logging.getLogger(__name__)

EMBED_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_embeddings():
    """Return a LangChain-compatible embeddings instance."""
    return get_langchain_embeddings()


async def embed_text(text: str, model: Optional[str] = None) -> List[float]:
    client = get_embedding_client()
    if hasattr(client, "embed_query"):
        return await client.embed_query(text)
    vectors = await client.embed_texts([text])
    return vectors[0]


async def embed_texts(
    texts: List[str],
    model: Optional[str] = None,
    batch_size: int = 64,
) -> List[List[float]]:
    client = get_embedding_client()
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        vectors.extend(await client.embed_texts(texts[i : i + batch_size]))
    return vectors


async def check_model_available(model: Optional[str] = None) -> bool:
    try:
        get_embedding_client()
        return True
    except Exception as exc:
        logger.error("Embedding provider is not configured: %s", exc)
        return False


async def pull_model_if_missing(model: Optional[str] = None) -> None:
    """Compatibility no-op; provider-neutral embeddings do not pull local models."""
    if not await check_model_available(model):
        raise RuntimeError("Embedding provider is not configured.")
