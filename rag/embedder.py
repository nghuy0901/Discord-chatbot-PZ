"""
Embedding generation via LangChain OllamaEmbeddings.

Default model: nomic-embed-text (768-dim, high quality, fast).
Wraps langchain_ollama.OllamaEmbeddings for consistent usage across
the RAG pipeline and context manager.
"""

import os
import logging
import asyncio
from typing import List, Optional

import ollama as ollama_client
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# LangChain OllamaEmbeddings singleton
# ---------------------------------------------------------------------------
_embeddings: Optional[OllamaEmbeddings] = None


def get_embeddings() -> OllamaEmbeddings:
    """Return (and lazily create) the LangChain OllamaEmbeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        logger.info(f"OllamaEmbeddings ready: model={EMBED_MODEL}")
    return _embeddings


# ---------------------------------------------------------------------------
# Core embedding functions
# ---------------------------------------------------------------------------
async def embed_text(text: str, model: Optional[str] = None) -> List[float]:
    """
    Generate an embedding vector for a single text string.

    Uses LangChain OllamaEmbeddings.aembed_query (async).

    Args:
        text: The text to embed.
        model: Override the default embedding model (creates a one-off instance).

    Returns:
        A list of floats representing the embedding vector.
    """
    try:
        if model and model != EMBED_MODEL:
            # One-off embeddings instance with a different model
            emb = OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)
            return await emb.aembed_query(text)
        return await get_embeddings().aembed_query(text)
    except Exception as e:
        logger.error(f"Embedding failed for model '{model or EMBED_MODEL}': {e}")
        raise


async def embed_texts(
    texts: List[str],
    model: Optional[str] = None,
    batch_size: int = 64,
) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts.

    Uses LangChain OllamaEmbeddings.aembed_documents (async).
    Processes in sub-batches to avoid overwhelming GPU VRAM.

    Args:
        texts: List of strings to embed.
        model: Override the default embedding model.
        batch_size: Number of texts per sub-batch.

    Returns:
        Parallel list of embedding vectors.
    """
    emb = get_embeddings()
    if model and model != EMBED_MODEL:
        emb = OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        try:
            batch_embeddings = await emb.aembed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Batch embedding failed at index {i}: {e}")
            # Fall back to one-by-one for this batch
            for text in batch:
                try:
                    single = await emb.aembed_query(text)
                    all_embeddings.append(single)
                except Exception as inner_e:
                    logger.warning(f"Single embed fallback failed: {inner_e}")
                    dim = int(os.getenv("EMBEDDING_DIM", "768"))
                    all_embeddings.append([0.0] * dim)

        if i + batch_size < len(texts):
            await asyncio.sleep(0.05)

    return all_embeddings


# ---------------------------------------------------------------------------
# Model management (uses raw ollama client for pull/list)
# ---------------------------------------------------------------------------
async def check_model_available(model: Optional[str] = None) -> bool:
    """Check whether the embedding model is pulled and available in Ollama."""
    model = model or EMBED_MODEL
    try:
        client = ollama_client.AsyncClient(host=OLLAMA_BASE_URL)
        models_resp = await client.list()
        available = [m.get("name", m.get("model", "")) for m in models_resp.get("models", [])]
        return any(model in name for name in available)
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        return False


async def pull_model_if_missing(model: Optional[str] = None) -> None:
    """Pull the embedding model if it is not already available."""
    model = model or EMBED_MODEL
    if await check_model_available(model):
        logger.info(f"Embedding model '{model}' is available.")
        return
    logger.info(f"Pulling embedding model '{model}' — this may take a while …")
    client = ollama_client.AsyncClient(host=OLLAMA_BASE_URL)
    await client.pull(model)
    logger.info(f"Model '{model}' pulled successfully.")
