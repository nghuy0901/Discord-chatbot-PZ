"""
BM25 Lexical Search — In-memory BM25 index for Hybrid RAG.

Complements the vector (semantic) search with keyword-based lexical matching
using the Okapi BM25 algorithm. This is particularly effective for:
- Exact keyword matches (item names, command names, stats)
- Short queries where semantic embedding may be noisy
- Technical terms that embeddings don't capture well
- Vietnamese text with specific keywords/slang

The index is built from documents stored in PGVector (PostgreSQL) and
can be refreshed on demand (e.g., after knowledge base reload).

Architecture:
- Uses `rank_bm25.BM25Okapi` for scoring
- Vietnamese-aware tokenization (handles diacritics, compound words)
- Caches the full document corpus in memory for fast retrieval
- Thread-safe refresh mechanism
"""

import os
import re
import time
import logging
import asyncio
import threading
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "15"))
BM25_MIN_SCORE: float = float(os.getenv("BM25_MIN_SCORE", "0.5"))

# Vietnamese stopwords (common words that don't carry meaning)
VIETNAMESE_STOPWORDS = {
    "và", "hoặc", "hay", "nhưng", "mà", "thì", "là", "có", "không",
    "được", "của", "cho", "với", "trong", "trên", "dưới", "ngoài",
    "này", "đó", "kia", "nào", "gì", "ai", "đâu", "bao", "mấy",
    "rất", "lắm", "quá", "cũng", "đã", "đang", "sẽ", "vẫn", "còn",
    "từ", "đến", "về", "ra", "vào", "lên", "xuống", "sang", "qua",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "up", "down", "it", "its",
    "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them",
}

# Discord / noise patterns to strip before tokenizing
_NOISE_PATTERNS = re.compile(
    r"<@!?\d+>|<#\d+>|<@&\d+>|<a?:\w+:\d+>|https?://\S+|```[\s\S]*?```"
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25 indexing / querying.

    - Lowercases
    - Strips Discord formatting noise
    - Splits on whitespace + punctuation
    - Removes stopwords
    - Keeps tokens ≥ 2 chars (except single-char meaningful tokens)
    """
    text = _NOISE_PATTERNS.sub(" ", text.lower())
    # Split on non-alphanumeric, keeping Vietnamese diacritics
    tokens = re.findall(r"[\w\u00C0-\u024F\u1E00-\u1EFF]+", text)
    return [t for t in tokens if t not in VIETNAMESE_STOPWORDS and len(t) >= 2]


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------
class BM25Index:
    """
    In-memory BM25 index backed by documents from the PGVector store.

    Lifecycle:
        1. On first search, index is built from DB (lazy init)
        2. Can be explicitly refreshed via `refresh()`
        3. Search returns scored documents
    """

    def __init__(self):
        self._bm25 = None
        self._corpus_tokens: List[List[str]] = []
        self._documents: List[Dict[str, Any]] = []  # parallel to corpus_tokens
        self._lock = threading.Lock()
        self._last_refresh: float = 0
        self._doc_count: int = 0
        self._is_building: bool = False

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and self._doc_count > 0

    @property
    def doc_count(self) -> int:
        return self._doc_count

    def clear(self) -> None:
        """Invalidate all in-memory BM25 state."""
        with self._lock:
            self._bm25 = None
            self._corpus_tokens = []
            self._documents = []
            self._doc_count = 0
            self._last_refresh = 0
            self._is_building = False

    def refresh_from_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Build the BM25 index from a list of document dicts.
        Each dict should have at minimum: {"content": str, ...metadata}

        Returns the number of documents indexed.
        """
        from rank_bm25 import BM25Okapi

        if not documents:
            logger.warning("BM25: No documents to index.")
            return 0

        with self._lock:
            self._is_building = True
            try:
                start = time.time()
                tokens_list = []
                valid_docs = []

                for doc in documents:
                    content = doc.get("content", "")
                    if not content or len(content.strip()) < 5:
                        continue
                    tokens = tokenize(content)
                    if tokens:
                        tokens_list.append(tokens)
                        valid_docs.append(doc)

                if not tokens_list:
                    logger.warning("BM25: All documents produced empty tokens.")
                    return 0

                self._bm25 = BM25Okapi(tokens_list)
                self._corpus_tokens = tokens_list
                self._documents = valid_docs
                self._doc_count = len(valid_docs)
                self._last_refresh = time.time()

                elapsed = time.time() - start
                logger.info(
                    f"BM25 index built: {self._doc_count} docs, "
                    f"{sum(len(t) for t in tokens_list)} tokens, "
                    f"{elapsed:.2f}s"
                )
                return self._doc_count

            finally:
                self._is_building = False

    async def refresh_from_db(
        self,
        collection_name: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> int:
        """
        Rebuild the BM25 index from PGVector / PostgreSQL.
        Fetches all documents from the langchain_pg_embedding table.

        Returns the number of documents indexed.
        """
        try:
            from rag.db import get_pool, COLLECTION_NAME
            coll = collection_name or COLLECTION_NAME

            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT
                        e.document   AS content,
                        e.cmetadata  AS metadata
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = $1
                    AND e.document IS NOT NULL
                    AND LENGTH(e.document) > 5;
                """, coll)

            documents = []
            for row in rows:
                doc = {"content": row["content"]}
                meta = row["metadata"]
                if isinstance(meta, str):
                    import json
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                if isinstance(meta, dict):
                    doc.update(meta)
                documents.append(doc)

            logger.info(f"BM25: Fetched {len(documents)} documents from DB (collection={coll})")

            # Build index in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(
                None, self.refresh_from_documents, documents
            )
            return count

        except Exception as e:
            logger.error(f"BM25: Failed to refresh from DB: {e}")
            if raise_on_error:
                raise
            return 0

    async def refresh_from_kb(self) -> int:
        """
        Rebuild the BM25 index from the Knowledge Base PGVector store.
        """
        try:
            from rag.db import get_pool
            from knowledge.manager import KB_COLLECTION

            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT
                        e.document   AS content,
                        e.cmetadata  AS metadata
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = $1
                    AND e.document IS NOT NULL
                    AND LENGTH(e.document) > 5;
                """, KB_COLLECTION)

            documents = []
            for row in rows:
                doc = {"content": row["content"], "source_type": "knowledge_base"}
                meta = row["metadata"]
                if isinstance(meta, str):
                    import json
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                if isinstance(meta, dict):
                    doc.update(meta)
                documents.append(doc)

            logger.info(f"BM25 KB: Fetched {len(documents)} KB documents")

            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(
                None, self.refresh_from_documents, documents
            )
            return count

        except ImportError:
            logger.debug("BM25: Knowledge base module not available for BM25 indexing")
            return 0
        except Exception as e:
            logger.error(f"BM25 KB: Failed to refresh: {e}")
            return 0

    def search(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
        min_score: float = BM25_MIN_SCORE,
        filter_dict: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the BM25 index for relevant documents.

        Args:
            query: Text query to search.
            top_k: Max results.
            min_score: Minimum BM25 score to include.
            filter_dict: Optional metadata filter (e.g., {"channel_id": "123"}).

        Returns:
            List of result dicts with content, metadata, and bm25_score.
        """
        if not self.is_ready:
            logger.debug("BM25: Index not ready, returning empty results")
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            scores = self._bm25.get_scores(query_tokens)

        # Pair scores with documents, filter, and sort
        scored_docs = []
        for idx, score in enumerate(scores):
            if score < min_score:
                continue

            doc = self._documents[idx]

            # Apply metadata filter
            if filter_dict:
                match = all(
                    doc.get(k) == v for k, v in filter_dict.items()
                )
                if not match:
                    continue

            result = {
                **doc,
                "bm25_score": float(score),
                "retrieval_method": "bm25",
            }
            scored_docs.append(result)

        # Sort by score descending and limit
        scored_docs.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_docs[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        return {
            "is_ready": self.is_ready,
            "doc_count": self._doc_count,
            "is_building": self._is_building,
            "last_refresh": self._last_refresh,
            "avg_doc_length": (
                sum(len(t) for t in self._corpus_tokens) / max(self._doc_count, 1)
                if self._corpus_tokens else 0
            ),
        }


# ---------------------------------------------------------------------------
# Singletons: separate indices for chat history and knowledge base
# ---------------------------------------------------------------------------
_chat_bm25: Optional[BM25Index] = None
_kb_bm25: Optional[BM25Index] = None


def get_chat_bm25() -> BM25Index:
    """Return the singleton BM25 index for chat history."""
    global _chat_bm25
    if _chat_bm25 is None:
        _chat_bm25 = BM25Index()
    return _chat_bm25


def get_kb_bm25() -> BM25Index:
    """Return the singleton BM25 index for knowledge base."""
    global _kb_bm25
    if _kb_bm25 is None:
        _kb_bm25 = BM25Index()
    return _kb_bm25


async def init_bm25_indices() -> Dict[str, int]:
    """
    Initialize both BM25 indices from the database.
    Called during bot startup after DB init.

    Returns dict of index_name → doc_count.
    """
    results = {}

    chat_idx = get_chat_bm25()
    chat_count = await chat_idx.refresh_from_db()
    results["chat_history"] = chat_count

    kb_idx = get_kb_bm25()
    kb_count = await kb_idx.refresh_from_kb()
    results["knowledge_base"] = kb_count

    logger.info(f"BM25 indices initialized: {results}")
    return results
