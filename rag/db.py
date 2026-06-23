"""
PostgreSQL + pgvector database layer via LangChain PGVector.

Uses:
- langchain_community.vectorstores.PGVector for vector storage & similarity search
- provider-neutral LangChain embeddings from src.llm.embedding_factory
- asyncpg for direct SQL operations (edges table, counts, custom queries)

Tables (managed by LangChain + custom SQL):
- langchain_pg_collection / langchain_pg_embedding: LangChain PGVector tables
- message_edges: reply/mention relationships (custom)
"""

import os
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any

import asyncpg
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)
# LangChain PGVector needs the psycopg2/postgresql+psycopg URI scheme
# Convert asyncpg-style URL → SQLAlchemy-style if needed
PGVECTOR_CONNECTION: str = POSTGRES_URL.replace(
    "postgresql://", "postgresql+psycopg://", 1
) if "postgresql://" in POSTGRES_URL else POSTGRES_URL

EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIMENSION", os.getenv("EMBEDDING_DIM", "768")))
EMBED_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
COLLECTION_NAME: str = os.getenv("PGVECTOR_COLLECTION", "discord_messages")


# ---------------------------------------------------------------------------
# LangChain embeddings (singleton)
# ---------------------------------------------------------------------------
_embeddings: Optional[Any] = None


def get_embeddings() -> Any:
    """Return (and lazily create) the configured LangChain embeddings instance."""
    global _embeddings
    if _embeddings is None:
        from src.llm.embedding_factory import get_langchain_embeddings

        _embeddings = get_langchain_embeddings()
        logger.info("LangChain embeddings initialised: model=%s", EMBED_MODEL)
    return _embeddings


# ---------------------------------------------------------------------------
# LangChain PGVector store (singleton)
# ---------------------------------------------------------------------------
_vectorstore: Optional[PGVector] = None


def get_vectorstore() -> PGVector:
    """
    Return (and lazily create) the LangChain PGVector vector store.
    This automatically creates the pgvector extension and tables on first use.
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=PGVECTOR_CONNECTION,
            embedding_function=get_embeddings(),
            use_jsonb=True,
        )
        logger.info(f"PGVector store initialised: collection={COLLECTION_NAME}")
    return _vectorstore


# ---------------------------------------------------------------------------
# asyncpg pool for custom SQL (edges, counts)
# ---------------------------------------------------------------------------
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Return (and lazily create) the asyncpg connection pool."""
    global _pool
    if _pool is None or _pool._closed:
        logger.info("Creating asyncpg connection pool …")
        _pool = await asyncpg.create_pool(
            POSTGRES_URL,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("Connection pool created.")
    return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool."""
    global _pool
    if _pool and not _pool._closed:
        await _pool.close()
        _pool = None
        logger.info("Connection pool closed.")


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------
async def init_db() -> None:
    """
    Ensure pgvector extension and custom tables exist.
    LangChain PGVector creates its own tables automatically;
    we only need the edges table and pgvector extension.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Parent-child relationship edges (replies / mentions)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS message_edges (
                id              BIGSERIAL PRIMARY KEY,
                parent_msg_id   TEXT NOT NULL,
                child_msg_id    TEXT NOT NULL,
                edge_type       TEXT NOT NULL DEFAULT 'reply',
                UNIQUE (parent_msg_id, child_msg_id, edge_type)
            );
        """)

    # Trigger PGVector store creation (creates collection + embedding tables)
    get_vectorstore()

    logger.info("Database schema initialised (LangChain PGVector + edges table ready).")


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------
def _msg_to_document(
    message_id: str,
    channel_id: str,
    author_id: str,
    content: str,
    timestamp: str,
    thread_id: Optional[str] = None,
    author_name: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Document:
    """Convert a message dict into a LangChain Document."""
    meta = {
        "message_id": message_id,
        "channel_id": channel_id,
        "author_id": author_id,
        "author_name": author_name or "",
        "timestamp": timestamp,
    }
    if thread_id:
        meta["thread_id"] = thread_id
    if metadata:
        meta.update(metadata)

    return Document(page_content=content, metadata=meta)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------
def add_documents(docs: List[Document]) -> List[str]:
    """
    Add documents to the PGVector store (sync, calls LangChain).
    Returns list of IDs.
    """
    store = get_vectorstore()
    return store.add_documents(docs)


def add_documents_batch(
    messages: List[Dict[str, Any]],
    batch_size: int = 500,
) -> int:
    """
    Bulk-add message dicts as LangChain Documents.
    Returns the number of documents added.
    """
    if not messages:
        return 0

    count = 0
    for i in range(0, len(messages), batch_size):
        chunk = messages[i: i + batch_size]
        docs = []
        for msg in chunk:
            try:
                doc = _msg_to_document(
                    message_id=msg["message_id"],
                    channel_id=msg["channel_id"],
                    author_id=msg["author_id"],
                    content=msg["content"],
                    timestamp=msg["timestamp"],
                    thread_id=msg.get("thread_id"),
                    author_name=msg.get("author_name"),
                )
                docs.append(doc)
            except Exception as e:
                logger.warning(f"Skipped message {msg.get('message_id')}: {e}")

        if docs:
            try:
                add_documents(docs)
                count += len(docs)
            except Exception as e:
                logger.error(f"Batch add failed at index {i}: {e}")
                # Fallback: add one by one
                for doc in docs:
                    try:
                        add_documents([doc])
                        count += 1
                    except Exception as inner_e:
                        logger.warning(f"Single add failed: {inner_e}")

    return count


# ---------------------------------------------------------------------------
# Similarity search (via LangChain)
# ---------------------------------------------------------------------------
def search_similar(
    query: str,
    k: int = 15,
    filter_dict: Optional[Dict[str, str]] = None,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Search for similar documents using LangChain PGVector.

    Args:
        query: The text query to search against.
        k: Maximum results to return.
        filter_dict: Optional metadata filter (e.g. {"channel_id": "123"}).
        score_threshold: Minimum similarity score (0–1). None = no threshold.

    Returns:
        List of dicts with document content, metadata, and similarity score.
    """
    store = get_vectorstore()

    if score_threshold is not None:
        results = store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=filter_dict,
            score_threshold=score_threshold,
        )
    else:
        results = store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=filter_dict,
        )

    output = []
    for doc, score in results:
        entry = {
            "content": doc.page_content,
            "similarity": score,
            **doc.metadata,
        }
        output.append(entry)

    return output


# ---------------------------------------------------------------------------
# Edge operations (async, raw SQL)
# ---------------------------------------------------------------------------
async def insert_edge(
    parent_msg_id: str, child_msg_id: str, edge_type: str = "reply"
) -> None:
    """Record a parent→child relationship (reply or mention)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO message_edges (parent_msg_id, child_msg_id, edge_type)
            VALUES ($1, $2, $3)
            ON CONFLICT DO NOTHING;
            """,
            parent_msg_id,
            child_msg_id,
            edge_type,
        )


async def get_message_context(message_id: str) -> List[Dict[str, Any]]:
    """Retrieve the parent and sibling messages of a given message (via edges)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Query LangChain's embedding table joined with edges
        rows = await conn.fetch(
            """
            SELECT e.parent_msg_id AS message_id, e.edge_type
            FROM message_edges e
            WHERE e.child_msg_id = $1
            UNION
            SELECT e.child_msg_id AS message_id, e.edge_type
            FROM message_edges e
            WHERE e.parent_msg_id = $1;
            """,
            message_id,
        )
        return [dict(r) for r in rows]


async def get_message_count() -> int:
    """Return total number of documents in the PGVector collection."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT COUNT(*) AS cnt
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = $1;
            """,
            COLLECTION_NAME,
        )
        return row["cnt"] if row else 0
