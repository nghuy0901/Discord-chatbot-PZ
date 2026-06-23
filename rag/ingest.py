#!/usr/bin/env python3
"""
Ingestion script — processes a historical Discord conversation dataset (JSON),
generates embeddings via the configured embedding provider, and upserts into PostgreSQL + pgvector.

Usage:
    python -m rag.ingest <path_to_json> [--batch-size 100] [--model text-embedding-3-small]

The JSON file is expected to be either:
  - A JSON array of message objects, OR
  - A JSON object with a key containing a list of messages
    (e.g. {"messages": [...]})

Each message object should have at least:
  - content (str)
  - author / author_id (str)
  - timestamp (str, ISO-8601 preferred)
  - id / message_id (str)
  - channel_id (str, optional)

Optional fields: thread_id, references (dict with message_id), mentions (list).
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag.ingest")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_message(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalise a raw message dict into canonical form.
    Returns None if the message should be skipped (empty content, bot spam, etc.).
    """
    content: str = raw.get("content", raw.get("text", "")).strip()
    if not content:
        return None

    # Resolve author
    author_id = raw.get("author_id", "")
    author_name = raw.get("author_name", raw.get("author", ""))
    if isinstance(raw.get("author"), dict):
        author_id = raw["author"].get("id", author_id)
        author_name = raw["author"].get("name", raw["author"].get("username", author_name))
    if not author_id:
        author_id = str(author_name)

    # Resolve IDs
    message_id = str(raw.get("message_id", raw.get("id", "")))
    if not message_id:
        return None

    channel_id = str(raw.get("channel_id", "unknown"))
    thread_id = raw.get("thread_id")
    if thread_id:
        thread_id = str(thread_id)

    # Resolve timestamp
    ts_raw = raw.get("timestamp", raw.get("created_at", ""))
    if isinstance(ts_raw, (int, float)):
        ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc).isoformat()
    elif ts_raw:
        ts = str(ts_raw)
    else:
        ts = datetime.now(tz=timezone.utc).isoformat()

    # References (reply parents)
    references: List[str] = []
    # Handle 'reference' / 'references' key (standard Discord export)
    ref = raw.get("reference", raw.get("references", None))
    if isinstance(ref, dict):
        parent_id = ref.get("message_id", ref.get("messageId", ""))
        if parent_id:
            references.append(str(parent_id))
    elif isinstance(ref, list):
        for r in ref:
            if isinstance(r, dict):
                pid = r.get("message_id", r.get("messageId", ""))
                if pid:
                    references.append(str(pid))
            elif isinstance(r, str):
                references.append(r)

    # Handle 'reply_to' key (Chấn thương tâm lý dataset format)
    reply_to = raw.get("reply_to")
    if isinstance(reply_to, dict) and not references:
        parent_id = reply_to.get("message_id", "")
        if parent_id:
            references.append(str(parent_id))

    # Mentions
    mentions: List[str] = []
    raw_mentions = raw.get("mentions", [])
    for m in raw_mentions:
        if isinstance(m, dict):
            mentions.append(str(m.get("id", m.get("user_id", ""))))
        elif isinstance(m, str):
            mentions.append(m)

    return {
        "message_id": message_id,
        "channel_id": channel_id,
        "thread_id": thread_id,
        "author_id": author_id,
        "author_name": author_name,
        "content": content,
        "timestamp": ts,
        "references": references,
        "mentions": mentions,
    }


def _load_json(path: str) -> List[Dict[str, Any]]:
    """Load and return a list of raw message dicts from a JSON file."""
    logger.info(f"Loading JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try common key names
        for key in ("messages", "data", "records", "items", "content"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # Fallback: return single-element list if it looks like a message
        if "content" in data or "message_id" in data:
            return [data]
    raise ValueError(
        "Unrecognised JSON structure. Expected a list of messages or "
        "a dict with a 'messages' key."
    )


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------
async def ingest_file(
    path: str,
    batch_size: int = 100,
    embed_model: Optional[str] = None,
) -> Dict[str, int]:
    """
    Full ingestion pipeline:
    1. Load JSON
    2. Normalise messages
    3. Add to LangChain PGVector store (embedding handled by LangChain)
    4. Record reply/mention edges

    Returns summary stats.
    """
    # Import here to avoid circular imports at module level
    from rag.db import init_db, add_documents_batch, insert_edge
    from rag.embedder import pull_model_if_missing

    # Ensure DB schema exists
    await init_db()

    # Ensure embedding model is available
    await pull_model_if_missing(embed_model)

    # Load and normalise
    raw_messages = _load_json(path)
    logger.info(f"Loaded {len(raw_messages)} raw messages.")

    messages = []
    for raw in raw_messages:
        msg = _normalise_message(raw)
        if msg:
            messages.append(msg)
    logger.info(f"Normalised to {len(messages)} valid messages.")

    if not messages:
        logger.warning("No valid messages to ingest.")
        return {"total_raw": len(raw_messages), "normalised": 0, "ingested": 0, "edges": 0}

    # Insert documents via LangChain PGVector in batches
    # LangChain handles embedding generation internally
    ingested = 0
    edge_count = 0
    total = len(messages)

    for i in range(0, total, batch_size):
        batch = messages[i : i + batch_size]

        logger.info(f"Adding batch {i // batch_size + 1} "
                     f"({i + 1}–{min(i + batch_size, total)} of {total}) …")

        count = add_documents_batch(batch, batch_size=batch_size)
        ingested += count

        # Record edges (reply/mention relationships)
        for msg in batch:
            for parent_id in msg.get("references", []):
                try:
                    await insert_edge(parent_id, msg["message_id"], "reply")
                    edge_count += 1
                except Exception as e:
                    logger.warning(f"Edge insert failed: {e}")

            for mentioned_id in msg.get("mentions", []):
                try:
                    await insert_edge(mentioned_id, msg["message_id"], "mention")
                    edge_count += 1
                except Exception as e:
                    logger.warning(f"Mention edge insert failed: {e}")

        logger.info(f"  ✓ Ingested {ingested}/{total} messages, {edge_count} edges")

    stats = {
        "total_raw": len(raw_messages),
        "normalised": len(messages),
        "ingested": ingested,
        "edges": edge_count,
    }
    logger.info(f"Ingestion complete: {stats}")
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ingest Discord conversation history into PostgreSQL + pgvector."
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to the JSON conversation dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of messages per embedding batch (default: 100).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model override (default: EMBEDDING_MODEL from .env).",
    )
    args = parser.parse_args()

    if not Path(args.file).is_file():
        logger.error(f"File not found: {args.file}")
        sys.exit(1)

    stats = asyncio.run(ingest_file(args.file, args.batch_size, args.model))
    print(f"\n{'='*50}")
    print(f"  Ingestion Summary")
    print(f"{'='*50}")
    for k, v in stats.items():
        print(f"  {k:>15}: {v}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
