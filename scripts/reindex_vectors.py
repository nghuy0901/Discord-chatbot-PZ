"""Build a verified candidate KB collection without touching the active one.

Run this from the same runtime image and environment as the bot.  The command
does not change ``.env`` or delete any collection; promotion is a separate,
human-reviewed configuration change.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# ``python scripts/reindex_vectors.py`` places ``scripts/`` rather than the
# repository root on sys.path. Keep the documented command working in Docker
# as well as on a developer machine.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Build a new NomNom KB vector collection after an embedding change."
    )
    parser.add_argument(
        "--target-collection",
        required=True,
        help="Empty candidate collection name, e.g. nomnom_knowledge_v2.",
    )
    parser.add_argument(
        "--collection-version",
        default="v2",
        help="Version written into the candidate's embedding signature (default: v2).",
    )
    parser.add_argument(
        "--min-chunks",
        type=int,
        default=1,
        help="Fail if fewer chunks are written (default: 1).",
    )
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()

    if args.confirm != "REINDEX":
        raise SystemExit("Pass --confirm REINDEX to run.")

    active_collection = os.getenv("KB_COLLECTION", "knowledge_base")
    if args.target_collection == active_collection:
        raise SystemExit(
            "Target must differ from the active KB_COLLECTION. "
            "Use a new versioned name so rollback remains possible."
        )

    # These modules cache configuration at import time, so set the candidate
    # environment before importing project code.
    os.environ["KB_COLLECTION"] = args.target_collection
    os.environ["EMBEDDING_COLLECTION_VERSION"] = args.collection_version

    from knowledge.manager import get_knowledge_manager
    from rag.db import close_pool, get_pool, init_db

    try:
        # This only initializes shared schema. The KB signature is recorded by
        # ``KnowledgeManager.load_all`` for the candidate collection below.
        await init_db()
        pool = await get_pool()
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM langchain_pg_collection WHERE name = $1);",
                args.target_collection,
            )
        if exists:
            raise SystemExit(
                f"Candidate collection '{args.target_collection}' already exists. "
                "Choose a new collection name; this command will not overwrite it."
            )

        manager = get_knowledge_manager()
        per_domain = await manager.load_all()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*)::int AS chunks,
                       COUNT(DISTINCT e.cmetadata->>'source')::int AS sources,
                       COUNT(DISTINCT e.cmetadata->>'domain')::int AS domains
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                WHERE c.name = $1;
                """,
                args.target_collection,
            )

        result = {
            "active_collection_unchanged": active_collection,
            "candidate_collection": args.target_collection,
            "collection_version": args.collection_version,
            "domains_indexed": per_domain,
            "chunks": row["chunks"],
            "sources": row["sources"],
            "domains": row["domains"],
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        if row["chunks"] < args.min_chunks:
            raise SystemExit(
                f"Candidate has only {row['chunks']} chunks; expected at least {args.min_chunks}."
            )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
