import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from knowledge.redaction import redact_sensitive_text


TRUSTED_KB_DIRS = ("pz", "server_rules")
KNOWLEDGE_ROOT = ROOT_DIR / "knowledge" / "docs"


def _title_from_text(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("_", " ").replace("-", " ")


def _excerpt(text: str, max_chars: int = 500) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:max_chars]


def _assert_safe_excerpt(source_id: str, excerpt: str) -> None:
    redacted = redact_sensitive_text(excerpt)
    if redacted.redaction_count:
        raise ValueError(f"Refusing to export sensitive excerpt from {source_id}")


def kb_inventory() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for domain in TRUSTED_KB_DIRS:
        domain_dir = KNOWLEDGE_ROOT / domain
        if not domain_dir.exists():
            continue
        for path in sorted(domain_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".rst"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            excerpt = _excerpt(text)
            rel = path.relative_to(KNOWLEDGE_ROOT).as_posix()
            source_id = f"{rel}#0"
            _assert_safe_excerpt(source_id, excerpt)
            rows.append(
                {
                    "source_id": source_id,
                    "source_kind": "knowledge_base",
                    "domain": domain,
                    "title": _title_from_text(path, text),
                    "excerpt": excerpt,
                }
            )
    return rows


async def approved_chat_inventory() -> List[Dict[str, Any]]:
    try:
        from rag.db import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                  e.cmetadata->>'message_id' AS message_id,
                  e.cmetadata->>'channel_id' AS channel_id,
                  e.document AS content
                FROM langchain_pg_embedding e
                WHERE e.cmetadata->>'approval_status' = 'approved'
                  AND e.cmetadata->>'trusted' = 'true'
                  AND e.document IS NOT NULL
                ORDER BY e.cmetadata->>'message_id'
                """
            )
    except Exception:
        return []

    output = []
    for row in rows:
        message_id = row["message_id"]
        excerpt = _excerpt(row["content"])
        source_id = f"approved_chat:{message_id}"
        _assert_safe_excerpt(source_id, excerpt)
        output.append(
            {
                "source_id": source_id,
                "source_kind": "approved_chat",
                "domain": "approved_chat",
                "title": f"Approved chat {message_id}",
                "channel_id": row["channel_id"],
                "excerpt": excerpt,
            }
        )
    return output


async def build_inventory() -> List[Dict[str, Any]]:
    rows = kb_inventory()
    rows.extend(await approved_chat_inventory())
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_inventory(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def draft_candidates(inventory: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = []
    for source in inventory:
        title = source.get("title") or source["source_id"]
        excerpt = source.get("excerpt", "")
        question = f"{title} nói gì?"
        candidates.append(
            {
                "question": question,
                "draft_ground_truth": excerpt[:240],
                "candidate_source_ids": [source["source_id"]],
                "candidate_category": "answerable",
                "review_status": "unreviewed",
            }
        )
    return candidates


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Export trusted eval source inventory and draft candidates.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subcommands.add_parser("inventory")
    inventory_parser.add_argument("--output", required=True)

    draft_parser = subcommands.add_parser("draft")
    draft_parser.add_argument("--inventory", required=True)
    draft_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "inventory":
        rows = await build_inventory()
        write_json(Path(args.output), rows)
        print(f"wrote {len(rows)} inventory rows")
        return 0

    rows = draft_candidates(load_inventory(Path(args.inventory)))
    write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows)} draft candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
