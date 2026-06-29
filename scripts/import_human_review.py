import argparse
import asyncio
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


BOOLEAN_VALUES = {"true": True, "false": False}


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in BOOLEAN_VALUES:
        raise ValueError(f"invalid boolean: {value}")
    return BOOLEAN_VALUES[normalized]


def load_reviews(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            correctness = float(row["correctness"])
            faithfulness = float(row["faithfulness"])
            if not 0 <= correctness <= 1:
                raise ValueError(f"{row['example_id']}: correctness out of range")
            if not 0 <= faithfulness <= 1:
                raise ValueError(f"{row['example_id']}: faithfulness out of range")
            rows.append(
                {
                    **row,
                    "correctness": correctness,
                    "faithfulness": faithfulness,
                    "unsupported_claim": parse_bool(row["unsupported_claim"]),
                    "critical_error": parse_bool(row["critical_error"]),
                    "behavior_correct": parse_bool(row["behavior_correct"]),
                }
            )
    return rows


async def upsert_reviews(rows: list[dict]) -> None:
    from rag.db import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        for row in rows:
            await conn.execute(
                """
                INSERT INTO human_eval_reviews (
                  run_id, example_id, reviewer_id, correctness, faithfulness,
                  unsupported_claim, critical_error, behavior_correct, notes
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (run_id, example_id, reviewer_id) DO UPDATE SET
                  correctness = EXCLUDED.correctness,
                  faithfulness = EXCLUDED.faithfulness,
                  unsupported_claim = EXCLUDED.unsupported_claim,
                  critical_error = EXCLUDED.critical_error,
                  behavior_correct = EXCLUDED.behavior_correct,
                  notes = EXCLUDED.notes,
                  reviewed_at = NOW()
                """,
                row["run_id"],
                row["example_id"],
                row["reviewer_id"],
                row["correctness"],
                row["faithfulness"],
                row["unsupported_claim"],
                row["critical_error"],
                row["behavior_correct"],
                row.get("notes", ""),
            )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Import human RAG evaluation reviews.")
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    rows = load_reviews(Path(args.csv))
    await upsert_reviews(rows)
    print(f"imported {len(rows)} human review rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
