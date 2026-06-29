import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.dataset_schema import validate_dataset


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a RAG evaluation JSONL dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--require-release-quota", action="store_true")
    args = parser.parse_args()

    try:
        rows = load_jsonl(Path(args.dataset))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    errors = validate_dataset(rows, require_release_quota=args.require_release_quota)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {len(rows)} dataset rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
