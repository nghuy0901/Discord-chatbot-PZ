import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _agreement(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row[f"judge_{key}"]) == bool(row[f"human_{key}"])) / len(rows)


def calibrate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sample_size = len(rows)
    correctness_mae = (
        sum(abs(float(row["judge_correctness"]) - float(row["human_correctness"])) for row in rows)
        / sample_size
        if sample_size
        else 0.0
    )
    faithfulness_mae = (
        sum(abs(float(row["judge_faithfulness"]) - float(row["human_faithfulness"])) for row in rows)
        / sample_size
        if sample_size
        else 0.0
    )
    judge_accuracy = (
        sum(float(row["judge_correctness"]) for row in rows) / sample_size
        if sample_size
        else 0.0
    )
    human_accuracy = (
        sum(float(row["human_correctness"]) for row in rows) / sample_size
        if sample_size
        else 0.0
    )
    aggregate_gap = abs(judge_accuracy - human_accuracy)
    unsupported_agreement = _agreement(rows, "unsupported_claim")
    critical_agreement = _agreement(rows, "critical_error")
    return {
        "sample_size": sample_size,
        "correctness_mean_absolute_error": round(correctness_mae, 4),
        "faithfulness_mean_absolute_error": round(faithfulness_mae, 4),
        "unsupported_claim_agreement": round(unsupported_agreement, 4),
        "critical_error_agreement": round(critical_agreement, 4),
        "aggregate_accuracy_gap": round(aggregate_gap, 4),
        "eligible_for_release_gate": (
            sample_size >= 60
            and aggregate_gap <= 0.10
            and critical_agreement == 1.0
            and unsupported_agreement >= 0.90
        ),
    }


def load_joined_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("rows", []))
    return list(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate local judge scores against human review.")
    parser.add_argument("--joined-json", help="JSON rows with judge_* and human_* fields.")
    parser.add_argument("--run-id", help="Reserved for DB-backed calibration workflows.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.joined_json:
        rows = load_joined_json(Path(args.joined_json))
    else:
        rows = []

    report = calibrate(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote calibration report sample_size={report['sample_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
