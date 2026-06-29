import argparse
import json
from pathlib import Path
from typing import Any, Dict


def summarize_beta(
    *,
    total_queries: int,
    positive_feedback: int,
    negative_feedback: int,
    reviewed_factual_negatives: int,
    confirmed_factual_errors: int,
    critical_incidents: int,
    pii_leaks: int,
    unapproved_source_leaks: int,
    p95_latency_ms: float,
) -> Dict[str, Any]:
    feedback_total = max(positive_feedback + negative_feedback, 1)
    confirmed_factual_error_rate = (
        confirmed_factual_errors / total_queries if total_queries else 1.0
    )
    positive_feedback_rate = positive_feedback / feedback_total

    raw_counts = {
        "total_queries": total_queries,
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback,
        "reviewed_factual_negatives": reviewed_factual_negatives,
        "confirmed_factual_errors": confirmed_factual_errors,
        "critical_incidents": critical_incidents,
        "pii_leaks": pii_leaks,
        "unapproved_source_leaks": unapproved_source_leaks,
        "p95_latency_ms": p95_latency_ms,
    }
    metrics = {
        "confirmed_factual_error_rate": round(confirmed_factual_error_rate, 4),
        "positive_feedback_rate": round(positive_feedback_rate, 4),
    }
    failures = {}
    if total_queries < 300:
        failures["total_queries"] = {"actual": total_queries, "required": 300}
    if reviewed_factual_negatives != negative_feedback:
        failures["reviewed_factual_negatives"] = {
            "actual": reviewed_factual_negatives,
            "required": negative_feedback,
        }
    if confirmed_factual_error_rate > 0.02:
        failures["confirmed_factual_error_rate"] = {
            "actual": round(confirmed_factual_error_rate, 4),
            "required": 0.02,
        }
    if critical_incidents != 0:
        failures["critical_incidents"] = {"actual": critical_incidents, "required": 0}
    if positive_feedback_rate < 0.80:
        failures["positive_feedback_rate"] = {
            "actual": round(positive_feedback_rate, 4),
            "required": 0.80,
        }
    if pii_leaks != 0:
        failures["pii_leaks"] = {"actual": pii_leaks, "required": 0}
    if unapproved_source_leaks != 0:
        failures["unapproved_source_leaks"] = {
            "actual": unapproved_source_leaks,
            "required": 0,
        }
    if p95_latency_ms > 15000:
        failures["p95_latency_ms"] = {"actual": p95_latency_ms, "required": 15000}

    return {
        "passed": not failures,
        "failures": failures,
        "metrics": metrics,
        "raw_counts": raw_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize NomNom public beta quality gates.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-queries", type=int, default=0)
    parser.add_argument("--positive-feedback", type=int, default=0)
    parser.add_argument("--negative-feedback", type=int, default=0)
    parser.add_argument("--reviewed-factual-negatives", type=int, default=0)
    parser.add_argument("--confirmed-factual-errors", type=int, default=0)
    parser.add_argument("--critical-incidents", type=int, default=0)
    parser.add_argument("--pii-leaks", type=int, default=0)
    parser.add_argument("--unapproved-source-leaks", type=int, default=0)
    parser.add_argument("--p95-latency-ms", type=float, default=0.0)
    args = parser.parse_args()

    report = summarize_beta(
        total_queries=args.total_queries,
        positive_feedback=args.positive_feedback,
        negative_feedback=args.negative_feedback,
        reviewed_factual_negatives=args.reviewed_factual_negatives,
        confirmed_factual_errors=args.confirmed_factual_errors,
        critical_incidents=args.critical_incidents,
        pii_leaks=args.pii_leaks,
        unapproved_source_leaks=args.unapproved_source_leaks,
        p95_latency_ms=args.p95_latency_ms,
    )
    report["window_days"] = args.days
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote beta summary passed={report['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
