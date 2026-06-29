import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["REDIS_CACHE_ENABLED"] = "false"

from evaluation.dataset_schema import GoldenExample, validate_dataset
from evaluation.reporting import EvaluationItemResult, EvaluationReport
from evaluation.gates import evaluate_gates
from evaluation.retrieval_metrics import (
    behavior_confusion,
    keyword_coverage,
    latency_percentiles,
    mean_reciprocal_rank,
    ndcg_at_k,
    provenance_coverage,
    recall_at_k,
    runtime_error_rate,
    source_hit_rate,
    unapproved_source_leakage_rate,
)
from rag.result import RAGDecision
from rag.responses import decision_response
from src.observability.request_context import RequestContext


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def versions() -> Dict[str, str]:
    return {
        "prompt_version": os.getenv("PROMPT_VERSION", "runtime"),
        "kb_version": os.getenv("KB_VERSION", "v1"),
        "llm_model": os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "")),
        "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
        "retrieval_config_version": os.getenv("RETRIEVAL_CONFIG_VERSION", "v1"),
    }


def deterministic_scores(example: GoldenExample, source_ids: List[str], context: str) -> Dict[str, float]:
    expected_sources = set(example.expected_sources)
    return {
        "recall_at_5": recall_at_k(source_ids, expected_sources, 5),
        "mrr": mean_reciprocal_rank(source_ids, expected_sources),
        "ndcg_at_5": ndcg_at_k(source_ids, expected_sources, 5),
        "source_hit_rate": source_hit_rate(source_ids, expected_sources),
        "keyword_coverage": keyword_coverage(context, example.expected_context_keywords),
    }


async def run_example(example: GoldenExample) -> EvaluationItemResult:
    start = time.time()
    try:
        from prompts.system_prompt import build_system_prompt
        from rag.retriever import build_rag_result
        from src.ollama_provider import chat_completion

        request_context = RequestContext.new(source="release_eval")
        rag_result = await build_rag_result(
            query=example.question,
            recent_messages=[],
            request_context=request_context,
        )
        if rag_result.decision is RAGDecision.ANSWER:
            messages = [
                {
                    "role": "system",
                    "content": build_system_prompt(
                        rag_context=rag_result.context,
                        domain_prompt=rag_result.domain_prompt,
                    ),
                },
                {"role": "user", "content": example.question},
            ]
            answer = await chat_completion(
                messages=messages,
                temperature=0.1,
                request_context=request_context,
            )
        else:
            answer = decision_response(rag_result.metric.query_language, rag_result.decision)

        provenance = [item.to_dict() for item in rag_result.provenance]
        source_ids = [item["source_id"] for item in provenance]
        latency_ms = (time.time() - start) * 1000
        return EvaluationItemResult(
            example_id=example.id,
            expected_behavior=example.expected_behavior.value,
            actual_behavior=rag_result.decision.value,
            answer=answer,
            retrieved_source_ids=source_ids,
            provenance=provenance,
            deterministic_scores=deterministic_scores(example, source_ids, rag_result.context),
            judge_scores={},
            latency_ms=latency_ms,
        )
    except Exception as exc:
        return EvaluationItemResult(
            example_id=example.id,
            expected_behavior=example.expected_behavior.value,
            actual_behavior="error",
            answer="",
            retrieved_source_ids=[],
            provenance=[],
            deterministic_scores={},
            judge_scores={},
            latency_ms=(time.time() - start) * 1000,
            error=str(exc),
        )


def summarize(items: List[EvaluationItemResult]) -> Dict[str, float]:
    if not items:
        return {}
    summary: Dict[str, float] = {}
    for key in ("recall_at_5", "mrr", "ndcg_at_5", "source_hit_rate", "keyword_coverage"):
        summary[key] = sum(item.deterministic_scores.get(key, 0.0) for item in items) / len(items)
    summary.update(
        behavior_confusion(
            [item.expected_behavior for item in items],
            [item.actual_behavior for item in items],
        )
    )
    summary.update(latency_percentiles([item.latency_ms for item in items]))
    summary["unapproved_source_leakage_rate"] = unapproved_source_leakage_rate(
        [item.provenance for item in items]
    )
    summary["provenance_coverage"] = provenance_coverage(
        [item.actual_behavior for item in items],
        [item.provenance for item in items],
    )
    summary["runtime_error_rate"] = runtime_error_rate([item.error for item in items])
    return {key: round(value, 4) for key, value in summary.items()}


def load_gate_config() -> Dict[str, Any]:
    path = ROOT_DIR / "evaluation" / "baselines" / "nomnom_public_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def apply_gate_verdict(report: EvaluationReport) -> None:
    gate_result = evaluate_gates(report.summary, load_gate_config())
    report.summary["gate_status"] = "pass" if gate_result.passed else "fail"
    report.summary["product_ready"] = False
    report.summary["gate_failure_count"] = len(gate_result.failures)
    report.summary["gate_failures"] = gate_result.failures


async def run_once(rows: List[Dict[str, Any]], split: str, repeat_index: int) -> EvaluationReport:
    examples = [
        GoldenExample.from_dict(row)
        for row in rows
        if row.get("split") == split
    ]
    items = [await run_example(example) for example in examples]
    dataset_version = examples[0].dataset_version if examples else "unknown"
    return EvaluationReport(
        run_id=f"release-{split}-{repeat_index}-{uuid.uuid4().hex[:8]}",
        dataset_version=dataset_version,
        split=split,
        repeat_index=repeat_index,
        versions=versions(),
        summary=summarize(items),
        items=items,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run production-path RAG release evaluation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=["development", "holdout"], required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset))
    errors = validate_dataset(rows, require_release_quota=False)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    reports = []
    for repeat_index in range(1, args.repeat + 1):
        report = await run_once(rows, args.split, repeat_index)
        apply_gate_verdict(report)
        reports.append(report)
        if args.output_dir:
            report.write_json(Path(args.output_dir) / f"{report.run_id}.json")

    if args.output:
        if len(reports) == 1:
            reports[0].write_json(args.output)
        else:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps([report.to_dict() for report in reports], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # The gate is now authoritative: a failing gate fails the process so CI (or
    # any caller) actually blocks the release instead of going green (audit C5).
    gate_failed = any(
        report.summary.get("gate_status") == "fail" for report in reports
    )
    print(f"completed {len(reports)} release eval run(s)")
    if gate_failed:
        failed_runs = [
            r.run_id for r in reports if r.summary.get("gate_status") == "fail"
        ]
        print(
            f"RELEASE GATE FAILED for {len(failed_runs)} run(s): "
            + ", ".join(failed_runs),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
