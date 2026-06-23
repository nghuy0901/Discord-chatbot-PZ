import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from evaluation.dataset_schema import validate_example
from evaluation.ragas_runner import run_ragas_evaluation

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ragas_evaluator")


async def execute_naive_rag(query: str, domain: Optional[str] = None) -> Dict[str, Any]:
    start_time = time.time()
    try:
        from rag.db import get_vectorstore
        from src.ollama_provider import chat_completion

        store = get_vectorstore()
        filter_dict = {"domain": domain} if domain else None
        docs = store.similarity_search(query, k=3, filter=filter_dict)
        contexts = [doc.page_content for doc in docs]
        context_text = "\n\n".join(contexts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer only from the retrieved "
                    f"context.\n\n{context_text}"
                ),
            },
            {"role": "user", "content": query},
        ]
        answer = await chat_completion(messages, temperature=0.8)
        return {
            "answer": answer,
            "contexts": contexts,
            "latency_ms": (time.time() - start_time) * 1000,
        }
    except Exception as exc:
        logger.error("Naive RAG execution failed: %s", exc)
        return {"answer": f"Error: {exc}", "contexts": [], "latency_ms": 0}


async def execute_optimized_rag(query: str, domain: Optional[str] = None) -> Dict[str, Any]:
    start_time = time.time()
    try:
        from prompts.system_prompt import build_system_prompt
        from rag.retriever import build_rag_context
        from src.ollama_provider import chat_completion
        from utils.cache import get_query_cache

        cache = get_query_cache()
        cached = await cache.get(query, domain)
        if cached:
            return {
                "answer": cached["response_text"],
                "contexts": ["CACHED_HIT"],
                "latency_ms": (time.time() - start_time) * 1000,
                "cached": True,
            }

        rag_context, domain_prompt_text, metric = await build_rag_context(
            query=query,
            recent_messages=None,
            channel_name=domain,
        )
        domain_prompt = ""
        if domain_prompt_text:
            domain_prompt = f"\n# Domain-Specific Instructions\n{domain_prompt_text}\n"

        messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    rag_context=rag_context,
                    domain_prompt=domain_prompt,
                ),
            },
            {"role": "user", "content": query},
        ]
        answer = await chat_completion(messages, temperature=0.8)

        if metric.query_intent in ("analytical", "hybrid", "narrative"):
            await cache.set(
                query=query,
                domain=metric.detected_domain,
                result={"response_text": answer, "prompt_tokens": 0, "completion_tokens": 0},
            )

        return {
            "answer": answer,
            "contexts": [rag_context] if rag_context else [],
            "latency_ms": (time.time() - start_time) * 1000,
            "cached": False,
        }
    except Exception as exc:
        logger.error("Optimized RAG execution failed: %s", exc)
        return {"answer": f"Error: {exc}", "contexts": [], "latency_ms": 0, "cached": False}


def _row_for_ragas(example, run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": example.question,
        "answer": run["answer"],
        "contexts": run["contexts"],
        "ground_truth": example.ground_truth,
        "latency_ms": run["latency_ms"],
        "cached": run.get("cached", False),
        "expected_sources": example.expected_sources,
        "expected_context_keywords": example.expected_context_keywords,
    }


def _aggregate_ragas(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key not in {"latency_ms"}
    ]
    return {
        key: round(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows), 4)
        for key in numeric_keys
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run mandatory official RAGAS evaluation.")
    parser.add_argument("--dataset", default="evaluation/data/qa_dataset.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default="evaluation/reports/comparison_report.json")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(
            f"Dataset file not found: {args.dataset}. "
            "Run scripts/generate_eval_dataset.py first."
        )

    with open(args.dataset, "r", encoding="utf-8") as file:
        raw_items = json.load(file)

    examples = [validate_example(item) for item in raw_items[: args.limit]]
    naive_rows: List[Dict[str, Any]] = []
    optimized_rows: List[Dict[str, Any]] = []

    try:
        from rag.db import init_db

        await init_db()
    except Exception as exc:
        logger.warning("RAG DB init skipped or failed: %s", exc)

    for idx, example in enumerate(examples, start=1):
        logger.info("[%s/%s] %s", idx, len(examples), example.question)
        naive_rows.append(_row_for_ragas(example, await execute_naive_rag(example.question, example.domain)))
        optimized_rows.append(
            _row_for_ragas(example, await execute_optimized_rag(example.question, example.domain))
        )

    logger.info("Running official RAGAS for naive RAG...")
    naive_scores = run_ragas_evaluation(naive_rows)
    logger.info("Running official RAGAS for optimized RAG...")
    optimized_scores = run_ragas_evaluation(optimized_rows)

    report = {
        "metadata": {
            "eval_count": len(examples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summaries": {
            "naive_rag": _aggregate_ragas(naive_scores),
            "optimized_rag": _aggregate_ragas(optimized_scores),
        },
        "details": {
            "naive": naive_rows,
            "optimized": optimized_rows,
            "naive_ragas": naive_scores,
            "optimized_ragas": optimized_scores,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    logger.info("RAGAS report written to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
