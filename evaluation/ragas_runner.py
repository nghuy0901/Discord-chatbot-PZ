from __future__ import annotations

import os
from typing import Any, Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


RAGAS_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
]


def require_eval_config() -> Dict[str, str]:
    required = [
        "EVAL_LLM_PROVIDER",
        "EVAL_LLM_MODEL",
        "EVAL_EMBEDDING_PROVIDER",
        "EVAL_EMBEDDING_MODEL",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing RAGAS evaluator configuration: {missing}")
    return {name: os.environ[name] for name in required}


def build_ragas_dataset(rows: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": list(row.get("contexts") or []),
                "ground_truth": row["ground_truth"],
            }
            for row in rows
        ]
    )


def run_ragas_evaluation(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    require_eval_config()
    dataset = build_ragas_dataset(rows)
    result = evaluate(dataset, metrics=RAGAS_METRICS)
    return result.to_pandas().to_dict(orient="records")
