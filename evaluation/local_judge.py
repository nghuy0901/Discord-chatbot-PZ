from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class JudgeScore:
    correctness: float
    faithfulness: float
    unsupported_claim: bool
    critical_error: bool
    reason: str


def parse_judge_response(text: str) -> JudgeScore:
    data = json.loads(text)
    return JudgeScore(
        correctness=float(data["correctness"]),
        faithfulness=float(data["faithfulness"]),
        unsupported_claim=bool(data["unsupported_claim"]),
        critical_error=bool(data["critical_error"]),
        reason=str(data.get("reason", "")),
    )


def build_judge_prompt(
    *,
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
) -> List[dict]:
    system = (
        "You are a strict RAG evaluator. Judge only against the supplied "
        "ground truth and retrieved contexts. Output exactly one JSON object. "
        "Unsupported factual detail sets unsupported_claim=true. A contradiction "
        "of a server rule sets critical_error=true. Do not reward style, "
        "verbosity, or general plausibility."
    )
    user = {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "contexts": contexts,
        "schema": {
            "correctness": "float 0..1",
            "faithfulness": "float 0..1",
            "unsupported_claim": "boolean",
            "critical_error": "boolean",
            "reason": "short string",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


async def judge_answer(
    *,
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
) -> JudgeScore:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=os.environ["EVAL_LLM_BASE_URL"],
        api_key=os.getenv("EVAL_LLM_API_KEY", "local"),
    )
    response = await client.chat.completions.create(
        model=os.environ["EVAL_LLM_MODEL"],
        messages=build_judge_prompt(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts,
        ),
        temperature=0,
    )
    return parse_judge_response(response.choices[0].message.content or "{}")
