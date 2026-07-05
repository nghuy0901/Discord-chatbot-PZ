# RAG Hallucination Eval Script Proposal

Proposed script: `scripts/evaluate_rag_hallucination.py`

Inputs:

- `--dataset evaluation/rag_hallucination_eval_template.json`
- `--mode retrieval-only|generate|judge`
- `--limit N`
- `--repeat N` for variance checks
- `--output evaluation/reports/rag_hallucination_report.json`
- `--fail-on-gate` for CI/release use

## Required Runtime Capture

For each dataset row, capture:

- Query metadata: `query_id`, `domain`, `intent`, `expected_behavior`.
- Retrieval: raw KB/chat/tool contexts, source IDs, source URLs, domain, source kind, trust flag, similarity/RRF/BM25 scores, Self-RAG grades.
- Generation: final answer, cited IDs, citation validation result, token usage, latency, cache hit.
- Decision: `actual_behavior` (`answer` or `abstain`), `evidence_score`, `rag_decision`, `decision_reason`.
- Optional judge: RAGAS metrics plus local claim-support and abstention checks.

## Local Deterministic Metrics

- `retrieval_recall_at_5`: expected source appears in top 5 retrieved sources.
- `domain_mismatch_rate`: retrieved KB source domain not in expected/allowed domains.
- `source_url_metadata_coverage`: KB retrieved chunks with nonempty `source_url`.
- `untrusted_context_ratio`: retrieved factual contexts with `trusted=false` or `approval_status=unapproved`.
- `answer_citation_rate`: factual answer rows with at least `min_citations`.
- `citation_validity_rate`: cited IDs exist in retrieved evidence.
- `citation_support_precision`: answer claims are supported by cited chunks/tool JSON.
- `unsupported_claim_rate`: any material claim not supported by retrieved/tool evidence.
- `abstain_recall`: unanswerable rows where final behavior is abstain.
- `wrong_abstention_rate`: answerable rows where final behavior is abstain.
- `tool_numeric_mismatch_rate`: numbers in final answer not present in tool outputs.
- `cache_provenance_presence_rate`: cache hits that include provenance and validation status.

## Proposed Execution Flow

```python
async def evaluate_row(row):
    evidence = await run_retrieval(row["query"], domain=row.get("domain"))
    retrieval_scores = score_retrieval(row, evidence)

    if mode == "retrieval-only":
        return {**retrieval_scores}

    decision = decide_answer_or_abstain(row, evidence)
    answer = await generate_or_abstain(row, evidence, decision)
    citations = extract_citations(answer.text)
    citation_scores = validate_citations(citations, evidence)
    support_scores = validate_claim_support(answer.text, evidence, citations)

    return {
        "id": row["id"],
        "query": row["query"],
        "actual_behavior": answer.behavior,
        "retrieval": retrieval_scores,
        "citations": citation_scores,
        "support": support_scores,
        "latency_ms": answer.latency_ms,
        "cache_hit": answer.cache_hit,
    }
```

## Gate Defaults

- `embedding_config_ok == true`
- `kb_loaded_domain_count > 0`
- `domain_mismatch_rate == 0`
- `answer_citation_rate >= 0.98`
- `citation_validity_rate >= 0.98`
- `unsupported_claim_rate <= 0.05`
- `abstain_recall >= 0.95`
- `wrong_abstention_rate <= 0.05`
- `untrusted_context_ratio == 0` for factual rows unless explicitly allowed
- `tool_numeric_mismatch_rate == 0`

## Implementation Notes

- Do not rely only on RAGAS. Keep deterministic gates for citation and abstention because RAGAS will not reliably catch missing citation IDs or wrong source domains.
- Use a fake deterministic LLM in CI for citation/abstain unit tests, and use configured evaluator models only in release gates.
- Store per-row evidence in the report so a failed gate can be traced back to exact retrieved chunks and code paths.
- Cache hits must be evaluated separately because they bypass retrieval unless the cache stores provenance.
