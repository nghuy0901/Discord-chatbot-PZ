# RAG Quality And Golden Dataset Design

## Scope

Continue the approved remediation order without adding new services or model calls:

1. Boost exact item/title matches inside the existing hybrid candidate list.
2. Clarify referential questions that have no resolvable subject.
3. Allow only direct semantic equivalents in groundedness checks.
4. Produce a reviewable 300-row release dataset using the existing JSONL schema.

## Retrieval Design

After RRF, candidates whose final heading segment or `record_name` appears as a complete phrase in the augmented query move ahead of non-exact candidates. Existing RRF order remains the tie-breaker. This distinguishes `Axe` from `Firefighter Axe` without introducing a reranker model.

## Clarification Design

The existing evidence policy remains the single clarification gate. Referential Vietnamese and English forms such as `cái này`, `cái đó`, `nó`, `this one`, and `it` clarify when the remaining words describe only a property or operation, not a concrete subject. Queries containing an explicit noun such as `axe`, `rìu`, or `generator` continue through retrieval.

## Groundedness Design

Generation and judge prompts may accept direct translation or deterministic semantic equivalence, for example `guarantees` ↔ `always` ↔ `100%`. Vague probability terms such as `likely` or `high chance` must not be converted into numbers. Thresholds and fail-closed behavior remain unchanged.

## Golden Dataset Design

Create `evaluation/data/release_qa.v2.jsonl` with exactly 300 rows and the quota already enforced by `evaluation.dataset_schema`:

- 180 answerable: 60 independently sourced fact groups with three paraphrases each.
- 60 unanswerable.
- 30 ambiguous.
- 30 adversarial: 24 abstain and 6 clarify.
- 210 development and 90 holdout, assigned by fact group to prevent paraphrase leakage.
- 240 Vietnamese and 60 English.

Answerable rows must name an existing source, contain a source-backed ground truth, and include useful context keywords. Rows are marked `approved_by=codex-draft` so human review is explicit rather than implied.

## Verification

- RED/GREEN unit tests for each behavior change.
- Dataset schema validation with release quotas enabled.
- Source existence validation against the active collection/source inventory.
- Duplicate ID/question and cross-split fact-group checks.
- Full pytest and a development release-eval sample after implementation.

