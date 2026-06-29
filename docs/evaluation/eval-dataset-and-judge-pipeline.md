# Eval dataset & LLM-judge pipeline

How the NomNom RAG release evaluation is built and gated. This is the
source-of-truth for the dataset authoring flow and how the LLM-judge feeds the
release gate.

## End-to-end flow

```
 (1) inventory ──► (2) draft ──► (3) human review ──► (4) build ──► (5) validate
                                                                        │
                                                                        ▼
 (8) gate ◄── (7) calibrate judge ◄────────────────── (6) run release eval (+ judge)
```

### 1–2. Export trusted sources & draft candidates
```bash
python scripts/export_eval_candidates.py inventory --output tmp/inventory.json
python scripts/export_eval_candidates.py draft --inventory tmp/inventory.json --output tmp/candidates.jsonl
```
`inventory` scans the trusted KB (`pz`, `server_rules`) and approved chat,
redaction-checking every excerpt. `draft` emits unreviewed candidate rows.

### 3. Human review
A reviewer edits each candidate row, setting at minimum:
`review_status="approved"`, `expected_behavior` (answer/abstain/clarify),
`category` (answerable/unanswerable/ambiguous/adversarial), `language` (vi/en),
`ground_truth` + `expected_sources` (for answer rows), `expected_context_keywords`,
`critical`, `split` (development/holdout), and `reviewer`.

### 4. Build a schema-valid dataset
```bash
python scripts/build_eval_dataset.py \
  --candidates tmp/candidates.jsonl \
  --output evaluation/data/release_qa.v1.jsonl \
  --dataset-version public-v1
```
Only `review_status == "approved"` rows are emitted; the output is validated
against the `GoldenExample` schema before writing.

### 5. Validate
```bash
python scripts/validate_eval_dataset.py --dataset evaluation/data/release_qa.v1.jsonl
# release set must also pass the strict 300-row quota:
python scripts/validate_eval_dataset.py --dataset evaluation/data/release_qa.v1.jsonl --require-release-quota
```

### 6. Run the release eval (with the LLM-judge)
```bash
export EVAL_LLM_BASE_URL=...  EVAL_LLM_MODEL=...   # enables the judge
python scripts/run_release_eval.py --dataset evaluation/data/release_qa.v1.jsonl --split development --output evaluation/reports/dev.json
```
For each ANSWER item the judge scores `correctness`, `faithfulness`,
`unsupported_claim`, `critical_error` against the ground truth + retrieved
contexts. Judge failures are swallowed (item left unjudged). `summarize()`
aggregates `answer_correctness`, `faithfulness`, `unsupported_claim_rate`,
`critical_error_count` over judged items only. If `EVAL_LLM_*` is unset the
judge is skipped and only deterministic metrics are produced.

### 7. Calibrate the judge against humans
```bash
python scripts/import_human_review.py --csv human_reviews.csv      # → human_eval_reviews
python scripts/calibrate_local_judge.py --joined-json joined.json --output evaluation/reports/judge_calibration.json
```
Calibration is `eligible_for_release_gate` only when sample ≥ 60, aggregate
accuracy gap ≤ 0.10, critical agreement = 1.0, unsupported agreement ≥ 0.90.

### 8. Gate
`run_release_eval` exits non-zero when the gate fails. The gate enforces the
metrics in `evaluation/baselines/nomnom_public_v1.json` `minimums`/`maximums`.
Judge metrics (`faithfulness`, `unsupported_claim_rate`, `critical_error_count`)
live under `deferred_*` and are promoted into the **enforced** gate **only when**
`JUDGE_CALIBRATION_REPORT` points at a calibration report with
`eligible_for_release_gate=true`:
```bash
export JUDGE_CALIBRATION_REPORT=evaluation/reports/judge_calibration.json
```
An uncalibrated judge can never gate a release. The run summary records
`judge_calibrated` and `judge_metrics_enforced` for transparency.

## Current state (2026-06-29)
- **Seed dataset**: `evaluation/data/release_qa.v1.jsonl` has 20 rows covering all
  categories/behaviors/languages/splits. They are marked `approved_by:
  "maintainer-seed"` and must be re-reviewed by a domain human before they count
  toward a release. The strict 300-row release quota is **not** yet met.
- **Judge**: wired and calibration-gated, but needs `EVAL_LLM_*` configured and a
  calibration report before its metrics enforce the gate.
- Still pending for a true release gate: grow the dataset to the 300-row quota,
  collect ≥60 human reviews, and produce an eligible calibration report.
