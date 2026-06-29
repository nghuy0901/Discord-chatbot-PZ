# RAG Quality Evaluation Handoff

Date: 2026-06-26

## Context

The work follows these two planning/design files:

- `docs/superpowers/specs/2026-06-25-rag-quality-evaluation-design.md`
- `docs/superpowers/plans/2026-06-25-rag-quality-evaluation.md`

The goal is to make NomNom's public RAG release measurable and safe: trusted-source-only retrieval, evidence-based answer/clarify/abstain decisions, reviewed golden dataset, deterministic release evaluation, local judge calibration, release gates, beta gate, and final product-readiness verdict.

## What Was Discussed

- The repository was checked against the RAG quality design and implementation plan.
- Existing completed work was identified and marked in the plan.
- The current failure in the full test suite was investigated. Root cause: `tests/test_branding.py` scanned `.worktrees`, so it picked up legacy branding from an internal worktree, not the active project. The test now ignores `.worktrees`.
- The user asked which tasks Codex can perform, which tasks require user/admin review, and how review/import/confirmation should happen.
- Codex proceeded with all implementation work that can be completed without human review data, real beta traffic, or a final release decision.

## What Was Implemented

### Task 1: Trust, Provenance, Decision Schema

- Added migration `migrations/004_add_rag_trust_and_quality.sql`.
- Extended `RAGMetric` with:
  - `rag_decision`
  - `decision_reason`
  - `provenance`
  - `trusted_source_count`
  - `untrusted_source_count`
  - `evidence_score`
- Added migration and metrics persistence tests.

### Task 2: Redaction And Admin-Approved Chat

- Added `knowledge/redaction.py` for email, phone, secret, and Discord mention redaction.
- Added `admin/chat_approval.py` for approve/revoke workflow.
- Added `scripts/approve_chat_source.py`.
- Newly ingested chat metadata now defaults to unapproved/untrusted.
- Added redaction and approval tests.

### Task 3: Trusted Source Filtering

- Added `rag/trust.py`.
- Restricted trusted KB domains to `pz` and `server_rules`.
- Enforced approved/trusted chat filters in vector, BM25, hybrid retrieval, and router fallback.
- Marked KB chunks with trust metadata.
- Added trust filter tests.

### Task 4: Evidence Decision Runtime

- Added:
  - `rag/result.py`
  - `rag/evidence.py`
  - `rag/responses.py`
- Added `RAGBuildResult`, `RAGDecision`, provenance items, and deterministic `EvidencePolicy`.
- Refactored `build_rag_context` to wrap `build_rag_result`.
- Added evidence policy tests.

### Task 5: Runtime Abstention

- Replaced permissive RAG prompt with strict no-general-knowledge instructions.
- API now uses `build_rag_result`.
- API returns `decision`, `source_count`, and optional `sources`.
- API skips LLM/tool calls for `abstain` and `clarify`.
- Discord path sends deterministic abstain/clarify responses directly.
- Cache now stores only `answer` decisions with non-empty provenance.
- Added API abstention E2E test.

### Task 6: Golden Dataset Schema V2

- Reworked `evaluation/dataset_schema.py` for v2 dataset fields and validation rules.
- Added `ExpectedBehavior`.
- Added release quota validation logic.
- Added `scripts/validate_eval_dataset.py`.
- Added initial seed dataset: `evaluation/data/release_qa.v1.jsonl`.
- Added dataset v2 tests.

### Task 7: Candidate Dataset Tooling

- Added `scripts/export_eval_candidates.py`.
- Exported trusted source inventory to `evaluation/data/source_inventory.v1.json`.
- Generated draft candidate questions to `tmp/eval_candidates.jsonl`.
- Added `docs/evaluation/dataset-review-guide.md`.

Important: draft candidates are not approved golden rows.

### Task 8: Production-Path Evaluation

- Added behavior metrics and latency/provenance helpers.
- Added `evaluation/reporting.py`.
- Added `scripts/run_release_eval.py`.
- Updated `scripts/run_ragas_eval.py` so it remains optional and is not used as the release gate.
- Added reporting tests.

### Task 9: Local Judge And Calibration Tooling

- Added `evaluation/local_judge.py`.
- Added parser tests for strict JSON judge output.
- Updated `evaluation/ragas_runner.py` to use explicit local OpenAI-compatible clients.
- Pinned evaluator dependencies in `requirements.txt`:
  - `ragas==0.2.15`
  - `datasets==3.2.0`
  - `pandas==2.2.3`
- Added:
  - `scripts/import_human_review.py`
  - `scripts/calibrate_local_judge.py`
  - `evaluation/data/human_review_template.csv`

Important: `pip install -r requirements.txt` has not been run after pinning.

### Task 10: Release Gates

- Added strict gate config: `evaluation/baselines/nomnom_public_v1.json`.
- Added `evaluation/gates.py`.
- Integrated gate verdict into release eval reports.
- `product_ready` remains fail-closed until all human/beta/holdout requirements are met.

### Task 11: Beta Gate

- Added `scripts/summarize_beta.py`.
- Added beta summary tests.
- Added exact `bot_message_id -> query_id` mapping for feedback, reducing risk of reaction feedback attaching to the wrong answer.

### Task 12: CI And Runbook

- Updated `.github/workflows/ci.yml`.
- Added self-hosted `rag-release-eval` job gated by `RUN_RAG_RELEASE_EVAL`.
- Updated `.env.example`.
- Added `docs/evaluation/rag-quality-runbook.md`.
- Updated `evaluation/README.md`.

Note: PR CI validates the current seed dataset without release quota. Full `--require-release-quota` is reserved for release validation after Task 7 is completed.

### Task 13: Verification So Far

- Full automated suite was run.
- Static trust-boundary checks were run.

## Verification Results

Latest full test result:

```text
python -m pytest -q
120 passed, 22 warnings
```

Static trust-boundary checks:

- No permissive runtime instruction such as `MAY use your general knowledge` found in `prompts`, `rag`, `knowledge`, `api`, or `src`.
- Trust enforcement appears in DB metadata, BM25, KB manager/router, API, Discord client, and RAG runtime.

## Current Important Files

- `docs/superpowers/plans/2026-06-25-rag-quality-evaluation.md`
- `docs/evaluation/rag-quality-runbook.md`
- `docs/evaluation/dataset-review-guide.md`
- `evaluation/data/release_qa.v1.jsonl`
- `evaluation/data/source_inventory.v1.json`
- `tmp/eval_candidates.jsonl`
- `evaluation/data/human_review_template.csv`
- `evaluation/baselines/nomnom_public_v1.json`

## What The User/Admin Needs To Do

### 1. Review Golden Dataset Rows

Review location:

- `evaluation/data/release_qa.v1.jsonl`
- Draft candidates: `tmp/eval_candidates.jsonl`
- Review instructions: `docs/evaluation/dataset-review-guide.md`

For every approved row, verify:

- Question is natural Discord language.
- Expected behavior is correct: `answer`, `abstain`, or `clarify`.
- Ground truth is fully supported by trusted sources.
- `expected_sources` point to approved/trusted evidence.
- PII/secrets are absent.
- `approved_by` and `approved_at` are real.

### 2. Fill Exact Dataset Quotas

The final release dataset must contain exactly 300 reviewed rows:

- 180 answerable rows with behavior `answer`.
- 60 unanswerable rows with behavior `abstain`.
- 30 ambiguous rows with behavior `clarify`.
- 30 adversarial rows: 24 `abstain`, 6 `clarify`.
- 210 development rows.
- 90 holdout rows.
- Vietnamese ratio between 75% and 85%.
- At least 30 server-rule examples, including all critical rules.

Validation command:

```powershell
python scripts/validate_eval_dataset.py --dataset evaluation/data/release_qa.v1.jsonl --require-release-quota
```

### 3. Provide Human Review Scores

Use:

- `evaluation/data/human_review_template.csv`

Create a completed CSV such as:

- `evaluation/data/human_review_completed.csv`

Required columns:

```text
run_id,example_id,reviewer_id,correctness,faithfulness,unsupported_claim,critical_error,behavior_correct,notes
```

Import command:

```powershell
python scripts/import_human_review.py --csv evaluation/data/human_review_completed.csv
```

At least 60 reviewed examples are required for local judge calibration.

### 4. Provide Local Evaluator Endpoint

Set local OpenAI-compatible evaluator config in `.env`:

```dotenv
EVAL_LLM_PROVIDER=openai_compatible
EVAL_LLM_MODEL=
EVAL_LLM_BASE_URL=http://127.0.0.1:8001/v1
EVAL_LLM_API_KEY=local
EVAL_EMBEDDING_PROVIDER=openai_compatible
EVAL_EMBEDDING_MODEL=
EVAL_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
EVAL_EMBEDDING_API_KEY=local
```

Then run calibration after development evaluation:

```powershell
$developmentRunId = (Get-Content evaluation/reports/development.json -Raw | ConvertFrom-Json).run_id
python scripts/calibrate_local_judge.py --run-id $developmentRunId --output evaluation/reports/judge-calibration.json
```

### 5. Run Real Release Evaluation

After dataset and development tuning are frozen:

```powershell
python scripts/run_release_eval.py --dataset evaluation/data/release_qa.v1.jsonl --split holdout --repeat 3 --output-dir evaluation/reports/holdout
```

Holdout results must not be used to tune the same release candidate.

### 6. Run Beta And Provide Real Counts

After 7-14 days and at least 300 valid production queries:

```powershell
python scripts/summarize_beta.py --days 14 --output evaluation/reports/beta-public-v1.json
```

If raw counts are provided manually, include:

- total queries
- positive feedback
- negative feedback
- reviewed factual negatives
- confirmed factual errors
- critical incidents
- PII leaks
- unapproved source leaks
- p95 latency

### 7. Confirm Data Authenticity

Before product-ready verdict, the user/admin should explicitly confirm:

- The 300-row dataset was reviewed by real reviewer IDs.
- Holdout was not used for tuning.
- Human review CSV was produced by real reviewers.
- Beta report is from real production traffic.
- Critical errors, PII leaks, and unapproved-source leaks are zero.

## Still Not Complete

These remain unchecked because they require user/admin data, external services, or a commit:

- Commit steps for completed tasks.
- Task 7 Step 4-6: exact 300-row reviewed dataset.
- Task 9 Step 1: run `pip install -r requirements.txt` and verify `ragas==0.2.15`.
- Task 13 Step 3: run migrations twice against an empty test PostgreSQL database.
- Task 13 Step 4: trust leakage E2E against real retrieval setup.
- Task 13 Step 5-9: development eval, human calibration, holdout, consecutive run, beta gate.
- Task 13 Step 10: final product-readiness decision.

## Current Product Readiness

Current verdict remains:

```json
{"product_ready": false}
```

Reason: tooling is implemented and tests pass, but reviewed 300-row dataset, human calibration, holdout repeats, beta report, and final release confirmation are not complete.
