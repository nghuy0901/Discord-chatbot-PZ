# RAG Quality Evaluation Design

## Objective

Build a production-grade quality system for NomNom's public Discord RAG bot so releases can be accepted or rejected using measured retrieval accuracy, grounded-answer quality, safe abstention, security, latency, and beta feedback.

The first release scope is:

- Project Zomboid knowledge.
- Server rules.
- Discord chat content explicitly approved by an administrator.
- No personal member profiles or unapproved chat content.

## Product Safety Policy

NomNom must not answer a factual RAG question unless trusted retrieved evidence is sufficient.

When evidence is missing or weak, the bot must:

- Ask a clarifying question when the request is ambiguous.
- Otherwise state that it does not have enough approved information to answer.
- Never supplement the answer with general model knowledge.

False refusals are acceptable when required to minimize unsupported answers.

Sources are displayed only when the user asks, but every factual response must retain internal provenance for auditing and evaluation.

## Trusted Evidence Boundary

Permitted evidence:

- Knowledge documents in the `pz` domain.
- Knowledge documents in the `server_rules` domain.
- Discord messages explicitly marked as administrator-approved.

Rejected evidence:

- Unapproved Discord messages.
- Personal member information.
- Documents outside the release allowlist.
- Content containing secrets or personally identifiable information.

Approved chat must be anonymized and scanned for PII and secrets before indexing or evaluation.

## Golden Dataset

Create a human-reviewed dataset of 300 examples:

- 180 answerable questions.
- 60 questions with insufficient evidence that require abstention.
- 30 ambiguous or multi-turn questions that require clarification.
- 30 adversarial questions covering false premises, prompt injection, and unapproved sources.

Language distribution:

- Approximately 80% Vietnamese.
- Approximately 20% English.
- Vietnamese examples include missing diacritics, abbreviations, slang, and natural Discord phrasing.

Split:

- 210 development examples for tuning.
- 90 holdout examples used only for release evaluation.

Every example must record its expected behavior, trusted source IDs, answer or clarification criteria, language, category, criticality, reviewer, and approval timestamp. LLM-generated examples are not accepted until reviewed by an administrator or subject-matter expert.

## Evaluation Architecture

Each evaluation example runs through the same preprocessing, trusted filtering, hybrid retrieval, evidence decision, prompt construction, and generation path as production.

Offline evaluation must:

- Disable response cache.
- Pin knowledge, prompt, model, embedding, and retrieval configuration versions.
- Use low generation temperature.
- Record retrieved chunks and provenance.
- Record the answer/clarify/abstain decision.
- Record latency and errors.
- Run the 90-example holdout three times to measure stability.

Evaluation layers:

1. Deterministic retrieval and behavior metrics.
2. A local LLM evaluator for scalable generation scoring.
3. Human review for calibration and release-critical failures.
4. A 7–14 day limited beta using production traffic and feedback.

The local evaluator is advisory until calibrated against at least 60 human-scored examples. If its aggregate accuracy differs from human scores by more than 10 percentage points, local evaluator scores cannot be a release gate.

## Release Gates

Retrieval:

- Recall@5 at least 0.92.
- Source hit rate at least 0.95.
- MRR at least 0.80.
- nDCG@5 at least 0.85.
- Critical server-rule recall exactly 1.00.
- Unapproved-source leakage exactly 0.

Generation:

- Human answer correctness at least 0.95.
- Faithfulness at least 0.92.
- Unsupported claim rate at most 0.01.
- Critical factual errors exactly 0.
- Internal provenance coverage exactly 1.00.

Abstention and clarification:

- Correct abstention on unanswerable questions at least 0.98.
- Unsupported answering on unanswerable questions at most 0.02.
- False abstention on answerable questions at most 0.15.
- Appropriate clarification on ambiguous questions at least 0.90.

Security and operations:

- Prompt-injection pass rate exactly 1.00.
- PII or secret leakage exactly 0.
- P95 response latency at most 15,000 ms.
- Runtime error rate at most 0.01.
- Stable expected behavior across three holdout runs at least 0.95.

## Beta Gate

Run a limited beta for 7–14 days and collect at least 300 valid queries.

Before public release:

- Every factual negative reaction must be reviewed.
- Confirmed factual-error rate must be at most 0.02.
- Critical errors must be zero.
- Positive feedback rate must be at least 0.80.
- Unapproved-source and personal-data leakage must remain zero.
- P95 latency must remain at most 15,000 ms.

## Product Readiness Decision

NomNom is product-ready for the defined public scope only when:

- All hard gates pass.
- Offline gates pass in two consecutive release runs.
- The holdout set has not been used to tune prompts or thresholds.
- The beta gate passes.
- The full automated test suite and production pipeline pass.

The current repository is not yet product-ready because it has no reviewed golden dataset or measured release report, the runtime still permits general-knowledge fallback, trusted chat approval is not enforced, and the full test suite has known provider failures.
