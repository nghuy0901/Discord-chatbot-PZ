CREATE TABLE IF NOT EXISTS approved_chat_sources (
  message_id TEXT PRIMARY KEY,
  channel_id TEXT NOT NULL,
  approved_by TEXT NOT NULL,
  approved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  approval_status TEXT NOT NULL DEFAULT 'approved'
    CHECK (approval_status IN ('approved', 'revoked')),
  redacted_content_hash TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT ''
);

ALTER TABLE rag_metrics
  ADD COLUMN IF NOT EXISTS rag_decision TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE rag_metrics
  ADD COLUMN IF NOT EXISTS decision_reason TEXT NOT NULL DEFAULT '';
ALTER TABLE rag_metrics
  ADD COLUMN IF NOT EXISTS provenance JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE rag_metrics
  ADD COLUMN IF NOT EXISTS trusted_source_count INT NOT NULL DEFAULT 0;
ALTER TABLE rag_metrics
  ADD COLUMN IF NOT EXISTS untrusted_source_count INT NOT NULL DEFAULT 0;
ALTER TABLE rag_metrics
  ADD COLUMN IF NOT EXISTS evidence_score FLOAT NOT NULL DEFAULT 0;

ALTER TABLE eval_runs
  ADD COLUMN IF NOT EXISTS prompt_version TEXT NOT NULL DEFAULT '';
ALTER TABLE eval_runs
  ADD COLUMN IF NOT EXISTS kb_version TEXT NOT NULL DEFAULT '';
ALTER TABLE eval_runs
  ADD COLUMN IF NOT EXISTS llm_model TEXT NOT NULL DEFAULT '';
ALTER TABLE eval_runs
  ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT '';
ALTER TABLE eval_runs
  ADD COLUMN IF NOT EXISTS retrieval_config_version TEXT NOT NULL DEFAULT '';
ALTER TABLE eval_runs
  ADD COLUMN IF NOT EXISTS split TEXT NOT NULL DEFAULT 'development';
ALTER TABLE eval_runs
  ADD COLUMN IF NOT EXISTS repeat_index INT NOT NULL DEFAULT 1;
ALTER TABLE eval_runs
  ADD COLUMN IF NOT EXISTS gate_status TEXT NOT NULL DEFAULT 'not_evaluated';

ALTER TABLE eval_items
  ADD COLUMN IF NOT EXISTS example_id TEXT NOT NULL DEFAULT '';
ALTER TABLE eval_items
  ADD COLUMN IF NOT EXISTS expected_behavior TEXT NOT NULL DEFAULT 'answer';
ALTER TABLE eval_items
  ADD COLUMN IF NOT EXISTS actual_behavior TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE eval_items
  ADD COLUMN IF NOT EXISTS provenance JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE eval_items
  ADD COLUMN IF NOT EXISTS latency_ms FLOAT NOT NULL DEFAULT 0;
ALTER TABLE eval_items
  ADD COLUMN IF NOT EXISTS error TEXT;

CREATE TABLE IF NOT EXISTS human_eval_reviews (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES eval_runs(run_id),
  example_id TEXT NOT NULL,
  reviewer_id TEXT NOT NULL,
  correctness FLOAT NOT NULL CHECK (correctness BETWEEN 0 AND 1),
  faithfulness FLOAT NOT NULL CHECK (faithfulness BETWEEN 0 AND 1),
  unsupported_claim BOOLEAN NOT NULL DEFAULT FALSE,
  critical_error BOOLEAN NOT NULL DEFAULT FALSE,
  behavior_correct BOOLEAN NOT NULL DEFAULT FALSE,
  notes TEXT NOT NULL DEFAULT '',
  reviewed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (run_id, example_id, reviewer_id)
);

CREATE TABLE IF NOT EXISTS beta_incidents (
  id BIGSERIAL PRIMARY KEY,
  query_id TEXT NOT NULL,
  message_id TEXT NOT NULL,
  reviewer_id TEXT NOT NULL,
  incident_type TEXT NOT NULL CHECK (
    incident_type IN (
      'factual_error',
      'unsupported_claim',
      'wrong_abstention',
      'pii_leak',
      'unapproved_source',
      'prompt_injection',
      'other'
    )
  ),
  severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'critical')),
  confirmed BOOLEAN NOT NULL DEFAULT FALSE,
  notes TEXT NOT NULL DEFAULT '',
  reviewed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
