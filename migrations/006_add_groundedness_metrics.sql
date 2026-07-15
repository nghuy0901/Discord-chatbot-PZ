ALTER TABLE rag_metrics
  ADD COLUMN IF NOT EXISTS groundedness_score FLOAT NOT NULL DEFAULT 1.0;
ALTER TABLE rag_metrics
  ADD COLUMN IF NOT EXISTS groundedness_reason TEXT NOT NULL DEFAULT '';
ALTER TABLE rag_metrics
  ADD COLUMN IF NOT EXISTS groundedness_unsupported_count INT NOT NULL DEFAULT 0;
ALTER TABLE rag_metrics
  ADD COLUMN IF NOT EXISTS groundedness_error TEXT;
