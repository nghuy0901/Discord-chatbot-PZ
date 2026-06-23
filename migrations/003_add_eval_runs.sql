CREATE TABLE IF NOT EXISTS eval_runs (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT UNIQUE NOT NULL,
  dataset_name TEXT NOT NULL,
  dataset_version TEXT NOT NULL,
  evaluator_model TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  summary JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS eval_items (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT REFERENCES eval_runs(run_id),
  question TEXT NOT NULL,
  ground_truth TEXT NOT NULL,
  answer TEXT NOT NULL,
  contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
  scores JSONB NOT NULL DEFAULT '{}'::jsonb
);
