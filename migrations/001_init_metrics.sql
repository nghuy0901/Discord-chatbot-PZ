CREATE TABLE IF NOT EXISTS rag_metrics (
  id BIGSERIAL PRIMARY KEY,
  query_id TEXT UNIQUE NOT NULL,
  timestamp TIMESTAMPTZ DEFAULT NOW(),
  channel_id TEXT,
  user_id TEXT,
  original_query TEXT,
  processed_query TEXT,
  query_language TEXT,
  detected_domain TEXT,
  retrieval_time_ms FLOAT,
  num_results INT,
  avg_similarity FLOAT,
  max_similarity FLOAT,
  min_similarity FLOAT,
  kb_results INT DEFAULT 0,
  chat_history_results INT DEFAULT 0,
  response_time_ms FLOAT,
  response_length INT,
  error TEXT,
  prompt_tokens INT DEFAULT 0,
  completion_tokens INT DEFAULT 0,
  total_tokens INT DEFAULT 0,
  estimated_cost_usd FLOAT DEFAULT 0,
  cache_hit BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS rag_feedback (
  id BIGSERIAL PRIMARY KEY,
  query_id TEXT REFERENCES rag_metrics(query_id),
  message_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  feedback_score INT NOT NULL,
  timestamp TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(message_id, user_id)
);
