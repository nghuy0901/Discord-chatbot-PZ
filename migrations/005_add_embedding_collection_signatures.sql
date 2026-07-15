CREATE TABLE IF NOT EXISTS rag_embedding_collection_signatures (
  collection_name    TEXT PRIMARY KEY,
  provider           TEXT NOT NULL,
  model              TEXT NOT NULL,
  dimension          INT  NOT NULL,
  collection_version TEXT NOT NULL,
  input_mode         TEXT NOT NULL,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
