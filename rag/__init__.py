"""
RAG (Retrieval-Augmented Generation) module for Discord bot.

Components:
- db.py: PostgreSQL + pgvector connection and table management
- embedder.py: Embedding generation via configured embedding provider
- ingest.py: Historical conversation dataset ingestion
- retriever.py: Semantic search and context retrieval (orchestrator)
- bm25_search.py: BM25 lexical search for hybrid RAG (E1)
- hybrid_retriever.py: Reciprocal Rank Fusion of BM25 + Vector (E1)
- self_rag.py: LLM-based relevance grading of retrieval results (E2)
- query_preprocessor.py: Query cleaning and enrichment (A2)
- metrics.py: RAG pipeline performance metrics (A3)

Pipeline (E1 + E2):
    Query → Preprocess → [Vector Search + BM25 Search] → RRF Fusion → Self-RAG Grading → Prompt
"""
