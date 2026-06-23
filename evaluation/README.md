# NomNom RAG Evaluation

This folder stores offline RAG evaluation datasets and generated reports.

RAGAS metrics:

- Faithfulness
- Answer relevancy
- Context precision
- Context recall
- Answer correctness

Deterministic retrieval metrics:

- Recall@k
- Precision@k
- MRR
- nDCG@k
- Source hit rate
- Keyword coverage

Release gates:

- `faithfulness >= 0.80`
- `answer_correctness >= 0.75`
- `context_recall >= 0.80`
- `source_hit_rate >= 0.80`
- `empty_retrieval_rate <= 0.10`

RAGAS is required for evaluating, testing, and optimizing NomNom RAG. Do not remove it unless the RAG architecture itself is removed.

Execution modes:

- Unit tests: mock `ragas.evaluate` and validate dataset/report wiring.
- CI default: run deterministic retrieval metrics and schema tests.
- Release gate: run official RAGAS with `EVAL_LLM_PROVIDER`, `EVAL_LLM_MODEL`, `EVAL_EMBEDDING_PROVIDER`, and `EVAL_EMBEDDING_MODEL` configured.
