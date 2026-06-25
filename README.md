# NomNom

NomNom is a Discord-first RAG assistant with an optional FastAPI query API.

Implemented runtime features:

- Discord bot responses through mentions, replies, threads, and `/chat`
- Provider-neutral LLM calls through OpenAI, Gemini, Ollama Cloud, or OpenAI-compatible endpoints
- Local model serving through vLLM's OpenAI-compatible API
- PostgreSQL pgvector retrieval through LangChain PGVector
- Markdown/text knowledge ingestion
- BM25 + vector hybrid retrieval with Reciprocal Rank Fusion
- Redis response caching
- Basic developer metrics for latency, retrieval count, token usage, cache hit rate, and feedback
- Official RAGAS evaluation for faithfulness, answer relevancy, context precision, context recall, and answer correctness
- Deterministic retrieval accuracy metrics including Recall@k, Precision@k, MRR, nDCG@k, source hit rate, and keyword coverage

Not implemented:

- OCR or image document parsing
- LangGraph
- Durable queue workers
- Full production evaluation dashboard
- Automatic permission filtering for private knowledge sources

## Supported Knowledge Inputs

NomNom currently supports text-based knowledge sources:

- Markdown (`.md`)
- Plain text (`.txt`)
- reStructuredText (`.rst`)
- Discord JSON exports through the ingestion script

## Configuration

Copy `.env.example` to `.env` and configure the required secrets and providers.

Required runtime values:

- `DISCORD_BOT_TOKEN`
- `API_KEY`
- `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`
- `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_DIMENSION`
- `POSTGRES_URL`
- `REDIS_URL`

Provider values:

- `openai`
- `gemini`
- `openai_compatible`
- `ollama`

For self-hosted local models, serve the model through a vLLM OpenAI-compatible `/v1` endpoint and set `LLM_PROVIDER=openai_compatible`.

For Ollama Cloud, use the native Ollama API and configure:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3-coder-next:cloud
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your_ollama_api_key
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Discord bot:

```bash
python main.py
```

Run the FastAPI API:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Run with Docker Compose:

```bash
docker compose up -d
```

## API

The API requires `X-API-Key` on protected routes.

```bash
curl http://127.0.0.1:8000/api/health
curl -H "X-API-Key: $API_KEY" http://127.0.0.1:8000/api/metrics
curl -X POST http://127.0.0.1:8000/api/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is an axe?","domain":"pz","user_id":"u1","channel_id":"c1"}'
```

## Evaluation

Run deterministic retrieval metric tests:

```bash
python -m pytest tests/test_retrieval_metrics.py -q
```

Run official RAGAS evaluation only when evaluator provider credentials are configured:

```bash
python scripts/run_ragas_eval.py --dataset evaluation/data/release_qa.json --limit 50 --output evaluation/reports/release_report.json
```

## Production Checklist

- Set `API_KEY` to a non-default value of at least 32 characters.
- Set `LLM_PROVIDER`, `LLM_MODEL`, and provider credentials.
- Use vLLM for self-hosted local models through `/v1`.
- Run `python -m pytest -q`.
- Run `docker compose config`.
- Run RAGAS and deterministic retrieval evaluation before release.
- Verify `/api/health` reports Postgres and LLM healthy.
- Verify `/api/metrics` shows non-zero query and latency data after test traffic.
