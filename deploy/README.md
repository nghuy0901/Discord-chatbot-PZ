# NomNom Deployment Guide

This guide describes a Docker Compose deployment for the NomNom Discord bot, FastAPI API, PostgreSQL pgvector, and Redis.

## Architecture

```mermaid
graph TD
    Discord[Discord Users] --> Bot[NomNom Discord Bot]
    Client[API Clients] --> API[FastAPI API]
    Bot --> Postgres[(PostgreSQL + pgvector)]
    API --> Postgres
    Bot --> Redis[(Redis Cache)]
    API --> Redis
    Bot --> LLM[OpenAI, Gemini, or OpenAI-compatible /v1 endpoint]
    API --> LLM
```

## Environment

Create `.env` from `.env.example` and set production values:

```dotenv
DISCORD_BOT_TOKEN=
API_KEY=

LLM_PROVIDER=openai_compatible
LLM_MODEL=vllm-local
LLM_BASE_URL=https://llm.example.com/v1
LLM_API_KEY=

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_BASE_URL=https://llm.example.com/v1
EMBEDDING_API_KEY=
EMBEDDING_DIMENSION=1024

POSTGRES_USER=postgres
POSTGRES_PASSWORD=
POSTGRES_DB=postgres
POSTGRES_URL=postgresql://postgres:postgres@postgres:5432/postgres
REDIS_URL=redis://redis:6379/0
```

For self-hosted local models, run vLLM or another OpenAI-compatible server separately and point `LLM_BASE_URL` and `EMBEDDING_BASE_URL` at its `/v1` endpoint.

## Validate

```bash
docker compose -f deploy/docker-compose.prod.yml config
docker build -t nomnom-bot:prod .
```

## Start

```bash
docker compose -f deploy/docker-compose.prod.yml --env-file .env up -d --build
```

Check service status:

```bash
docker compose -f deploy/docker-compose.prod.yml ps
docker compose -f deploy/docker-compose.prod.yml logs -f nomnom-api
docker compose -f deploy/docker-compose.prod.yml logs -f nomnom-bot
```

## Migrations

Run migrations after PostgreSQL is available:

```bash
POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/postgres python scripts/migrate_db.py
```

In CI or a remote host, use the real PostgreSQL credentials instead of the example URL.

## Reverse Proxy

Expose only the API through your reverse proxy. Terminate TLS at the proxy and forward traffic to `localhost:8000`.

Minimal Nginx location:

```nginx
location / {
    proxy_pass http://localhost:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

## Operations

Back up PostgreSQL:

```bash
docker exec -t nomnom-postgres pg_dumpall -c -U postgres > backup.sql
```

Reload knowledge after updating `knowledge/docs/`:

```bash
curl -X POST https://api.example.com/api/admin/reload \
  -H "X-API-Key: your-secure-api-key"
```

Inspect metrics:

```bash
curl https://api.example.com/api/metrics \
  -H "X-API-Key: your-secure-api-key"
```
