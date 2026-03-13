# CLCT Discord Bot — Setup Guide

> **CLCT** — inspired by Jarvis and the Hitchhiker's Guide to the Galaxy.
> A Grok-style Discord bot powered by local Ollama models with RAG from historical conversations.

---

## Architecture Overview

```
┌──────────────┐    ┌────────────┐    ┌──────────────────┐
│  Discord API │◄──►│  CLCT Bot   │◄──►│  Ollama (local)  │
│              │    │  (bot.py)   │    │  llama3.1:8b     │
└──────────────┘    └─────┬──────┘    └──────────────────┘
                          │
                    ┌─────┴──────┐
                    │  RAG System │
                    │  (pgvector) │
                    └─────┬──────┘
                          │
                   ┌──────┴───────┐
                   │  PostgreSQL   │
                   │  + pgvector   │
                   └──────────────┘
```

### Key Components

| File / Module | Purpose |
|---|---|
| `main.py` | Entry point — validates env, launches bot |
| `src/bot.py` | Discord event handlers, slash commands, trigger logic |
| `src/aclient.py` | `CLCTClient` class — core bot logic, streaming, prompt building |
| `src/ollama_provider.py` | Async Ollama chat completion + streaming |
| `utils/context_manager.py` | Per-channel rolling message windows, implicit reply detection, prompt assembly |
| `rag/db.py` | LangChain PGVector store + asyncpg (edges & counts) |
| `rag/embedder.py` | LangChain OllamaEmbeddings wrapper |
| `rag/ingest.py` | CLI script to ingest historical conversation JSON (via LangChain) |
| `rag/retriever.py` | Semantic search + citation formatting (via LangChain PGVector) |

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — [Install Ollama](https://ollama.com/download)
3. **PostgreSQL 15+** with **pgvector** extension
4. **Discord Bot Token** — [Discord Developer Portal](https://discord.com/developers/applications)

---

## Step 1: Install Ollama & Pull Models

```bash
# Install Ollama (follow https://ollama.com/download for your OS)

# Pull the chat model
ollama pull llama3.1:8b

# Pull the embedding model (required for RAG)
ollama pull nomic-embed-text

# Verify
ollama list
```

**Optional models:**
- `llama3.1:70b` — higher quality, needs ~40GB VRAM
- `llava:13b` — multimodal (image + text)
- `mistral:7b` — fast alternative

---

## Step 2: Set Up PostgreSQL + pgvector

### Option A: Docker (recommended)

```bash
# Start PostgreSQL with pgvector (included in docker-compose.yml)
docker compose up -d postgres

# Verify
docker exec -it clct-postgres psql -U clct -d clct_rag -c "SELECT 1;"
```

### Option B: Native Installation

```bash
# Ubuntu/Debian
sudo apt install postgresql-16 postgresql-16-pgvector

# macOS (Homebrew)
brew install postgresql@16
brew install pgvector

# Windows (use installers)
# 1. Install PostgreSQL from https://www.postgresql.org/download/windows/
# 2. Install pgvector: https://github.com/pgvector/pgvector#windows
```

Then create the database:

```sql
-- Connect as superuser
psql -U postgres

CREATE USER clct WITH PASSWORD 'clct';
CREATE DATABASE clct_rag OWNER clct;
\c clct_rag
CREATE EXTENSION IF NOT EXISTS vector;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO clct;

-- Verify pgvector
SELECT * FROM pg_extension WHERE extname = 'vector';
```

---

## Step 3: Configure Environment

```bash
# Copy the example config
cp .env.example .env

# Edit with your values
# At minimum, set:
#   DISCORD_BOT_TOKEN=your_token
#   POSTGRES_URL=postgresql://clct:clct@localhost:5432/clct_rag
```

Key settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | *(required)* | Your Discord bot token |
| `OLLAMA_MODEL` | `llama3.1:8b` | Chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `POSTGRES_URL` | `postgresql://clct:clct@localhost:5432/clct_rag` | PostgreSQL connection |
| `ENABLE_RAG` | `true` | Enable/disable RAG retrieval |
| `ENABLE_STREAMING` | `true` | Stream responses token-by-token |
| `ENABLE_IMPLICIT_REPLIES` | `false` | Detect implicit replies via similarity |
| `CONTEXT_WINDOW_SIZE` | `30` | Messages per channel context window |
| `PERSONALITY_TEMPERATURE` | `0.8` | LLM temperature (0.0–1.0) |

---

## Step 4: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Step 5: Ingest Historical Conversations (RAG)

The ingest script processes your historical Discord conversation dataset (JSON format) and stores embeddings in PostgreSQL.

### Expected JSON Format

The ingester is flexible and handles multiple formats. Ideal structure:

```json
[
  {
    "id": "1234567890",
    "channel_id": "9876543210",
    "author": {"id": "111", "name": "Alice"},
    "content": "Hello world!",
    "timestamp": "2024-01-15T10:30:00Z",
    "reference": {"message_id": "1234567889"},
    "mentions": [{"id": "222"}]
  }
]
```

Also supports: flat `author_id`/`author_name` fields, Unix timestamps, `messages` wrapper key.

### Run Ingestion

```bash
# Basic usage
python -m rag.ingest path/to/conversations.json

# With options
python -m rag.ingest data/chat_export.json --batch-size 50 --model nomic-embed-text

# Example with the existing dataset in the repo
python -m rag.ingest "Chấn_thương_tâm_lý_cư-dân-chat_2026-02-11_16-02-00.json"
```

**What it does:**
1. Loads and normalises messages from the JSON file
2. Adds documents to LangChain PGVector store (embedding generated automatically by LangChain OllamaEmbeddings)
3. Records reply/mention edges in `message_edges` table
4. Prints a summary: total loaded, normalised, ingested, edges

**Performance notes:**
- ~100 messages/batch with GPU embedding ≈ 2–5 seconds per batch
- 10,000 messages ≈ 5–10 minutes on decent hardware
- The script is idempotent — safe to re-run

---

## Step 6: Run the Bot

```bash
# Run directly
python main.py

# Or with Docker
docker compose up -d

# Check logs
docker compose logs -f chatgpt-discord-bot
```

---

## How It Works

### Trigger System (Grok-style)

The bot responds when:

| Trigger | How |
|---|---|
| **@mention** | User mentions the bot in a message |
| **Reply to bot** | User replies to a bot message |
| **Active thread** | Bot has previously replied in this thread |
| **replyAll mode** | `/replyall` — bot responds to everything in the channel |
| **Implicit reply** *(optional)* | Message has >0.75 cosine similarity to a recent bot message |

### Response Generation Flow

```
User message
    │
    ├─► Track in short-term context (per-channel, 30-msg window)
    │
    ├─► Check trigger conditions
    │
    ├─► Build RAG query (user msg + recent context)
    │   └─► LangChain PGVector similarity search → top 15 results
    │
    ├─► Assemble prompt:
    │   ├── System prompt (CLCT personality)
    │   ├── Retrieved historical messages (with citations)
    │   ├── Recent conversation history (channel context)
    │   └── Current user message
    │
    ├─► Stream to Ollama (llama3.1:8b, temp=0.8)
    │
    └─► Stream tokens → Discord message edits (every 1.5s)
```

### Context Management

- **Short-term**: Rolling deque of last 30 messages per channel/thread, stored in memory
- **Long-term**: RAG retrieval from PostgreSQL + pgvector
- **Interaction tracking**: Reply chains and mentions stored as edges in DB
- **Auto-cleanup**: Expired contexts (>1 hour idle) are garbage-collected

---

## Slash Commands

| Command | Description |
|---|---|
| `/chat [message]` | Chat with CLCT directly |
| `/reset` | Clear this channel's conversation context |
| `/resetall` | Clear ALL contexts (admin) |
| `/status` | Show Ollama status, model, RAG stats |
| `/draw [prompt]` | Generate an image (requires legacy provider) |
| `/switchpersona [name]` | Change AI personality |
| `/private` | Toggle private/public responses |
| `/replyall` | Toggle reply-all mode |
| `/provider` | Switch AI provider (legacy) |
| `/help` | Show all commands |

---

## Testing

```bash
# Run tests
pytest

# Run with verbose output
pytest -v

# Test specific module
pytest tests/test_aclient.py -v
```

### Manual Testing Checklist

1. **@mention test**: Send `@CLCT what is 2+2?` — should respond
2. **Reply test**: Reply to a bot message — should continue conversation
3. **Thread test**: Start a thread, mention bot, then send follow-ups — bot should respond
4. **Context test**: Have a multi-turn conversation, bot should remember recent messages
5. **RAG test**: Ask about topics from ingested history — should cite sources `[1]`, `[2]`
6. **Streaming test**: Response should appear incrementally (not all at once)
7. **Reset test**: `/reset` then ask about previous conversation — should not remember
8. **Status test**: `/status` should show Ollama online, model name, RAG message count

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `Ollama is not reachable` | Ensure `ollama serve` is running, check `OLLAMA_BASE_URL` |
| `Model not found` | Run `ollama pull llama3.1:8b` |
| `pgvector extension not found` | Run `CREATE EXTENSION vector;` in the database |
| `Connection refused (PostgreSQL)` | Check `POSTGRES_URL`, ensure PostgreSQL is running |
| `Bot not responding to mentions` | Ensure `message_content` intent is enabled in Discord Developer Portal |
| `Streaming looks choppy` | Increase `STREAM_EDIT_INTERVAL` to 2.0 or 2.5 |
| `Out of VRAM` | Use a smaller model (`llama3.2:3b`) or reduce `OLLAMA_NUM_CTX` |

---

## Project Structure

```
chatGPT-discord-bot/
├── main.py                     # Entry point
├── .env.example                # Environment config template
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker setup (bot + PostgreSQL)
├── Dockerfile
├── system_prompt.txt           # CLCT personality prompt
├── pytest.ini
│
├── src/
│   ├── __init__.py
│   ├── aclient.py              # CLCTClient — core bot logic
│   ├── bot.py                  # Event handlers & slash commands
│   ├── ollama_provider.py      # Ollama async chat + streaming
│   ├── providers.py            # Legacy AI providers (OpenAI, etc.)
│   ├── personas.py             # AI personality profiles
│   ├── log.py                  # Logging setup
│   └── art.py
│
├── rag/
│   ├── __init__.py
│   ├── db.py                   # LangChain PGVector store + asyncpg edges
│   ├── embedder.py             # LangChain OllamaEmbeddings wrapper
│   ├── ingest.py               # Historical conversation ingestion CLI
│   └── retriever.py            # LangChain PGVector search + citations
│
├── utils/
│   ├── __init__.py
│   ├── context_manager.py      # Per-channel context windows + prompt building
│   └── message_utils.py        # Discord message splitting
│
└── tests/
    ├── __init__.py
    ├── test_aclient.py
    ├── test_personas.py
    └── test_providers.py
```
