# Ollama Cloud Authentication Design

## Problem

The repository can point Ollama clients directly at `https://ollama.com`, but
the configured `OLLAMA_API_KEY` is not propagated to the HTTP Authorization
header. Model discovery may still return HTTP 200 while authenticated inference
fails with HTTP 403.

The active `main.py` entrypoint uses `src.llm.factory`, while the older
`discord_bot.py` entrypoint uses `llm_provider.py`. Both paths must be covered.

## Design

Add a native asynchronous Ollama client under `src/llm/` that calls
`POST {base_url}/api/chat` and sends `Authorization: Bearer <key>` when a key is
configured. Extend the provider factory with `LLM_PROVIDER=ollama` and accept
`OLLAMA_MODEL`, `OLLAMA_BASE_URL`, and `OLLAMA_API_KEY` as compatibility
fallbacks.

Update the legacy LlamaIndex `Ollama` construction to supply the same
Authorization header through its client configuration. Never log or include
the API key in error messages.

## Error Handling

Configuration fails early when the Ollama cloud host is selected without an
API key. Non-success HTTP responses preserve the status code and safe response
message so operational errors remain diagnosable.

## Testing

Regression tests verify that:

- Ollama environment variables produce an authenticated provider config.
- The native client sends the Bearer header and parses chat/token usage.
- Cloud configuration without a key is rejected.
- The legacy LlamaIndex constructor receives authenticated client headers.

