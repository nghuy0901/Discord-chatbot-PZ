from aiohttp import web
import pytest

from src.llm.ollama_client import OllamaClient
from src.llm.types import LLMConfig


@pytest.mark.asyncio
async def test_chat_sends_bearer_auth_and_parses_usage(unused_tcp_port):
    captured = {}

    async def chat_handler(request):
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = await request.json()
        return web.json_response(
            {
                "message": {"role": "assistant", "content": "authenticated"},
                "prompt_eval_count": 5,
                "eval_count": 7,
            }
        )

    app = web.Application()
    app.router.add_post("/api/chat", chat_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()

    try:
        client = OllamaClient(
            LLMConfig(
                provider="ollama",
                model="qwen3-coder-next:cloud",
                api_key="ollama-test-key",
                base_url=f"http://127.0.0.1:{unused_tcp_port}",
            )
        )

        response = await client.chat(
            [{"role": "user", "content": "hello"}],
            temperature=0.2,
        )
    finally:
        await runner.cleanup()

    assert captured["authorization"] == "Bearer ollama-test-key"
    assert captured["body"]["model"] == "qwen3-coder-next:cloud"
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["temperature"] == 0.2
    assert response.content == "authenticated"
    assert response.usage.prompt_tokens == 5
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 12


@pytest.mark.asyncio
async def test_chat_error_does_not_expose_api_key(unused_tcp_port):
    async def chat_handler(request):
        return web.Response(status=403, text="subscription required")

    app = web.Application()
    app.router.add_post("/api/chat", chat_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()

    try:
        client = OllamaClient(
            LLMConfig(
                provider="ollama",
                model="qwen3-coder-next:cloud",
                api_key="secret-ollama-key",
                base_url=f"http://127.0.0.1:{unused_tcp_port}",
            )
        )
        with pytest.raises(RuntimeError) as exc_info:
            await client.chat([{"role": "user", "content": "hello"}])
    finally:
        await runner.cleanup()

    assert "status 403" in str(exc_info.value)
    assert "secret-ollama-key" not in str(exc_info.value)
