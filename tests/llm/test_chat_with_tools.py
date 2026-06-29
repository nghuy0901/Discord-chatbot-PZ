"""chat_with_tools: schema conversion + tool execution loop + fail-safe."""

import asyncio

import src.ollama_provider as provider
from src.llm.types import LLMResponse, TokenUsage


def _resp(content, tool_calls=None):
    message = {"content": content, "tool_calls": tool_calls or []}
    return LLMResponse(
        content=content,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        raw={"choices": [{"message": message}]},
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def chat(self, messages, temperature=0.8, **kwargs):
        self.calls.append({"messages": list(messages), "kwargs": kwargs})
        return self.responses[len(self.calls) - 1]


def search_items(sort_by: str = None):
    """Search items."""
    return "{}"


def test_tool_loop_executes_and_returns_final(monkeypatch):
    captured = {}

    def tool(**kwargs):
        captured.update(kwargs)
        return '{"items":[{"name":"Battle Axe","max_damage":35}]}'

    fake = FakeClient([
        _resp("", tool_calls=[{
            "id": "c1",
            "function": {"name": "search_items", "arguments": '{"sort_by": "max_damage"}'},
        }]),
        _resp("Rìu mạnh nhất là Battle Axe."),
    ])
    monkeypatch.setattr(provider, "get_chat_client", lambda: fake)

    result = asyncio.run(provider.chat_with_tools(
        messages=[{"role": "user", "content": "rìu nào mạnh nhất?"}],
        tools=[search_items],
        tool_functions={"search_items": tool},
    ))

    assert result == "Rìu mạnh nhất là Battle Axe."
    assert captured == {"sort_by": "max_damage"}          # tool actually ran
    # Tools were sent as JSON schemas, not raw callables.
    sent_tools = fake.calls[0]["kwargs"]["tools"]
    assert sent_tools[0]["type"] == "function"
    # The tool result was fed back to the model.
    tool_msgs = [m for m in fake.calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and "Battle Axe" in tool_msgs[0]["content"]


def test_no_tool_calls_returns_content(monkeypatch):
    fake = FakeClient([_resp("Trả lời trực tiếp.")])
    monkeypatch.setattr(provider, "get_chat_client", lambda: fake)
    result = asyncio.run(provider.chat_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        tools=[search_items],
        tool_functions={"search_items": lambda **k: "{}"},
    ))
    assert result == "Trả lời trực tiếp."


def test_failure_returns_empty_string(monkeypatch):
    class BoomClient:
        async def chat(self, **kwargs):
            raise RuntimeError("Object of type function is not JSON serializable")

    monkeypatch.setattr(provider, "get_chat_client", lambda: BoomClient())
    result = asyncio.run(provider.chat_with_tools(
        messages=[{"role": "user", "content": "x"}],
        tools=[search_items],
        tool_functions={},
    ))
    assert result == ""  # graceful fallback, no crash
