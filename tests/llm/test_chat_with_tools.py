"""Tests for the provider-neutral tool-calling loop in src.ollama_provider.

Covers the full round-trip: model requests a tool -> tool executes -> result is
fed back -> model returns a final grounded answer. Also covers the CJK strip and
the fail-safe empty-string fallback.
"""

import pytest

import src.ollama_provider as op
from src.llm.types import LLMResponse, TokenUsage


class _FakeClient:
    """Returns queued LLMResponses; records the messages/tools it was called with."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def chat(self, messages, temperature=0.7, tools=None, max_tokens=None, **kw):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self._responses.pop(0)


def _assistant_toolcall(name, arguments):
    return LLMResponse(
        content="",
        usage=TokenUsage(),
        raw={
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
            }
        },
    )


def _assistant_final(text):
    return LLMResponse(
        content=text,
        usage=TokenUsage(),
        raw={"message": {"role": "assistant", "content": text}},
    )


# A stand-in tool with the same name the model will "call".
def search_items(category=None, sort_by=None, limit=10):
    """Search items.

    Args:
        category: category filter
        sort_by: column to sort by
        limit: max rows
    """
    return '{"count": 1, "items": [{"name": "Axe", "max_damage": 2.5}]}'


@pytest.mark.asyncio
async def test_tool_call_executes_and_returns_final_answer(monkeypatch):
    fake = _FakeClient([
        _assistant_toolcall("search_items", {"category": "Weapons", "sort_by": "max_damage", "limit": 1}),
        _assistant_final("Vũ khí sát thương cao nhất là **Axe** (2.5)."),
    ])
    monkeypatch.setattr(op, "get_chat_client", lambda: fake)

    captured = {}

    def tool(**kwargs):
        captured.update(kwargs)
        return search_items(**kwargs)

    out = await op.chat_with_tools(
        messages=[{"role": "user", "content": "vũ khí nào sát thương cao nhất?"}],
        tools=[search_items],
        tool_functions={"search_items": tool},
    )

    assert out == "Vũ khí sát thương cao nhất là **Axe** (2.5)."
    # The model's chosen arguments were actually passed to the tool.
    assert captured["sort_by"] == "max_damage"
    # Tool output was captured for verification.
    results = op.get_last_tool_results()
    assert results and results[0]["name"] == "search_items"
    # Second model call received the tool result in its message list.
    assert any(m.get("role") == "tool" for m in fake.calls[1]["messages"])
    # Tools were sent as JSON schemas, not raw callables.
    assert isinstance(fake.calls[0]["tools"], list)
    assert fake.calls[0]["tools"][0]["type"] == "function"


@pytest.mark.asyncio
async def test_final_answer_has_chinese_stripped(monkeypatch):
    fake = _FakeClient([
        _assistant_toolcall("search_items", {"sort_by": "max_damage"}),
        _assistant_final("Vũ khí mạnh nhất 全部 là Axe."),
    ])
    monkeypatch.setattr(op, "get_chat_client", lambda: fake)

    out = await op.chat_with_tools(
        messages=[{"role": "user", "content": "q"}],
        tools=[search_items],
        tool_functions={"search_items": lambda **k: "{}"},
    )
    assert "全" not in out and "部" not in out
    assert "Axe" in out


@pytest.mark.asyncio
async def test_no_tools_path_returns_stripped_content(monkeypatch):
    fake = _FakeClient([_assistant_final("Chào cậu 你好!")])
    monkeypatch.setattr(op, "get_chat_client", lambda: fake)

    out = await op.chat_with_tools(messages=[{"role": "user", "content": "hi"}], tools=None)
    assert "你" not in out and "好" not in out
    assert "Chào cậu" in out


@pytest.mark.asyncio
async def test_failsafe_returns_empty_string_on_error(monkeypatch):
    class _Boom:
        async def chat(self, **kw):
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(op, "get_chat_client", lambda: _Boom())

    out = await op.chat_with_tools(
        messages=[{"role": "user", "content": "q"}],
        tools=[search_items],
        tool_functions={"search_items": lambda **k: "{}"},
    )
    # Empty string lets the caller fall back to a plain streamed answer.
    assert out == ""
