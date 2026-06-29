"""Regression: analytical query → tool calling no longer crashes.

Reproduces "❌ Sorry, I ran into an issue: Object of type function is not JSON
serializable" by driving the real Discord handler down the analytical/tool path
with the real chat_with_tools, mocking only the LLM client.
"""

import pytest

import src.aclient as aclient
import src.ollama_provider as provider
from rag.metrics import RAGMetric
from rag.result import RAGBuildResult, RAGDecision
from src.llm.types import LLMResponse, TokenUsage


def _resp(content, tool_calls=None):
    return LLMResponse(
        content=content,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        raw={"choices": [{"message": {"content": content, "tool_calls": tool_calls or []}}]},
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def chat(self, messages, temperature=0.8, **kwargs):
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def _answer_rag_result():
    return RAGBuildResult(
        context="ctx",
        domain_prompt="",
        metric=RAGMetric(query_language="vi", query_intent="analytical", rag_decision="answer"),
        decision=RAGDecision.ANSWER,
        decision_reason="ok",
        evidence_score=0.9,
        provenance=[],
        retrieved_results=[],
        primary_domain="pz",
    )


def search_items(category: str = None, sort_by: str = None, limit: int = 10):
    """Search items."""
    return '{"items":[{"name":"Battle Axe","max_damage":35}]}'


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_analytical_query_runs_tools_without_json_crash(mock_discord_message, monkeypatch):
    async def fake_build_prompt(**kwargs):
        return (
            [{"role": "user", "content": kwargs["user_message"]}],
            0.1,
            "analytical",
            _answer_rag_result(),
        )

    aclient.discordClient.context_manager.build_prompt = fake_build_prompt
    monkeypatch.setattr(aclient, "ENABLE_TOOL_CALLING", True)
    monkeypatch.setattr(aclient, "STRUCTURED_TOOLS_AVAILABLE", True)
    monkeypatch.setattr(aclient, "PZ_TOOLS", [search_items])
    monkeypatch.setattr(aclient, "PZ_TOOL_FUNCTIONS", {"search_items": search_items})

    fake = FakeClient([
        _resp("", tool_calls=[{
            "id": "c1",
            "function": {"name": "search_items", "arguments": '{"sort_by": "max_damage", "sub_category": "Axes"}'},
        }]),
        _resp("Rìu sát thương cao nhất là **Battle Axe** (35)."),
    ])
    monkeypatch.setattr(provider, "get_chat_client", lambda: fake)

    await aclient.discordClient._generate_and_send(
        mock_discord_message, "rìu nào có sát thương cao nhất nomnom?"
    )

    contents = [m.content for m in mock_discord_message.channel.sent]
    assert contents, "no message sent"
    joined = "\n".join(contents)
    assert "Sorry, I ran into an issue" not in joined   # no crash
    assert "JSON serializable" not in joined
    assert "Battle Axe" in joined                        # real tool-derived answer
