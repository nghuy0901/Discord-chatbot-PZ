from src.llm.types import LLMConfig, LLMResponse, TokenUsage


def test_token_usage_total_defaults_to_sum():
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
    assert usage.total_tokens == 15


def test_llm_config_requires_provider():
    config = LLMConfig(provider="openai", model="gpt-4o-mini")
    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"


def test_llm_response_defaults_to_empty_usage():
    response = LLMResponse(content="hello")
    assert response.usage.total_tokens == 0
