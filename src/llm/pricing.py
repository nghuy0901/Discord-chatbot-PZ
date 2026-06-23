from typing import Tuple


PRICE_PER_1M_TOKENS = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "vllm-local": (0.0, 0.0),
    "ollama-openai-compatible": (0.0, 0.0),
}


def get_price_per_1m_tokens(model: str) -> Tuple[float, float]:
    return PRICE_PER_1M_TOKENS.get(model, (0.0, 0.0))


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_price, output_price = get_price_per_1m_tokens(model)
    input_cost = (prompt_tokens / 1_000_000) * input_price
    output_cost = (completion_tokens / 1_000_000) * output_price
    return input_cost + output_cost
