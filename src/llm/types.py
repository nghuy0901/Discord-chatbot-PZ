from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


ChatMessage = Dict[str, Any]


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: Optional[int] = None
    estimated_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.total_tokens is None:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: Optional[Dict[str, Any]] = None


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    embedding_model: Optional[str] = None
    timeout_seconds: int = 120
