"""
Compatibility shim for legacy imports.

Local models are accessed through provider-neutral clients. For local runtime,
serve a model behind an OpenAI-compatible endpoint such as vLLM and configure
LLM_PROVIDER=openai_compatible.
"""

import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv

from src.llm.factory import get_chat_client

load_dotenv()

OLLAMA_BASE_URL: str = os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
OLLAMA_MODEL: str = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "vllm-local"))
OLLAMA_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT_SECONDS", os.getenv("OLLAMA_TIMEOUT", "120")))

_last_token_usage: Dict[str, Any] = {}


def get_last_token_usage() -> Dict[str, Any]:
    return _last_token_usage.copy()


async def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.8,
    images: Optional[List[str]] = None,
    request_context: Optional[Any] = None,
) -> str:
    global _last_token_usage
    client = get_chat_client()
    response = await client.chat(messages=messages, temperature=temperature)
    _last_token_usage = response.usage.__dict__.copy()
    return response.content


async def chat_with_tools(
    messages: List[Dict[str, str]],
    tools: Optional[list] = None,
    tool_functions: Optional[dict] = None,
    model: Optional[str] = None,
    temperature: float = 0.8,
    request_context: Optional[Any] = None,
) -> str:
    global _last_token_usage
    client = get_chat_client()
    kwargs: Dict[str, Any] = {}
    if tools:
        kwargs["tools"] = tools
    response = await client.chat(messages=messages, temperature=temperature, **kwargs)
    _last_token_usage = response.usage.__dict__.copy()
    return response.content


async def chat_completion_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.8,
    images: Optional[List[str]] = None,
    request_context: Optional[Any] = None,
) -> AsyncGenerator[str, None]:
    yield await chat_completion(
        messages,
        model=model,
        temperature=temperature,
        images=images,
        request_context=request_context,
    )


async def stream_to_discord_chunks(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.8,
    chunk_interval: float = 1.0,
    images: Optional[List[str]] = None,
    request_context: Optional[Any] = None,
) -> AsyncGenerator[str, None]:
    yield await chat_completion(
        messages,
        model=model,
        temperature=temperature,
        images=images,
        request_context=request_context,
    )


async def list_models() -> List[Dict[str, Any]]:
    return [{"name": OLLAMA_MODEL, "model": OLLAMA_MODEL}]


async def ensure_model(model: Optional[str] = None) -> bool:
    return await health_check()


async def health_check() -> bool:
    try:
        get_chat_client()
        return True
    except Exception:
        return False
