"""
Provider-neutral chat shim used by the Discord client and HTTP API.

All chat traffic funnels through here, so this is the single choke point where
the CJK output filter (audit: language-purity safety net) is applied and where
bare Python tool callables are converted to JSON tool schemas before being sent
to the provider.

Configure the backend via LLM_PROVIDER (openai|gemini|openai_compatible|ollama).
"""

import os
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv

from src.llm.factory import get_chat_client
from utils.text_filters import strip_chinese

load_dotenv()

logger = logging.getLogger(__name__)

# Max model<->tool round trips before giving up. Accept either env name
# (MAX_TOOL_ROUNDS is what .env.sample/.env ship with; TOOL_MAX_ROUNDS kept for
# backwards compatibility).
TOOL_MAX_ROUNDS: int = int(
    os.getenv("MAX_TOOL_ROUNDS", os.getenv("TOOL_MAX_ROUNDS", "3"))
)

OLLAMA_BASE_URL: str = os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
OLLAMA_MODEL: str = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "vllm-local"))
OLLAMA_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT_SECONDS", os.getenv("OLLAMA_TIMEOUT", "120")))

_last_token_usage: Dict[str, Any] = {}

# Optional hard cap on generated tokens. Unset → provider default.
_raw_max_tokens = os.getenv("LLM_MAX_TOKENS")
LLM_MAX_TOKENS: Optional[int] = int(_raw_max_tokens) if _raw_max_tokens else None

# Tool outputs executed during the most recent chat_with_tools call, so callers
# can verify a tool-grounded answer against the real tool data.
_last_tool_results: List[Dict[str, Any]] = []


def get_last_token_usage() -> Dict[str, Any]:
    return _last_token_usage.copy()


def get_last_tool_results() -> List[Dict[str, Any]]:
    return list(_last_tool_results)


async def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.8,
    images: Optional[List[str]] = None,
    request_context: Optional[Any] = None,
) -> str:
    global _last_token_usage
    client = get_chat_client()
    response = await client.chat(
        messages=messages, temperature=temperature, max_tokens=LLM_MAX_TOKENS
    )
    _last_token_usage = response.usage.__dict__.copy()
    return strip_chinese(response.content)


def _extract_message(raw: Optional[dict]) -> dict:
    """Pull the assistant message out of a raw chat response.

    Handles both the OpenAI shape (``choices[0].message``) and the Ollama
    native shape (top-level ``message``).
    """
    raw = raw or {}
    choices = raw.get("choices")
    if choices:
        return (choices[0] or {}).get("message") or {}
    return raw.get("message") or {}


async def chat_with_tools(
    messages: List[Dict[str, str]],
    tools: Optional[list] = None,
    tool_functions: Optional[dict] = None,
    model: Optional[str] = None,
    temperature: float = 0.8,
    request_context: Optional[Any] = None,
) -> str:
    """Run a tool-calling conversation and return the final text.

    Tools are bare Python callables; they are converted to JSON tool schemas
    before being sent (passing raw functions is not JSON-serializable for any
    provider). Tool calls returned by the model are executed via
    ``tool_functions`` and fed back until the model produces a final answer.

    Fails safe: any error returns an empty string so the caller can degrade
    gracefully (fall back to a plain streamed answer) instead of surfacing a
    serialization crash to the user.
    """
    global _last_token_usage, _last_tool_results
    _last_tool_results = []
    client = get_chat_client()

    if not tools:
        response = await client.chat(
            messages=messages, temperature=temperature, max_tokens=LLM_MAX_TOKENS
        )
        _last_token_usage = response.usage.__dict__.copy()
        return strip_chinese(response.content)

    try:
        from src.llm.tool_schema import functions_to_openai_tools

        schemas = functions_to_openai_tools(tools)
        working: List[Dict[str, Any]] = list(messages)
        last_content = ""

        for _ in range(TOOL_MAX_ROUNDS):
            response = await client.chat(
                messages=working, temperature=temperature, tools=schemas,
                max_tokens=LLM_MAX_TOKENS,
            )
            _last_token_usage = response.usage.__dict__.copy()
            last_content = response.content or last_content

            message = _extract_message(response.raw)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return strip_chinese(response.content)

            # Echo the assistant's tool-call turn, then append each tool result.
            working.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                spec = call.get("function") or {}
                name = spec.get("name", "")
                raw_args = spec.get("arguments")
                try:
                    args = (
                        raw_args
                        if isinstance(raw_args, dict)
                        else json.loads(raw_args or "{}")
                    )
                except (json.JSONDecodeError, TypeError):
                    args = {}
                func = (tool_functions or {}).get(name)
                try:
                    result = (
                        func(**args)
                        if func
                        else json.dumps({"error": f"unknown tool: {name}"})
                    )
                except Exception as tool_err:
                    logger.warning(f"Tool '{name}' execution failed: {tool_err}")
                    result = json.dumps({"error": str(tool_err)})
                _last_tool_results.append({"name": name, "output": str(result)})
                working.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "name": name,
                        "content": str(result),
                    }
                )

        # Rounds exhausted while the model kept calling tools: make one final
        # call WITHOUT tools to force a natural-language answer grounded in the
        # tool results we already gathered.
        try:
            final = await client.chat(
                messages=working, temperature=temperature, max_tokens=LLM_MAX_TOKENS
            )
            _last_token_usage = final.usage.__dict__.copy()
            if final.content:
                return strip_chinese(final.content)
        except Exception as final_err:
            logger.warning(f"Final tool-free summary call failed: {final_err}")

        return strip_chinese(last_content)
    except Exception as exc:
        logger.warning(
            f"Tool-calling failed ({exc}); returning empty for graceful fallback"
        )
        return ""


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
