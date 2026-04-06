"""
Ollama LLM provider — async chat completion with streaming support,
timeout handling, and multimodal capabilities.
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

import ollama
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # seconds
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))  # context window
MAX_TOOL_ROUNDS: int = int(os.getenv("MAX_TOOL_ROUNDS", "3"))   # max tool call rounds

# ---------------------------------------------------------------------------
# Async client (singleton)
# ---------------------------------------------------------------------------
_client: Optional[ollama.AsyncClient] = None


def _get_client() -> ollama.AsyncClient:
    global _client
    if _client is None:
        _client = ollama.AsyncClient(host=OLLAMA_BASE_URL)
    return _client


# ---------------------------------------------------------------------------
# Chat completion (non-streaming)
# ---------------------------------------------------------------------------
async def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.8,
    images: Optional[List[str]] = None,
) -> str:
    """
    Generate a full chat response from Ollama.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model: Override default model.
        temperature: Sampling temperature.
        images: Optional base64 images for multimodal models.

    Returns:
        The assistant's response text.
    """
    model = model or OLLAMA_MODEL
    client = _get_client()

    # Attach images to the last user message if provided
    if images and messages:
        messages = [m.copy() for m in messages]
        messages[-1]["images"] = images

    try:
        response = await asyncio.wait_for(
            client.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_ctx": OLLAMA_NUM_CTX,
                },
            ),
            timeout=OLLAMA_TIMEOUT,
        )
        return response["message"]["content"]

    except asyncio.TimeoutError:
        logger.error(f"Ollama chat timed out after {OLLAMA_TIMEOUT}s")
        raise TimeoutError(
            f"Response generation timed out after {OLLAMA_TIMEOUT} seconds. "
            "Try a shorter message or a faster model."
        )
    except ollama.ResponseError as e:
        logger.error(f"Ollama API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Ollama chat failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Chat completion with tool calling
# ---------------------------------------------------------------------------
async def chat_with_tools(
    messages: List[Dict[str, str]],
    tools: list = None,
    tool_functions: dict = None,
    model: Optional[str] = None,
    temperature: float = 0.8,
) -> str:
    """
    Generate a chat response with tool calling support.

    If the model decides to call tools, this function:
    1. Detects tool_calls in the response
    2. Executes the tools using tool_functions registry
    3. Feeds results back to the model
    4. Returns the final text response

    Args:
        messages: Chat messages.
        tools: List of tool functions (with Google-style docstrings).
        tool_functions: Dict mapping function name → callable.
        model: Override default model.
        temperature: Sampling temperature.

    Returns:
        The assistant's final response text.
    """
    if not tools or not tool_functions:
        # No tools available — fall back to regular chat
        return await chat_completion(messages, model, temperature)

    model = model or OLLAMA_MODEL
    client = _get_client()

    # Copy messages to avoid mutating the original
    working_messages = [m.copy() for m in messages]

    for round_num in range(MAX_TOOL_ROUNDS):
        try:
            response = await asyncio.wait_for(
                client.chat(
                    model=model,
                    messages=working_messages,
                    tools=tools,
                    options={
                        "temperature": temperature,
                        "num_ctx": OLLAMA_NUM_CTX,
                    },
                ),
                timeout=OLLAMA_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(f"Ollama tool call timed out (round {round_num + 1})")
            raise TimeoutError(f"Tool call timed out after {OLLAMA_TIMEOUT}s")
        except Exception as e:
            logger.error(f"Ollama tool call failed: {e}")
            # Fallback to regular chat without tools
            logger.info("Falling back to regular chat without tools")
            return await chat_completion(messages, model, temperature)

        # Check if model wants to call tools
        msg = response.get("message", {})
        tool_calls = msg.get("tool_calls")

        if not tool_calls:
            # No tool calls — model responded directly
            return msg.get("content", "")

        # Execute tool calls
        logger.info(
            f"🔧 Tool call round {round_num + 1}: "
            f"{len(tool_calls)} call(s)"
        )

        # Add the assistant's message (with tool_calls) to conversation
        working_messages.append(msg)

        for tool_call in tool_calls:
            func_name = tool_call.get("function", {}).get("name", "")
            func_args = tool_call.get("function", {}).get("arguments", {})

            logger.info(f"  🔧 Calling: {func_name}({func_args})")

            func = tool_functions.get(func_name)
            if func:
                try:
                    result = func(**func_args)
                    logger.info(
                        f"  ✅ {func_name} returned "
                        f"{len(result)} chars"
                    )
                except Exception as e:
                    result = json.dumps({"error": f"Tool execution failed: {e}"})
                    logger.error(f"  ❌ {func_name} error: {e}")
            else:
                result = json.dumps({"error": f"Unknown tool: {func_name}"})
                logger.warning(f"  ⚠️ Unknown tool: {func_name}")

            # Add tool result to messages
            working_messages.append({
                "role": "tool",
                "content": str(result),
            })

    # If we exhausted all rounds, return whatever we have
    logger.warning(f"Exhausted {MAX_TOOL_ROUNDS} tool rounds, returning last content")
    return msg.get("content", "I completed the tool calls but couldn't formulate a final answer.")
# ---------------------------------------------------------------------------
async def chat_completion_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.8,
    images: Optional[List[str]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream chat response tokens from Ollama.

    Yields individual text chunks as they arrive.
    """
    model = model or OLLAMA_MODEL
    client = _get_client()

    if images and messages:
        messages = [m.copy() for m in messages]
        messages[-1]["images"] = images

    try:
        stream = await client.chat(
            model=model,
            messages=messages,
            stream=True,
            options={
                "temperature": temperature,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        )

        async for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

    except ollama.ResponseError as e:
        logger.error(f"Ollama stream error: {e}")
        yield f"\n❌ Ollama error: {e}"
    except Exception as e:
        logger.error(f"Ollama stream failed: {e}")
        yield f"\n❌ Generation failed: {e}"


# ---------------------------------------------------------------------------
# Collect streamed response into Discord-friendly chunks
# ---------------------------------------------------------------------------
async def stream_to_discord_chunks(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.8,
    chunk_interval: float = 1.0,
    images: Optional[List[str]] = None,
) -> AsyncGenerator[str, None]:
    """
    Collect streamed tokens into larger chunks suitable for Discord edits.

    Yields accumulated text at regular intervals, and a final complete text.
    This is designed to work with Discord message editing for a streaming UX:
      - Send an initial message
      - Edit it periodically with accumulated content
      - Final edit with complete response

    Args:
        messages: Chat messages for Ollama.
        model: Model override.
        temperature: Sampling temperature.
        chunk_interval: Seconds between yielded chunks.
        images: Optional images for multimodal.

    Yields:
        Accumulated response text at each interval.
    """
    buffer = ""
    last_yield_time = asyncio.get_event_loop().time()

    async for token in chat_completion_stream(
        messages, model, temperature, images
    ):
        buffer += token
        now = asyncio.get_event_loop().time()

        if now - last_yield_time >= chunk_interval:
            yield buffer
            last_yield_time = now

    # Always yield the final complete text
    if buffer:
        yield buffer


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------
async def list_models() -> List[Dict[str, Any]]:
    """List all models available in the local Ollama instance."""
    client = _get_client()
    try:
        response = await client.list()
        return response.get("models", [])
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        return []


async def ensure_model(model: Optional[str] = None) -> bool:
    """Pull the model if it is not already available. Returns True if ready."""
    model = model or OLLAMA_MODEL
    client = _get_client()
    try:
        models = await list_models()
        names = [m.get("name", m.get("model", "")) for m in models]
        if any(model in n for n in names):
            return True

        logger.info(f"Pulling model '{model}' — this may take a while …")
        await client.pull(model)
        logger.info(f"Model '{model}' ready.")
        return True
    except Exception as e:
        logger.error(f"Failed to ensure model '{model}': {e}")
        return False


async def health_check() -> bool:
    """Check if Ollama is reachable."""
    client = _get_client()
    try:
        await client.list()
        return True
    except Exception:
        return False
