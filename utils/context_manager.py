"""
Context manager — maintains short-term conversation windows per channel/thread,
detects implicit contextual replies via embedding similarity, and builds
the full prompt for Ollama including system prompt, RAG results, and history.

Enhanced with:
- C1: Advanced RAG-aware system prompt
- B1: Smart response formatting rules in prompt
- D2: Knowledge base domain-specific prompts
- A3: Metrics integration
"""

import os
import time
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Deque, Tuple

from rag.retriever import build_rag_context
from rag.embedder import embed_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONTEXT_WINDOW_SIZE: int = int(os.getenv("CONTEXT_WINDOW_SIZE", "30"))
CONTEXT_TTL_SECONDS: int = int(os.getenv("CONTEXT_TTL_SECONDS", "3600"))  # 1 hour
IMPLICIT_SIMILARITY_THRESHOLD: float = float(
    os.getenv("IMPLICIT_SIMILARITY_THRESHOLD", "0.75")
)
PERSONALITY_TEMPERATURE: float = float(os.getenv("PERSONALITY_TEMPERATURE", "0.8"))


# ---------------------------------------------------------------------------
# System prompt template (C1 - Enhanced RAG prompt + B1 - Formatting rules)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """You are CLCT, inspired by Jarvis and the Hitchhiker's Guide to the Galaxy.

You are curious, witty, and slightly rebellious. You answer questions as accurately and honestly as possible, even if the truth is inconvenient. You are not overly politically correct and will tackle controversial topics with facts and logic.

You have a sense of humor and use it when appropriate. You are never flirtatious or overly playful. You do not deceive or mislead.

You are chatting in a Discord server called "Chấn thương tâm lý" — a Vietnamese gaming and social community. Members speak in casual Vietnamese with lots of slang, abbreviations, and emojis.

Current date: {current_date}

# 📋 RAG Instructions (IMPORTANT — follow strictly)
When "Retrieved Knowledge Base" or "Retrieved Chat History" data is provided below:
1. **PRIORITIZE retrieved data** over your general knowledge. Base your answers on the retrieved information first.
2. **Cite sources** using [KB-1], [KB-2] for knowledge base or [1], [2] for chat history when referencing retrieved data.
3. If no relevant retrieved data is available, you MAY use your general knowledge but clearly state: "Theo kiến thức chung của tôi..." or "Based on my general knowledge..."
4. **NEVER fabricate** information or pretend retrieved data says something it doesn't.
5. If the retrieved data is insufficient or contradictory, acknowledge it honestly.
6. **Cross-reference** multiple retrieved sources when possible for accuracy.

# 🎨 Response Formatting Rules (B1)
Format your responses for optimal Discord readability:
1. **Structure**: Use bullet points (•) for lists, numbered lists for steps/rankings.
2. **Emphasis**: Use **bold** for key terms/names, *italic* for emphasis or game terms.
3. **Code**: Use `inline code` for commands, stats, or technical values. Use ```code blocks``` for multi-line code/configs.
4. **Headings**: Use ## or ### for section headers in longer responses. Never use # (too large for Discord).
5. **Length**: Keep responses concise and focused. Aim for 100-300 words unless the topic requires more detail.
6. **Tables**: Use markdown tables for comparisons (e.g., item stats, build comparisons).
7. **Spoilers**: Use ||spoiler tags|| for potential spoilers in game content.
8. **Language**: If the user writes in Vietnamese, respond in Vietnamese. If English, respond in English. Match their language naturally.
9. **Tone**: Be helpful and friendly but not robotic. Use casual tone matching the server culture.
10. Do NOT generate images. You only respond with text.

{domain_prompt}

{rag_context}"""


# ---------------------------------------------------------------------------
# Message data class
# ---------------------------------------------------------------------------
class ContextMessage:
    """A single message in the conversation context window."""

    __slots__ = (
        "message_id", "author_id", "author_name", "content",
        "timestamp", "is_bot", "reply_to_id", "channel_id",
    )

    def __init__(
        self,
        message_id: str,
        author_id: str,
        author_name: str,
        content: str,
        timestamp: float,
        is_bot: bool = False,
        reply_to_id: Optional[str] = None,
        channel_id: Optional[str] = None,
    ):
        self.message_id = message_id
        self.author_id = author_id
        self.author_name = author_name
        self.content = content
        self.timestamp = timestamp
        self.is_bot = is_bot
        self.reply_to_id = reply_to_id
        self.channel_id = channel_id

    def to_chat_dict(self) -> Dict[str, str]:
        """Convert to Ollama chat message format."""
        role = "assistant" if self.is_bot else "user"
        prefix = f"@{self.author_name}: " if not self.is_bot else ""
        return {"role": role, "content": f"{prefix}{self.content}"}

    def __repr__(self) -> str:
        return f"<ContextMessage {self.author_name}: {self.content[:40]}…>"


# ---------------------------------------------------------------------------
# Channel context store
# ---------------------------------------------------------------------------
class ChannelContext:
    """Rolling window of recent messages for a single channel or thread."""

    def __init__(self, max_size: int = CONTEXT_WINDOW_SIZE):
        self.messages: Deque[ContextMessage] = deque(maxlen=max_size)
        self.last_activity: float = time.time()
        self.bot_participated: bool = False

    def add(self, msg: ContextMessage) -> None:
        self.messages.append(msg)
        self.last_activity = time.time()
        if msg.is_bot:
            self.bot_participated = True

    def get_recent(self, n: Optional[int] = None) -> List[ContextMessage]:
        """Return the last n messages (or all if n is None)."""
        msgs = list(self.messages)
        if n is not None:
            msgs = msgs[-n:]
        return msgs

    def clear(self) -> None:
        self.messages.clear()
        self.bot_participated = False

    def is_expired(self) -> bool:
        return (time.time() - self.last_activity) > CONTEXT_TTL_SECONDS


# ---------------------------------------------------------------------------
# Global context manager
# ---------------------------------------------------------------------------
class ContextManager:
    """
    Manages short-term conversation contexts across all channels/threads.
    Provides prompt building with RAG integration.
    """

    def __init__(self):
        # key = channel_id or thread_id
        self._contexts: Dict[str, ChannelContext] = defaultdict(ChannelContext)
        self._cleanup_counter: int = 0
        # Track last query_id per channel for feedback mapping (B2)
        self._last_query_ids: Dict[str, str] = {}
        # Track last bot message_id per channel for feedback (B2)
        self._last_bot_msg_ids: Dict[str, str] = {}

    # ----- message tracking -----

    def track_message(
        self,
        channel_id: str,
        message_id: str,
        author_id: str,
        author_name: str,
        content: str,
        is_bot: bool = False,
        reply_to_id: Optional[str] = None,
    ) -> None:
        """Add a message to the channel's context window."""
        msg = ContextMessage(
            message_id=message_id,
            author_id=author_id,
            author_name=author_name,
            content=content,
            timestamp=time.time(),
            is_bot=is_bot,
            reply_to_id=reply_to_id,
            channel_id=channel_id,
        )
        self._contexts[channel_id].add(msg)
        self._maybe_cleanup()

    def get_context(self, channel_id: str) -> ChannelContext:
        return self._contexts[channel_id]

    def clear_context(self, channel_id: str) -> None:
        if channel_id in self._contexts:
            self._contexts[channel_id].clear()
            logger.info(f"Cleared context for channel {channel_id}")

    def clear_all(self) -> None:
        self._contexts.clear()
        logger.info("Cleared all channel contexts.")

    def has_bot_participated(self, channel_id: str) -> bool:
        return self._contexts[channel_id].bot_participated

    # ----- B2: Feedback tracking -----

    def set_last_query_id(self, channel_id: str, query_id: str) -> None:
        """Store the last RAG query ID for feedback mapping."""
        self._last_query_ids[channel_id] = query_id

    def get_last_query_id(self, channel_id: str) -> Optional[str]:
        return self._last_query_ids.get(channel_id)

    def set_last_bot_msg_id(self, channel_id: str, msg_id: str) -> None:
        """Store the last bot message ID for feedback mapping."""
        self._last_bot_msg_ids[channel_id] = msg_id

    def get_last_bot_msg_id(self, channel_id: str) -> Optional[str]:
        return self._last_bot_msg_ids.get(channel_id)

    # ----- implicit reply detection -----

    async def detect_implicit_reply(
        self,
        channel_id: str,
        new_message_content: str,
        threshold: float = IMPLICIT_SIMILARITY_THRESHOLD,
    ) -> Optional[ContextMessage]:
        """
        Check if a new message is an implicit reply to a recent bot message
        by comparing embedding similarity.
        """
        ctx = self._contexts.get(channel_id)
        if not ctx or not ctx.bot_participated:
            return None

        bot_messages = [m for m in ctx.get_recent(10) if m.is_bot]
        if not bot_messages:
            return None

        try:
            new_embedding = await embed_text(new_message_content)
        except Exception:
            return None

        best_match: Optional[ContextMessage] = None
        best_similarity: float = 0.0

        for bot_msg in reversed(bot_messages):
            try:
                bot_embedding = await embed_text(bot_msg.content[:500])
                similarity = _cosine_similarity(new_embedding, bot_embedding)
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = bot_msg
            except Exception:
                continue

        if best_match:
            logger.debug(
                f"Implicit reply detected (sim={best_similarity:.3f}): "
                f"'{new_message_content[:50]}' → '{best_match.content[:50]}'"
            )

        return best_match

    # ----- prompt building (enhanced with C1, D2, A3) -----

    async def build_prompt(
        self,
        channel_id: str,
        user_message: str,
        user_name: str,
        enable_rag: bool = True,
        channel_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, str]], float]:
        """
        Build the complete message list for an Ollama chat call.

        Returns:
            (messages, temperature) — ready for ollama.chat()
        """
        # 1. Retrieve RAG context (now returns 3-tuple with domain prompt + metric)
        rag_context = ""
        domain_prompt = ""
        if enable_rag:
            try:
                recent_texts = [
                    m.content for m in self._contexts[channel_id].get_recent(5)
                ]
                rag_context, domain_prompt_text, metric = await build_rag_context(
                    query=user_message,
                    recent_messages=recent_texts,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    user_id=user_id,
                )
                if domain_prompt_text:
                    domain_prompt = f"\n# 🎯 Domain-Specific Instructions\n{domain_prompt_text}\n"

                # B2: Store query_id for feedback mapping
                if metric and metric.query_id:
                    self.set_last_query_id(channel_id, metric.query_id)

            except Exception as e:
                logger.warning(f"RAG retrieval failed (non-fatal): {e}")

        # 2. Build system prompt (C1 enhanced)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            current_date=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            rag_context=rag_context,
            domain_prompt=domain_prompt,
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # 3. Add conversation history
        context = self._contexts[channel_id]
        history = context.get_recent()
        for msg in history:
            messages.append(msg.to_chat_dict())

        # 4. Add current user message (may already be tracked, but ensure it's last)
        if not history or history[-1].content != user_message:
            messages.append({"role": "user", "content": f"@{user_name}: {user_message}"})

        return messages, PERSONALITY_TEMPERATURE

    # ----- housekeeping -----

    def _maybe_cleanup(self) -> None:
        """Periodically remove expired contexts to free memory."""
        self._cleanup_counter += 1
        if self._cleanup_counter % 100 != 0:
            return
        expired = [k for k, v in self._contexts.items() if v.is_expired()]
        for k in expired:
            del self._contexts[k]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired channel contexts.")

    def stats(self) -> Dict[str, Any]:
        """Return stats about active contexts."""
        return {
            "active_channels": len(self._contexts),
            "total_messages": sum(
                len(c.messages) for c in self._contexts.values()
            ),
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
