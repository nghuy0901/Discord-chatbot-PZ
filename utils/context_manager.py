"""
Context manager — maintains short-term conversation windows per channel/thread,
detects implicit contextual replies via embedding similarity, and builds
the full prompt for Ollama including system prompt, RAG results, and history.

Enhanced with:
- C1: Advanced RAG-aware system prompt (loaded from prompts/templates/*.txt)
- B1: Smart response formatting rules in prompt
- D2: Knowledge base domain-specific prompts
- A3: Metrics integration
- F1: Conversation awareness (user identification + emotional reset)
"""

import os
import time
import logging
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Deque, Tuple

from rag.retriever import build_rag_result
from rag.result import RAGBuildResult
from rag.embedder import embed_text
from prompts.system_prompt import build_system_prompt

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
FACTUAL_TEMPERATURE: float = float(os.getenv("RAG_FACTUAL_TEMPERATURE", "0.1"))



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
        self._query_ids_by_bot_message: Dict[str, str] = {}

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

    def set_query_id_for_bot_message(self, bot_message_id: str, query_id: str) -> None:
        self._query_ids_by_bot_message[bot_message_id] = query_id

    def get_query_id_for_bot_message(self, bot_message_id: str) -> Optional[str]:
        return self._query_ids_by_bot_message.get(bot_message_id)

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

    # ----- prompt building (enhanced with C1, D2, A3, F1) -----

    async def build_prompt(
        self,
        channel_id: str,
        user_message: str,
        user_name: str,
        enable_rag: bool = True,
        channel_name: Optional[str] = None,
        user_id: Optional[str] = None,
        request_context: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, str]], float, Optional[str], Optional[RAGBuildResult]]:
        """
        Build the complete message list for an Ollama chat call.

        Returns:
            (messages, temperature, query_intent, rag_result) — ready for chat()
            query_intent: "analytical" | "narrative" | "hybrid" | "conversation" | None
        """
        # 1. Retrieve RAG context (now returns 3-tuple with domain prompt + metric)
        rag_context = ""
        domain_prompt = ""
        query_intent = None
        rag_result = None
        if enable_rag:
            try:
                recent_texts = [
                    m.content for m in self._contexts[channel_id].get_recent(5)
                ]
                rag_result = await build_rag_result(
                    query=user_message,
                    recent_messages=recent_texts,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    user_id=user_id,
                    request_context=request_context,
                )
                rag_context = rag_result.context
                domain_prompt_text = rag_result.domain_prompt
                metric = rag_result.metric
                if domain_prompt_text:
                    domain_prompt = f"\n# 🎯 Domain-Specific Instructions\n{domain_prompt_text}\n"

                # B2: Store query_id for feedback mapping
                if metric and metric.query_id:
                    self.set_last_query_id(channel_id, metric.query_id)

                # Phase 4: Extract query intent for routing
                if metric and metric.query_intent:
                    query_intent = metric.query_intent

            except Exception as e:
                logger.warning(f"RAG retrieval failed (non-fatal): {e}")

        # 2. Build system prompt (C1 enhanced — loaded from prompts/templates/)
        system_prompt = build_system_prompt(
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

        temperature = (
            PERSONALITY_TEMPERATURE
            if query_intent == "conversation"
            else FACTUAL_TEMPERATURE
        )
        return messages, temperature, query_intent, rag_result

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
