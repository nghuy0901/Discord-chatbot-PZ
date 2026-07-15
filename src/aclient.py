"""
Discord client — NomNom Bot.

Manages Discord connection, interaction triggers (mention, reply, thread),
provider-neutral response generation, and RAG integration.

Enhanced with:
- B2: Reaction-based feedback (👍/👎)
- C3: Knowledge admin commands via @mention
- D2: Knowledge base initialization
- A3: Metrics initialization
"""

import os
import re
import asyncio
import logging
import time
from typing import List, Dict, Optional, Any

import discord
from discord import app_commands
from dotenv import load_dotenv

from src.log import logger
from src import personas
from src.observability.request_context import RequestContext
from src.ollama_provider import (
    chat_completion,
    chat_with_tools,
    stream_to_discord_chunks,
    ensure_model,
    health_check,
)
from rag.responses import decision_response
from rag.result import RAGDecision
from rag.query_preprocessor import requires_structured_tools
from utils.context_manager import ContextManager
from utils.message_utils import send_split_message

# PZ Structured Data tools (Phase 3)
try:
    from knowledge.structured.pz_tools import PZ_TOOLS, PZ_TOOL_FUNCTIONS
    STRUCTURED_TOOLS_AVAILABLE = True
except Exception:
    PZ_TOOLS = []
    PZ_TOOL_FUNCTIONS = {}
    STRUCTURED_TOOLS_AVAILABLE = False

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENABLE_RAG: bool = os.getenv("ENABLE_RAG", "true").lower() == "true"
ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
ENABLE_IMPLICIT_REPLIES: bool = (
    os.getenv("ENABLE_IMPLICIT_REPLIES", "false").lower() == "true"
)
ENABLE_KNOWLEDGE_BASE: bool = os.getenv("ENABLE_KNOWLEDGE_BASE", "true").lower() == "true"
ENABLE_FEEDBACK: bool = os.getenv("ENABLE_FEEDBACK", "true").lower() == "true"
ENABLE_TOOL_CALLING: bool = os.getenv("ENABLE_TOOL_CALLING", "true").lower() == "true"
ENABLE_NAME_MENTION: bool = (
    os.getenv("ENABLE_NAME_MENTION", "true").lower() == "true"
)
STREAM_EDIT_INTERVAL: float = float(os.getenv("STREAM_EDIT_INTERVAL", "1.5"))
MAX_RESPONSE_LENGTH: int = int(os.getenv("MAX_RESPONSE_LENGTH", "4000"))

# Bot name aliases for mention detection (case-insensitive)
BOT_NAME_ALIASES: list = [
    alias.strip().lower()
    for alias in os.getenv("BOT_NAME_ALIASES", "nomnom,nom nom,nôm nôm").split(",")
    if alias.strip()
]

# Admin user IDs (comma-separated in .env)
ADMIN_USER_IDS: set = set(
    os.getenv("ADMIN_USER_IDS", "").split(",")
) - {""}

# C3: Knowledge admin command keywords
KB_COMMANDS = {
    "reload knowledge": "reload",
    "scan knowledge": "scan",
    "kb reload": "reload",
    "kb scan": "scan",
    "kb status": "scan",
    "knowledge status": "scan",
    "reload kb": "reload",
}


class NomNomClient(discord.Client):
    """
    NomNom Discord client with Grok-style interaction triggers:
    - @mention
    - Reply to bot message
    - Active thread participation
    - (optional) Implicit similarity-based replies
    - (C3) Knowledge admin commands via @mention
    - (B2) Reaction-based feedback
    """

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.reactions = True  # B2: Need reaction events
        super().__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)

        # Context manager (short-term memory per channel + RAG bridge)
        self.context_manager = ContextManager()

        # Legacy provider manager (kept for /provider command compatibility)
        try:
            from src.providers import ProviderManager, ProviderType
            self.provider_manager = ProviderManager()
            default_provider = os.getenv("DEFAULT_PROVIDER") or os.getenv("LLM_PROVIDER")
            try:
                self.provider_manager.set_current_provider(ProviderType(default_provider))
            except (TypeError, ValueError):
                logger.warning("Configured default provider is not available: %s", default_provider)
        except Exception:
            self.provider_manager = None

        self.current_model = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "vllm-local"))

        # Bot settings
        self.activity = discord.Activity(
            type=discord.ActivityType.listening,
            name="@mentions & replies",
        )
        self.isPrivate = False
        self.is_replying_all = os.getenv("REPLYING_ALL", "False") == "True"
        self.replying_all_discord_channel_id = os.getenv("REPLYING_ALL_DISCORD_CHANNEL_ID")
        self.current_channel = None

        # Message queue for rate-limiting
        self.message_queue: asyncio.Queue = asyncio.Queue()

        # Rate-limit tracker: channel_id → last_response_time
        self._rate_limits: Dict[str, float] = {}
        self._rate_limit_seconds: float = float(os.getenv("RATE_LIMIT_SECONDS", "2"))

    # ------------------------------------------------------------------
    # Mention resolution: convert @Username text → real Discord mentions
    # ------------------------------------------------------------------
    def _resolve_mentions(self, text: str, guild: discord.Guild) -> str:
        """Replace @Username plain text with real Discord <@user_id> mentions.

        The LLM often outputs '@SomeName' as plain text.  This scans for
        such patterns and substitutes them with the proper mention syntax
        so the tagged user actually receives a notification.
        """
        if not guild or not text:
            return text

        # Match @Name patterns (but not already-resolved <@id> mentions)
        at_pattern = re.compile(r'(?<![<\w])@(\w[\w._ ]{0,30}\w)')

        def _replace(match: re.Match) -> str:
            name = match.group(1).strip()
            # Try exact match on display_name, global_name, username
            for member in guild.members:
                if (
                    member.display_name.lower() == name.lower()
                    or member.name.lower() == name.lower()
                    or (member.global_name and member.global_name.lower() == name.lower())
                ):
                    return member.mention  # e.g. <@123456789>
            # No match found — keep original text
            return match.group(0)

        return at_pattern.sub(_replace, text)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup_hook(self) -> None:
        """Called once when the bot starts — initialise DB, KB, and models."""
        # Check configured LLM provider health
        if await health_check():
            logger.info("✅ LLM provider is configured.")
            await ensure_model()
        else:
            logger.warning("⚠️ LLM provider is not configured — responses will fail until it is set.")

        # Initialise RAG database (non-blocking)
        if ENABLE_RAG:
            try:
                from rag.db import init_db
                await init_db()
                logger.info("✅ RAG database initialised.")
            except Exception as e:
                logger.warning(f"⚠️ RAG DB init failed (non-fatal): {e}")

            # A3: Initialize metrics tables
            try:
                from rag.metrics import get_metrics_manager
                metrics = get_metrics_manager()
                await metrics.init_db()
                logger.info("✅ RAG metrics tables initialised.")
            except Exception as e:
                logger.warning(f"⚠️ Metrics DB init failed (non-fatal): {e}")

        # D2: Initialize Knowledge Base
        if ENABLE_KNOWLEDGE_BASE:
            try:
                from knowledge.manager import get_knowledge_manager
                kb = get_knowledge_manager()
                result = await kb.initialize_catalog()
                total = sum(result.values())
                logger.info(
                    f"✅ Knowledge catalog initialized: {total} files across {len(result)} domains"
                )
            except Exception as e:
                logger.warning(f"⚠️ Knowledge catalog init failed (non-fatal): {e}")

        # E1: Initialize BM25 indices for Hybrid RAG
        if ENABLE_RAG:
            try:
                from rag.bm25_search import init_bm25_indices
                bm25_results = await init_bm25_indices()
                total_bm25 = sum(bm25_results.values())
                logger.info(
                    f"✅ BM25 indices initialized: {total_bm25} docs "
                    f"(chat={bm25_results.get('chat_history', 0)}, "
                    f"kb={bm25_results.get('knowledge_base', 0)})"
                )
            except Exception as e:
                logger.warning(f"⚠️ BM25 index init failed (non-fatal, hybrid search degrades to vector-only): {e}")

    async def send_start_prompt(self) -> None:
        """No-op — NomNom doesn't need an initial prompt broadcast."""
        if ENABLE_TOOL_CALLING and STRUCTURED_TOOLS_AVAILABLE:
            logger.info(
                f"✅ Tool calling enabled: {len(PZ_TOOLS)} PZ tools available "
                f"({', '.join(PZ_TOOL_FUNCTIONS.keys())})"
            )
        elif ENABLE_TOOL_CALLING and not STRUCTURED_TOOLS_AVAILABLE:
            logger.warning("⚠️ Tool calling enabled but PZ tools failed to load")
        else:
            logger.info("ℹ️ Tool calling disabled")
        logger.info("NomNom ready — responding to @mentions and replies.")

    # ------------------------------------------------------------------
    # Trigger detection
    # ------------------------------------------------------------------

    def should_respond(self, message: discord.Message) -> bool:
        """
        Determine whether NomNom should respond to this message.
        """
        if message.author == self.user:
            return False
        if message.author.bot:
            return False

        # 1. Direct @mention
        if self.user in message.mentions:
            return True

        # 2. Reply to bot message
        if message.reference and message.reference.resolved:
            ref_msg = message.reference.resolved
            if hasattr(ref_msg, "author") and ref_msg.author == self.user:
                return True

        # 3. replyAll mode
        if self.is_replying_all:
            if self.replying_all_discord_channel_id:
                if str(message.channel.id) == str(self.replying_all_discord_channel_id):
                    return True
            else:
                return True

        # 4. Active thread where bot participated
        channel_id = str(message.channel.id)
        if isinstance(message.channel, discord.Thread):
            if self.context_manager.has_bot_participated(channel_id):
                return True

        # 5. Name mention detection (F1): Check if bot's name appears in message text
        if ENABLE_NAME_MENTION and message.content:
            content_lower = message.content.lower()
            for alias in BOT_NAME_ALIASES:
                if alias in content_lower:
                    logger.info(
                        f"🔍 Name mention detected: '{alias}' in message from "
                        f"{message.author.display_name}"
                    )
                    return True

        return False

    async def should_respond_implicit(self, message: discord.Message) -> bool:
        """Check for implicit similarity-based reply (expensive, optional)."""
        if not ENABLE_IMPLICIT_REPLIES:
            return False

        channel_id = str(message.channel.id)
        match = await self.context_manager.detect_implicit_reply(
            channel_id, message.content
        )
        return match is not None

    # ------------------------------------------------------------------
    # C3: Knowledge admin command detection
    # ------------------------------------------------------------------

    def _is_admin(self, user_id: str) -> bool:
        """Check if a user is an admin."""
        return user_id in ADMIN_USER_IDS

    def _detect_kb_command(self, message_content: str) -> Optional[str]:
        """
        Detect knowledge base admin commands in @mention messages.
        Returns command type or None.
        """
        content_lower = message_content.lower().strip()
        for keyword, command in KB_COMMANDS.items():
            if keyword in content_lower:
                return command
        return None

    async def _handle_kb_command(
        self, message: discord.Message, command: str
    ) -> bool:
        """
        Handle a knowledge base admin command.
        Returns True if handled, False otherwise.
        """
        user_id = str(message.author.id)

        # Admin check
        if not self._is_admin(user_id):
            await message.reply(
                "❌ Bạn không có quyền admin để thực hiện lệnh này.",
                mention_author=False,
            )
            return True

        if command == "reload":
            await message.add_reaction("🔄")
            try:
                from knowledge.manager import get_knowledge_manager
                kb = get_knowledge_manager()
                result = await kb.reload()
                total = sum(result.values())
                status_lines = [f"📚 **Knowledge Base Reloaded**\n"]
                for domain, count in result.items():
                    status_lines.append(f"• **{domain}**: {count} chunks")
                status_lines.append(f"\n**Total**: {total} chunks")

                # E1: Refresh BM25 index for hybrid search
                try:
                    from rag.bm25_search import get_kb_bm25
                    kb_bm25 = get_kb_bm25()
                    bm25_count = await kb_bm25.refresh_from_kb()
                    status_lines.append(f"🔍 **BM25 Index**: {bm25_count} docs reindexed")
                except Exception as bm25_err:
                    status_lines.append(f"⚠️ BM25 reindex skipped: {bm25_err}")

                await message.reply("\n".join(status_lines), mention_author=False)
                await message.remove_reaction("🔄", self.user)
                await message.add_reaction("✅")
            except Exception as e:
                await message.reply(f"❌ Reload failed: {e}", mention_author=False)
                await message.add_reaction("❌")
            return True

        elif command == "scan":
            try:
                from knowledge.manager import get_knowledge_manager
                kb = get_knowledge_manager()
                status = kb.get_status()
                lines = [f"📊 **Knowledge Base Status**\n"]
                lines.append(f"• Initialized: {'✅' if status['initialized'] else '❌'}")
                lines.append(f"• Total domains: {status['total_domains']}")
                lines.append(f"• Total documents: {status['total_docs']}")
                lines.append(f"• Total chunks: {status['total_chunks']}")
                if status['domains']:
                    lines.append("\n**Domains:**")
                    for name, info in status['domains'].items():
                        prompt_icon = "📝" if info.get("has_prompt") else "📄"
                        lines.append(
                            f"  {prompt_icon} **{name}**: "
                            f"{info['doc_count']} docs, {info['chunk_count']} chunks"
                        )
                else:
                    lines.append("\n*No domains loaded. Add documents to `knowledge/docs/<domain>/`*")
                await message.reply("\n".join(lines), mention_author=False)
            except Exception as e:
                await message.reply(f"❌ Scan failed: {e}", mention_author=False)
            return True

        return False

    # ------------------------------------------------------------------
    # B2: Reaction feedback handler
    # ------------------------------------------------------------------

    async def handle_reaction_feedback(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Process 👍/👎 reaction on bot messages for feedback tracking.
        """
        if not ENABLE_FEEDBACK:
            return

        # Only process reactions on bot messages
        if reaction.message.author != self.user:
            return

        # Don't process bot's own reactions
        if user.bot:
            return

        emoji = str(reaction.emoji)
        if emoji not in ("👍", "👎"):
            return

        score = 1 if emoji == "👍" else -1
        channel_id = str(reaction.message.channel.id)
        message_id = str(reaction.message.id)
        user_id = str(user.id)

        # Get the query_id associated with this exact bot message.
        query_id = self.context_manager.get_query_id_for_bot_message(message_id)
        if not query_id:
            query_id = self.context_manager.get_last_query_id(channel_id)
        if not query_id:
            logger.debug("No query_id found for feedback — skipping")
            return

        try:
            from rag.metrics import get_metrics_manager
            metrics = get_metrics_manager()
            await metrics.record_feedback(
                message_id=message_id,
                query_id=query_id,
                user_id=user_id,
                score=score,
            )
            logger.info(
                f"{'👍' if score > 0 else '👎'} Feedback recorded: "
                f"user={user.display_name}, msg={message_id[:8]}"
            )
        except Exception as e:
            logger.warning(f"Failed to record feedback: {e}")

    # ------------------------------------------------------------------
    # Message processing queue
    # ------------------------------------------------------------------

    async def process_messages(self) -> None:
        """Background loop that processes the message queue."""
        while True:
            try:
                message, user_message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=5.0
                )
                await self._generate_and_send(message, user_message)
                self.message_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"Error in process_messages: {e}")

    async def enqueue_message(self, message, user_message: str) -> None:
        """Add a message to the processing queue."""
        if hasattr(message, "response"):
            try:
                await message.response.defer(ephemeral=self.isPrivate)
            except Exception:
                pass
        await self.message_queue.put((message, user_message))

    # ------------------------------------------------------------------
    # Core response generation (enhanced with A3 metrics timing)
    # ------------------------------------------------------------------

    def _abstain_language(self, text: str) -> str:
        """Best-effort language for canned refusals when no RAG metric exists."""
        try:
            from rag.query_preprocessor import get_preprocessor

            return get_preprocessor()._detect_language(text)
        except Exception:
            return "vi"

    @staticmethod
    def _safe_error_message(request_context: Optional[RequestContext] = None) -> str:
        """User-facing error text that never leaks internal exception details
        (DSNs, hosts, paths, SQL) — the full error is logged server-side (H8)."""
        rid = getattr(request_context, "request_id", "") if request_context else ""
        ref = f" (mã lỗi: {str(rid)[:8]})" if rid else ""
        return f"❌ Xin lỗi, mình gặp sự cố nội bộ{ref}. Bạn thử lại sau nhé."

    async def _generate_and_send(
        self, message: discord.Message, user_message: str
    ) -> None:
        """
        Full response flow:
        1. Track message in context
        2. Build prompt (system + RAG + history)
        3. Generate via configured LLM provider (streaming or batch)
        4. Send to Discord with typing indicator
        5. Track bot response in context
        6. B2: Add feedback reactions
        """
        channel_id = str(message.channel.id)
        request_context = RequestContext.new(
            channel_id=channel_id,
            user_id=str(message.author.id),
            source="discord",
        )
        author_name = str(message.author.display_name)

        # Get channel name for domain detection (D2)
        channel_name = None
        if hasattr(message.channel, "name"):
            channel_name = message.channel.name

        # Rate-limit check
        now = time.time()
        last = self._rate_limits.get(channel_id, 0)
        if now - last < self._rate_limit_seconds:
            await asyncio.sleep(self._rate_limit_seconds - (now - last))
        self._rate_limits[channel_id] = time.time()

        response_start = time.time()

        # Check cache if RAG is enabled
        cached_response = None
        detected_domain = None
        query_intent_str = None
        if ENABLE_RAG:
            try:
                from rag.query_preprocessor import get_preprocessor
                from utils.cache import get_query_cache
                
                preprocessor = get_preprocessor()
                _, query_meta = preprocessor.preprocess(user_message, channel_name)
                detected_domain = query_meta.get("domain")
                query_intent = query_meta.get("query_intent")
                query_intent_str = query_intent.value if hasattr(query_intent, 'value') else str(query_intent)
                
                if query_intent_str in ("analytical", "hybrid", "narrative"):
                    cache = get_query_cache()
                    cached_response = await cache.get(user_message, detected_domain)
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")

        if cached_response:
            response_text = cached_response["response_text"]
            # Re-attach the chunk-level citation footer for cached answers
            # (it is stored separately from the raw answer text).
            if os.getenv("RAG_SHOW_CITATIONS", "true").lower() == "true":
                try:
                    from rag.citations import render_sources_footer_from_dicts

                    _footer = render_sources_footer_from_dicts(
                        response_text,
                        cached_response.get("provenance", []),
                        language=(cached_response.get("language") or "vi"),
                    )
                    if _footer:
                        response_text = f"{response_text}\n\n{_footer}"
                except Exception:
                    pass
            # Send response to Discord
            bot_msg = await self._send_response(message, response_text)
            
            # Record cache hit metric
            response_time = (time.time() - response_start) * 1000
            try:
                from rag.metrics import get_metrics_manager, RAGMetric
                metrics = get_metrics_manager()
                metric = RAGMetric(
                    channel_id=channel_id,
                    user_id=str(message.author.id),
                    original_query=user_message,
                    processed_query=user_message,
                    detected_domain=detected_domain,
                    query_intent=query_intent_str,
                    cache_hit=True,
                    response_time_ms=response_time,
                    response_length=len(response_text),
                    prompt_tokens=cached_response.get("prompt_tokens", 0),
                    completion_tokens=cached_response.get("completion_tokens", 0),
                    total_tokens=cached_response.get("prompt_tokens", 0) + cached_response.get("completion_tokens", 0),
                    rag_decision="answer",
                    provenance=cached_response.get("provenance", []),
                    trusted_source_count=len(cached_response.get("provenance", [])),
                )
                await metrics.record(metric)
                self.context_manager.set_last_query_id(channel_id, metric.query_id)
            except Exception as e:
                logger.warning(f"Failed to record cache hit metric: {e}")
                
            # Track message in context window
            self.context_manager.track_message(
                channel_id=channel_id,
                message_id=f"bot-{message.id}",
                author_id=str(self.user.id),
                author_name=str(self.user.display_name),
                content=response_text,
                is_bot=True,
            )
            
            # Add feedback reactions
            if ENABLE_FEEDBACK and bot_msg:
                try:
                    await bot_msg.add_reaction("👍")
                    await bot_msg.add_reaction("👎")
                    self.context_manager.set_last_bot_msg_id(channel_id, str(bot_msg.id))
                    self.context_manager.set_query_id_for_bot_message(str(bot_msg.id), metric.query_id)
                except Exception as e:
                    logger.debug(f"Failed to add feedback reactions for cached response: {e}")
            return

        try:
            # Build prompt with RAG + context (now passes channel_name for D2)
            prompt_messages, temperature, query_intent, rag_result = await self.context_manager.build_prompt(
                channel_id=channel_id,
                user_message=user_message,
                user_name=author_name,
                enable_rag=ENABLE_RAG,
                channel_name=channel_name,
                user_id=str(message.author.id),
                request_context=request_context,
            )

            response_text = ""
            bot_msg = None
            answered_with_tools = False
            tool_results = []
            finalized = None
            tools_available = (
                ENABLE_TOOL_CALLING
                and STRUCTURED_TOOLS_AVAILABLE
                and PZ_TOOLS
            )
            use_tools_for_query = tools_available and requires_structured_tools(query_intent)

            # C1: RAG was enabled but the build returned nothing (the exception
            # was swallowed inside build_prompt). Do NOT fall through to a
            # free-form LLM answer — that is exactly how the bot used to
            # fabricate game facts on a transient DB/embedding error. Abstain.
            if ENABLE_RAG and rag_result is None:
                response_text = decision_response(
                    self._abstain_language(user_message),
                    RAGDecision.ABSTAIN,
                    event="discord_rag_build_failed",
                    query=user_message,
                    reason="rag_result_missing_after_build_prompt",
                    channel_id=channel_id,
                    user_id=str(message.author.id),
                )
                bot_msg = await self._send_response(message, response_text)
                self.context_manager.track_message(
                    channel_id=channel_id,
                    message_id=f"bot-{message.id}",
                    author_id=str(self.user.id),
                    author_name=str(self.user.display_name),
                    content=response_text,
                    is_bot=True,
                )
                return

            if (
                rag_result
                and rag_result.decision is not RAGDecision.ANSWER
                and not use_tools_for_query
            ):
                response_text = decision_response(
                    rag_result.metric.query_language,
                    rag_result.decision,
                    event="discord_pre_generation_decision",
                    query=user_message,
                    rag_result=rag_result,
                    channel_id=channel_id,
                    user_id=str(message.author.id),
                )
                bot_msg = await self._send_response(message, response_text)
                response_time = (time.time() - response_start) * 1000
                rag_result.metric.response_time_ms = response_time
                rag_result.metric.response_length = len(response_text)
                rag_result.metric.citation_coverage = 0.0
                try:
                    from rag.metrics import get_metrics_manager

                    await get_metrics_manager().record(rag_result.metric)
                except Exception:
                    pass
                self.context_manager.set_last_query_id(channel_id, rag_result.metric.query_id)
                self.context_manager.track_message(
                    channel_id=channel_id,
                    message_id=f"bot-{message.id}",
                    author_id=str(self.user.id),
                    author_name=str(self.user.display_name),
                    content=response_text,
                    is_bot=True,
                )
                if ENABLE_FEEDBACK and bot_msg:
                    try:
                        await bot_msg.add_reaction("👍")
                        await bot_msg.add_reaction("👎")
                        self.context_manager.set_last_bot_msg_id(channel_id, str(bot_msg.id))
                        self.context_manager.set_query_id_for_bot_message(
                            str(bot_msg.id),
                            rag_result.metric.query_id,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to add feedback reactions: {e}")
                return

            # Phase 4: Intent-based routing
            # Route decision based on query_intent:
            #   ANALYTICAL → always use tools (no streaming — need full response for tool detection)
            #   HYBRID     → try tools first, fallback to streaming if no tools triggered
            #   NARRATIVE  → skip tools entirely, go straight to streaming
            #   CONVERSATION → skip tools entirely, go straight to streaming
            #   None       → hybrid behavior (backward compat)

            logger.info(
                f"🔀 Routing: intent={query_intent}, "
                f"tools_available={tools_available}, "
                f"use_tools={use_tools_for_query}"
            )

            if ENABLE_STREAMING:
                if use_tools_for_query:
                    # ANALYTICAL or HYBRID: Try tool calling first (non-streaming,
                    # because tool calls require full response to detect calls)
                    async with message.channel.typing():
                        response_text = await chat_with_tools(
                            messages=prompt_messages,
                            tools=PZ_TOOLS,
                            tool_functions=PZ_TOOL_FUNCTIONS,
                            temperature=temperature,
                            request_context=request_context,
                        )
                        from src.ollama_provider import get_last_tool_results
                        from rag.groundedness import sources_from_tool_results

                        tool_results = get_last_tool_results()
                        answered_with_tools = bool(sources_from_tool_results(tool_results))

                    if response_text:
                        if (
                            rag_result
                            and rag_result.decision is not RAGDecision.ANSWER
                            and not answered_with_tools
                        ):
                            response_text = decision_response(
                                rag_result.metric.query_language,
                                rag_result.decision,
                                event="discord_tool_response_without_tool_evidence",
                                query=user_message,
                                rag_result=rag_result,
                                channel_id=channel_id,
                                user_id=str(message.author.id),
                            )
                        # Tool calling produced a response — send directly
                        if len(response_text) > MAX_RESPONSE_LENGTH:
                            response_text = (
                                response_text[:MAX_RESPONSE_LENGTH]
                                + "\n\n*…response truncated*"
                            )
                        # Resolve @Username → real Discord mentions
                        response_text = self._resolve_mentions(
                            response_text, getattr(message, 'guild', None)
                        )
                        bot_msg = await message.channel.send(response_text[:2000])
                        # Send remaining parts if response > 2000 chars
                        remaining = response_text[2000:]
                        while remaining:
                            await message.channel.send(remaining[:2000])
                            remaining = remaining[2000:]
                    else:
                        if rag_result and rag_result.decision is not RAGDecision.ANSWER:
                            response_text = decision_response(
                                rag_result.metric.query_language,
                                rag_result.decision,
                                event="discord_empty_tool_response_decision",
                                query=user_message,
                                rag_result=rag_result,
                                channel_id=channel_id,
                                user_id=str(message.author.id),
                            )
                            bot_msg = await self._send_response(message, response_text)
                        else:
                            # Empty/no-tool response — fallback to streaming
                            response_text, bot_msg = await self._stream_response(
                                message, prompt_messages, temperature, request_context
                            )
                else:
                    # NARRATIVE or CONVERSATION: Pure streaming (skip tool overhead)
                    response_text, bot_msg = await self._stream_response(
                        message, prompt_messages, temperature, request_context
                    )
            else:
                # Non-streaming mode
                async with message.channel.typing():
                    if use_tools_for_query:
                        response_text = await chat_with_tools(
                            messages=prompt_messages,
                            tools=PZ_TOOLS,
                            tool_functions=PZ_TOOL_FUNCTIONS,
                            temperature=temperature,
                            request_context=request_context,
                        )
                        from src.ollama_provider import get_last_tool_results
                        from rag.groundedness import sources_from_tool_results

                        tool_results = get_last_tool_results()
                        answered_with_tools = bool(sources_from_tool_results(tool_results))
                        if (
                            rag_result
                            and rag_result.decision is not RAGDecision.ANSWER
                            and not answered_with_tools
                        ):
                            response_text = decision_response(
                                rag_result.metric.query_language,
                                rag_result.decision,
                                event="discord_nonstream_tool_response_without_tool_evidence",
                                query=user_message,
                                rag_result=rag_result,
                                channel_id=channel_id,
                                user_id=str(message.author.id),
                            )
                    else:
                        response_text = await chat_completion(
                            messages=prompt_messages,
                            temperature=temperature,
                            request_context=request_context,
                        )
                    if len(response_text) > MAX_RESPONSE_LENGTH:
                        response_text = (
                            response_text[:MAX_RESPONSE_LENGTH]
                            + "\n\n*…response truncated*"
                        )
                    bot_msg = await self._send_response(message, response_text)

            # ---- Groundedness gate + chunk-level citations (answered turns) ----
            # Capture this turn's token usage BEFORE the groundedness judge call
            # (which would otherwise clobber the provider's last-usage snapshot).
            main_token_usage = None
            if (
                rag_result
                and rag_result.decision is RAGDecision.ANSWER
                and not answered_with_tools  # tool answers come from the SQLite DB,
                # which isn't in the vector provenance the groundedness gate checks
            ):
                try:
                    from src.ollama_provider import get_last_token_usage as _gltu
                    main_token_usage = _gltu()
                except Exception:
                    main_token_usage = None
                try:
                    from rag.answer_finalize import finalize_rag_answer

                    finalized = await finalize_rag_answer(
                        rag_result, user_message, response_text
                    )
                    if finalized.overridden:
                        # Ungrounded answer → replace what the user already saw.
                        response_text = finalized.text
                        if bot_msg:
                            try:
                                await bot_msg.edit(content=response_text[:2000])
                            except Exception:
                                pass
                    elif finalized.footer and bot_msg:
                        # Chunk-level "Sources" footer as a follow-up message.
                        try:
                            await message.channel.send(finalized.footer[:2000])
                        except Exception:
                            pass
                except Exception as fe:
                    logger.warning(f"Answer finalize failed (non-fatal): {fe}")
            elif answered_with_tools:
                # C2: tool answers skip the vector-provenance gate above, so
                # verify them against the actual tool outputs instead.
                try:
                    from src.ollama_provider import get_last_token_usage as _gltu

                    main_token_usage = _gltu()
                except Exception:
                    main_token_usage = None
                try:
                    from rag.answer_finalize import finalize_tool_answer
                    from src.ollama_provider import get_last_tool_results

                    lang = (
                        rag_result.metric.query_language
                        if rag_result
                        else self._abstain_language(user_message)
                    )
                    finalized = await finalize_tool_answer(
                        user_message,
                        response_text,
                        tool_results,
                        language=lang,
                    )
                    if finalized.overridden:
                        response_text = finalized.text
                        if bot_msg:
                            try:
                                await bot_msg.edit(content=response_text[:2000])
                            except Exception:
                                pass
                except Exception as fe:
                    logger.warning(f"Tool answer finalize failed (non-fatal): {fe}")

            # A3: Record response timing + token usage
            response_time = (time.time() - response_start) * 1000
            try:
                from rag.metrics import get_metrics_manager
                from src.ollama_provider import get_last_token_usage
                metrics = get_metrics_manager()
                last_metric = rag_result.metric if rag_result else None
                if last_metric:
                    if finalized is not None:
                        last_metric.rag_decision = finalized.decision.value
                        if answered_with_tools:
                            last_metric.decision_reason = (
                                "tool_answer_ungrounded"
                                if finalized.overridden
                                else "tool_answer_grounded"
                            )
                            if finalized.groundedness is not None:
                                last_metric.groundedness_score = finalized.groundedness.score
                                last_metric.groundedness_reason = finalized.groundedness.reason
                                last_metric.groundedness_unsupported_count = len(
                                    finalized.groundedness.unsupported
                                )
                                last_metric.groundedness_error = finalized.groundedness.error
                    last_metric.response_time_ms = response_time
                    last_metric.response_length = len(response_text)
                    # Cost tracking: record token usage from LLM call.
                    # Prefer the snapshot taken before the groundedness judge call.
                    token_usage = main_token_usage or get_last_token_usage()
                    if token_usage:
                        last_metric.prompt_tokens = token_usage.get("prompt_tokens", 0)
                        last_metric.completion_tokens = token_usage.get("completion_tokens", 0)
                        last_metric.total_tokens = token_usage.get("total_tokens", 0)
                        last_metric.estimated_cost_usd = token_usage.get("estimated_cost_usd", 0.0)

                    # Redis caching: cache the response if it was a cache miss and is cacheable
                    provenance = (
                        [item.to_dict() for item in rag_result.provenance]
                        if rag_result
                        else []
                    )
                    from rag.citations import referenced_labels

                    last_metric.citation_coverage = 1.0 if (
                        provenance and referenced_labels(response_text)
                    ) else 0.0
                    if (
                        not getattr(last_metric, "cache_hit", False)
                        and last_metric.rag_decision == "answer"
                        and not answered_with_tools
                        and last_metric.query_intent in ("analytical", "hybrid", "narrative")
                        and response_text
                        and provenance
                    ):
                        try:
                            from utils.cache import get_query_cache
                            cache = get_query_cache()
                            await cache.set(
                                query=last_metric.original_query,
                                domain=last_metric.detected_domain,
                                result={
                                    "response_text": response_text,
                                    "decision": "answer",
                                    "provenance": provenance,
                                    "language": last_metric.query_language,
                                    "prompt_tokens": last_metric.prompt_tokens,
                                    "completion_tokens": last_metric.completion_tokens,
                                }
                            )
                        except Exception as ce:
                            logger.warning(f"Failed to cache response: {ce}")

                    await metrics.record(last_metric)
            except Exception:
                pass

            # Track bot response in context
            if response_text:
                self.context_manager.track_message(
                    channel_id=channel_id,
                    message_id=f"bot-{message.id}",
                    author_id=str(self.user.id),
                    author_name=str(self.user.display_name),
                    content=response_text,
                    is_bot=True,
                )

            # B2: Add feedback reactions to bot response
            if ENABLE_FEEDBACK and bot_msg:
                try:
                    await bot_msg.add_reaction("👍")
                    await bot_msg.add_reaction("👎")
                    # Track the bot message ID for feedback mapping
                    self.context_manager.set_last_bot_msg_id(channel_id, str(bot_msg.id))
                    mapped_query_id = (
                        rag_result.metric.query_id
                        if rag_result
                        else self.context_manager.get_last_query_id(channel_id)
                    )
                    if mapped_query_id:
                        self.context_manager.set_query_id_for_bot_message(
                            str(bot_msg.id),
                            mapped_query_id,
                        )
                except Exception as e:
                    logger.debug(f"Failed to add feedback reactions: {e}")

        except Exception as e:
            logger.exception(f"Response generation failed: {e}")
            error_msg = self._safe_error_message(request_context)
            try:
                await self._send_response(message, error_msg)
            except Exception:
                pass

    async def _stream_response(
        self,
        message: discord.Message,
        prompt_messages: List[Dict[str, str]],
        temperature: float,
        request_context: Optional[RequestContext] = None,
    ) -> tuple:
        """
        Stream LLM tokens and progressively edit a Discord message.
        Returns tuple of (final_text, bot_message) for feedback tracking.
        """
        # Send initial "thinking" message
        if hasattr(message, "followup"):
            bot_msg = await message.followup.send("🧠 *Thinking…*", wait=True)
        else:
            bot_msg = await message.channel.send("🧠 *Thinking…*")

        final_text = ""
        edit_count = 0
        MAX_EDITS = 30

        try:
            async for accumulated in stream_to_discord_chunks(
                messages=prompt_messages,
                temperature=temperature,
                chunk_interval=STREAM_EDIT_INTERVAL,
                request_context=request_context,
            ):
                final_text = accumulated
                if edit_count < MAX_EDITS:
                    display = accumulated
                    if len(display) > 1950:
                        display = display[:1950] + "\n\n*…streaming*"
                    try:
                        await bot_msg.edit(content=display)
                        edit_count += 1
                    except discord.HTTPException:
                        pass

            if final_text:
                # Resolve @Username → real Discord mentions
                final_text = self._resolve_mentions(
                    final_text, getattr(message, 'guild', None)
                )
                if len(final_text) > 2000:
                    await bot_msg.edit(content=final_text[:1990])
                    remaining = final_text[1990:]
                    while remaining:
                        chunk = remaining[:1990]
                        remaining = remaining[1990:]
                        await message.channel.send(chunk)
                else:
                    await bot_msg.edit(content=final_text)
            else:
                await bot_msg.edit(content="🤔 I generated an empty response. Try rephrasing?")

        except Exception as e:
            logger.error(f"Stream response error: {e}")
            try:
                await bot_msg.edit(content=self._safe_error_message(request_context))
            except Exception:
                pass
            final_text = ""

        return final_text, bot_msg

    async def _send_response(
        self, message, content: str
    ) -> Any:
        """Send a response, handling slash commands vs regular messages."""
        # Resolve @Username → real Discord mentions
        content = self._resolve_mentions(
            content, getattr(message, 'guild', None)
        )
        if hasattr(message, "followup"):
            await send_split_message(self, content, message)
            return None
        else:
            if len(content) > 2000:
                parts = [content[i : i + 1990] for i in range(0, len(content), 1990)]
                msg = None
                for part in parts:
                    msg = await message.channel.send(part)
                return msg
            else:
                return await message.reply(content, mention_author=False)

    # ------------------------------------------------------------------
    # Slash command response (legacy compatibility)
    # ------------------------------------------------------------------

    async def handle_response(self, user_message: str) -> str:
        """Legacy handle_response for /chat command."""
        response_start = time.time()
        request_context = RequestContext.new(
            channel_id="slash-command",
            user_id="api_user",
            source="discord_slash",
        )
        prompt_messages, temperature, query_intent, rag_result = await self.context_manager.build_prompt(
            channel_id="slash-command",
            user_message=user_message,
            user_name="User",
            enable_rag=ENABLE_RAG,
            request_context=request_context,
        )
        if rag_result and rag_result.decision is not RAGDecision.ANSWER:
            response = decision_response(
                rag_result.metric.query_language,
                rag_result.decision,
                event="discord_slash_pre_generation_decision",
                query=user_message,
                rag_result=rag_result,
                channel_id="slash-command",
                user_id="api_user",
            )
            rag_result.metric.response_time_ms = (time.time() - response_start) * 1000
            rag_result.metric.response_length = len(response)
            rag_result.metric.citation_coverage = 0.0
            try:
                from rag.metrics import get_metrics_manager

                await get_metrics_manager().record(rag_result.metric)
            except Exception:
                pass
            self.context_manager.set_last_query_id("slash-command", rag_result.metric.query_id)
            return response
        # Phase 4: Intent-based routing for slash commands
        use_tools = (
            ENABLE_TOOL_CALLING
            and STRUCTURED_TOOLS_AVAILABLE
            and requires_structured_tools(query_intent)
        )
        if use_tools:
            response = await chat_with_tools(
                messages=prompt_messages,
                tools=PZ_TOOLS,
                tool_functions=PZ_TOOL_FUNCTIONS,
                temperature=temperature,
                request_context=request_context,
            )
        else:
            response = await chat_completion(
                messages=prompt_messages,
                temperature=temperature,
                request_context=request_context,
            )

        from src.ollama_provider import get_last_token_usage

        main_token_usage = get_last_token_usage()
        finalized = None
        answered_with_tools = False
        tool_results = []
        if use_tools:
            from src.ollama_provider import get_last_tool_results
            from rag.groundedness import sources_from_tool_results

            tool_results = get_last_tool_results()
            answered_with_tools = bool(sources_from_tool_results(tool_results))

        # L2: route /chat answers through the same finalize gate as @mentions
        # (groundedness + citations for vector answers, tool-output verification
        # for tool answers) so /chat can't ship an ungrounded answer either.
        try:
            if response and answered_with_tools:
                from rag.answer_finalize import finalize_tool_answer

                finalized = await finalize_tool_answer(
                    user_message,
                    response,
                    tool_results,
                    language=(rag_result.metric.query_language if rag_result else "vi"),
                )
                response = finalized.text
            elif response and rag_result and rag_result.decision is RAGDecision.ANSWER:
                from rag.answer_finalize import finalize_rag_answer

                finalized = await finalize_rag_answer(rag_result, user_message, response)
                response = finalized.text
        except Exception as fe:
            logger.warning(f"/chat finalize failed (non-fatal): {fe}")

        if rag_result:
            metric = rag_result.metric
            if finalized is not None:
                metric.rag_decision = finalized.decision.value
                if answered_with_tools:
                    metric.decision_reason = (
                        "tool_answer_ungrounded"
                        if finalized.overridden
                        else "tool_answer_grounded"
                    )
                    if finalized.groundedness is not None:
                        metric.groundedness_score = finalized.groundedness.score
                        metric.groundedness_reason = finalized.groundedness.reason
                        metric.groundedness_unsupported_count = len(
                            finalized.groundedness.unsupported
                        )
                        metric.groundedness_error = finalized.groundedness.error
            metric.response_time_ms = (time.time() - response_start) * 1000
            metric.response_length = len(response)
            if main_token_usage:
                metric.prompt_tokens = main_token_usage.get("prompt_tokens", 0)
                metric.completion_tokens = main_token_usage.get("completion_tokens", 0)
                metric.total_tokens = main_token_usage.get("total_tokens", 0)
                metric.estimated_cost_usd = main_token_usage.get(
                    "estimated_cost_usd", 0.0
                )
            from rag.citations import referenced_labels

            metric.citation_coverage = 1.0 if (
                not answered_with_tools
                and rag_result.provenance
                and referenced_labels(response)
            ) else 0.0
            try:
                from rag.metrics import get_metrics_manager

                await get_metrics_manager().record(metric)
            except Exception:
                pass
            self.context_manager.set_last_query_id("slash-command", metric.query_id)

        self.context_manager.track_message(
            channel_id="slash-command",
            message_id=f"slash-{id(user_message)}",
            author_id="user",
            author_name="User",
            content=user_message,
        )
        self.context_manager.track_message(
            channel_id="slash-command",
            message_id=f"slash-resp-{id(response)}",
            author_id="bot",
            author_name="NomNom",
            content=response,
            is_bot=True,
        )
        return response

    async def send_message(self, message, user_message: str) -> None:
        """Legacy send_message for slash commands."""
        if hasattr(message, "user"):
            author = message.user.id
        else:
            author = message.author.id

        try:
            response = await self.handle_response(user_message)
            response_content = f"> **{user_message}** - <@{author}>\n\n{response}"
            await send_split_message(self, response_content, message)
        except Exception as e:
            logger.exception(f"Error sending: {e}")
            error_msg = self._safe_error_message()
            if hasattr(message, "followup"):
                await message.followup.send(error_msg)
            else:
                await message.channel.send(error_msg)

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def reset_conversation_history(self, channel_id: Optional[str] = None) -> None:
        """Reset conversation context."""
        if channel_id:
            self.context_manager.clear_context(channel_id)
        else:
            self.context_manager.clear_all()

    async def switch_persona(self, persona: str, user_id: Optional[str] = None) -> None:
        """Switch persona — resets context."""
        self.reset_conversation_history()
        personas.current_persona = persona

    def get_current_provider_info(self) -> Dict:
        """Get info about current provider/model."""
        return {
            "provider": os.getenv("LLM_PROVIDER", "not_configured"),
            "current_model": self.current_model,
            "available_models": [],
            "supports_images": False,
        }

    def switch_provider(self, provider_type, model: Optional[str] = None) -> None:
        """Legacy provider switching."""
        if self.provider_manager:
            self.provider_manager.set_current_provider(provider_type)
        if model:
            self.current_model = model


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
discordClient = NomNomClient()
