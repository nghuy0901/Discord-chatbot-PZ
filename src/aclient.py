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
from typing import List, Dict, Optional

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
                result = await kb.load_all()
                total = sum(result.values())
                logger.info(f"✅ Knowledge base loaded: {total} chunks across {len(result)} domains")
            except Exception as e:
                logger.warning(f"⚠️ Knowledge base init failed (non-fatal): {e}")

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

        # Get the query_id associated with this bot message
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

        try:
            # Build prompt with RAG + context (now passes channel_name for D2)
            # Phase 4: build_prompt now returns query_intent as 3rd element
            prompt_messages, temperature, query_intent = await self.context_manager.build_prompt(
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

            # Phase 4: Intent-based routing
            # Determine if tool calling should be attempted based on query intent
            tools_available = (
                ENABLE_TOOL_CALLING
                and STRUCTURED_TOOLS_AVAILABLE
                and PZ_TOOLS
            )

            # Route decision based on query_intent:
            #   ANALYTICAL → always use tools (no streaming — need full response for tool detection)
            #   HYBRID     → try tools first, fallback to streaming if no tools triggered
            #   NARRATIVE  → skip tools entirely, go straight to streaming
            #   CONVERSATION → skip tools entirely, go straight to streaming
            #   None       → hybrid behavior (backward compat)
            use_tools_for_query = tools_available and query_intent in (
                "analytical", "hybrid", None
            )

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

                    if response_text:
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
                    await self._send_response(message, response_text)

            # A3: Record response timing
            response_time = (time.time() - response_start) * 1000
            try:
                from rag.metrics import get_metrics_manager
                metrics = get_metrics_manager()
                # Update the most recent metric with response info
                if metrics._recent:
                    last_metric = metrics._recent[-1]
                    last_metric.response_time_ms = response_time
                    last_metric.response_length = len(response_text)
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
                except Exception as e:
                    logger.debug(f"Failed to add feedback reactions: {e}")

        except Exception as e:
            logger.exception(f"Response generation failed: {e}")
            error_msg = f"❌ Sorry, I ran into an issue: {str(e)[:200]}"
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
                await bot_msg.edit(content=f"❌ Streaming error: {str(e)[:300]}")
            except Exception:
                pass
            final_text = ""

        return final_text, bot_msg

    async def _send_response(
        self, message, content: str
    ) -> None:
        """Send a response, handling slash commands vs regular messages."""
        # Resolve @Username → real Discord mentions
        content = self._resolve_mentions(
            content, getattr(message, 'guild', None)
        )
        if hasattr(message, "followup"):
            await send_split_message(self, content, message)
        else:
            if len(content) > 2000:
                parts = [content[i : i + 1990] for i in range(0, len(content), 1990)]
                for part in parts:
                    await message.channel.send(part)
            else:
                await message.reply(content, mention_author=False)

    # ------------------------------------------------------------------
    # Slash command response (legacy compatibility)
    # ------------------------------------------------------------------

    async def handle_response(self, user_message: str) -> str:
        """Legacy handle_response for /chat command."""
        request_context = RequestContext.new(
            channel_id="slash-command",
            user_id="api_user",
            source="discord_slash",
        )
        prompt_messages, temperature, query_intent = await self.context_manager.build_prompt(
            channel_id="slash-command",
            user_message=user_message,
            user_name="User",
            enable_rag=ENABLE_RAG,
            request_context=request_context,
        )
        # Phase 4: Intent-based routing for slash commands
        use_tools = (
            ENABLE_TOOL_CALLING
            and STRUCTURED_TOOLS_AVAILABLE
            and query_intent in ("analytical", "hybrid", None)
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
            error_msg = f"❌ Error: {str(e)}"
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