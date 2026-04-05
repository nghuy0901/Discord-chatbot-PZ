"""
CLCT Discord Bot — command registration and event handlers.

Trigger logic:
- @mention → respond (or handle KB admin command via C3)
- Reply to bot → respond
- Active thread → respond
- /chat slash command → respond
- (optional) Implicit similarity reply → respond
- Reactions 👍/👎 → feedback tracking (B2)
"""

import os
import asyncio
import discord
from discord import app_commands
from typing import Optional

from src.aclient import discordClient
from src.log import logger
from src import personas


def run_discord_bot():

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    @discordClient.event
    async def on_ready():
        await discordClient.send_start_prompt()
        await discordClient.tree.sync()
        loop = asyncio.get_event_loop()
        loop.create_task(discordClient.process_messages())
        logger.info(f"✅ {discordClient.user} is now running! (CLCT mode)")

    # ------------------------------------------------------------------
    # B2: Reaction feedback events
    # ------------------------------------------------------------------

    @discordClient.event
    async def on_reaction_add(reaction: discord.Reaction, user: discord.User):
        """Handle reaction adds for feedback tracking (B2)."""
        await discordClient.handle_reaction_feedback(reaction, user)

    # ------------------------------------------------------------------
    # Core on_message — Grok-style trigger system + C3 KB commands
    # ------------------------------------------------------------------

    @discordClient.event
    async def on_message(message: discord.Message):
        # Ignore own messages
        if message.author == discordClient.user:
            return

        # Ignore empty messages
        if not message.content or not message.content.strip():
            return

        channel_id = str(message.channel.id)

        # Always track every message in short-term context
        reply_to_id = None
        if message.reference and message.reference.message_id:
            reply_to_id = str(message.reference.message_id)

        discordClient.context_manager.track_message(
            channel_id=channel_id,
            message_id=str(message.id),
            author_id=str(message.author.id),
            author_name=str(message.author.display_name),
            content=message.content,
            is_bot=False,
            reply_to_id=reply_to_id,
        )

        # Check if we should respond
        should_respond = discordClient.should_respond(message)

        # Debug: log trigger decision
        logger.debug(
            f"🔎 Trigger check: author={message.author.display_name}, "
            f"content='{message.content[:60]}', should_respond={should_respond}"
        )

        # Optional: check implicit reply (expensive, only if enabled)
        if not should_respond:
            should_respond = await discordClient.should_respond_implicit(message)

        if not should_respond:
            return

        # Clean the message content (remove the bot mention if present)
        user_message = message.content
        if discordClient.user:
            user_message = user_message.replace(
                f"<@{discordClient.user.id}>", ""
            ).strip()
            user_message = user_message.replace(
                f"<@!{discordClient.user.id}>", ""
            ).strip()

        if not user_message:
            user_message = "(mentioned without a message)"

        # C3: Check for knowledge base admin commands BEFORE normal response
        kb_command = discordClient._detect_kb_command(user_message)
        if kb_command:
            handled = await discordClient._handle_kb_command(message, kb_command)
            if handled:
                return

        username = str(message.author.display_name)
        logger.info(
            f"💬 Trigger: {username} in #{message.channel} — "
            f"{user_message[:80]}{'…' if len(user_message) > 80 else ''}"
        )

        discordClient.current_channel = message.channel
        await discordClient._generate_and_send(message, user_message)

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    @discordClient.tree.command(name="chat", description="Chat with CLCT")
    async def chat(interaction: discord.Interaction, *, message: str):
        if len(message) > 2000:
            await interaction.response.send_message(
                "❌ Message too long (max 2000 characters)", ephemeral=True
            )
            return

        message_text = message.replace("\x00", "").strip()
        if not message_text:
            await interaction.response.send_message(
                "❌ Please provide a message", ephemeral=True
            )
            return

        if interaction.user == discordClient.user:
            return

        username = str(interaction.user.display_name)
        discordClient.current_channel = interaction.channel
        logger.info(f"💬 /chat from {username}: {message_text[:80]}")

        await discordClient.enqueue_message(interaction, message_text)

    @discordClient.tree.command(name="reset", description="Clear conversation context for this channel")
    async def reset(interaction: discord.Interaction):
        channel_id = str(interaction.channel_id)
        discordClient.reset_conversation_history(channel_id)
        await interaction.response.send_message(
            "🔄 Conversation context cleared for this channel.", ephemeral=False
        )

    @discordClient.tree.command(name="resetall", description="Clear ALL conversation contexts (admin)")
    async def resetall(interaction: discord.Interaction):
        discordClient.reset_conversation_history()
        await interaction.response.send_message(
            "🔄 All conversation contexts cleared.", ephemeral=False
        )

    @discordClient.tree.command(name="status", description="Show CLCT status")
    async def status(interaction: discord.Interaction):
        """Enhanced status with A3 metrics and D2 knowledge base info."""
        from src.ollama_provider import health_check, OLLAMA_MODEL

        ollama_ok = await health_check()
        ctx_stats = discordClient.context_manager.stats()

        # RAG status
        rag_status = "disabled"
        try:
            from rag.db import get_message_count
            count = await get_message_count()
            rag_status = f"✅ {count:,} messages indexed"
        except Exception:
            rag_status = "⚠️ unavailable"

        # D2: Knowledge base status
        kb_status = "disabled"
        if os.getenv("ENABLE_KNOWLEDGE_BASE", "true").lower() == "true":
            try:
                from knowledge.manager import get_knowledge_manager
                kb = get_knowledge_manager()
                kb_info = kb.get_status()
                kb_status = (
                    f"✅ {kb_info['total_domains']} domains, "
                    f"{kb_info['total_docs']} docs, "
                    f"{kb_info['total_chunks']} chunks"
                )
            except Exception:
                kb_status = "⚠️ unavailable"

        # A3: Metrics summary
        metrics_info = ""
        try:
            from rag.metrics import get_metrics_manager
            metrics = get_metrics_manager()
            summary = metrics.get_summary()
            if summary["total_queries"] > 0:
                metrics_info = (
                    f"📊 Queries: {summary['total_queries']} total "
                    f"({summary['recent_queries']} recent)\n"
                    f"⏱️ Avg retrieval: {summary['avg_retrieval_time_ms']:.0f}ms\n"
                    f"🎯 Avg similarity: {summary['avg_similarity']:.2%}\n"
                    f"👍 {summary['feedback_positive']} / "
                    f"👎 {summary['feedback_negative']}"
                )
        except Exception:
            pass

        embed = discord.Embed(
            title="🤖 CLCT Status",
            color=discord.Color.green() if ollama_ok else discord.Color.red(),
        )
        embed.add_field(
            name="Ollama",
            value=f"{'✅ Online' if ollama_ok else '❌ Offline'}",
            inline=True,
        )
        embed.add_field(name="Model", value=OLLAMA_MODEL, inline=True)
        embed.add_field(name="RAG", value=rag_status, inline=True)
        embed.add_field(name="Knowledge Base", value=kb_status, inline=True)
        embed.add_field(
            name="Context",
            value=f"{ctx_stats['active_channels']} channels, {ctx_stats['total_messages']} messages",
            inline=True,
        )
        if metrics_info:
            embed.add_field(
                name="Metrics (A3)",
                value=metrics_info,
                inline=False,
            )
        await interaction.response.send_message(embed=embed, ephemeral=False)

    @discordClient.tree.command(name="provider", description="Switch AI provider and model")
    async def provider(interaction: discord.Interaction):
        """Interactive provider and model selection (legacy compatibility)."""
        if not discordClient.provider_manager:
            await interaction.response.send_message(
                "ℹ️ CLCT uses Ollama locally. Use `/status` to check model info.",
                ephemeral=True,
            )
            return

        from src.providers import ProviderType

        class ProviderSelect(discord.ui.Select):
            def __init__(self):
                options = []
                available_providers = discordClient.provider_manager.get_available_providers()
                for provider_type in available_providers:
                    emoji_map = {
                        ProviderType.FREE: "🆓",
                        ProviderType.OPENAI: "🟢",
                        ProviderType.CLAUDE: "🟣",
                        ProviderType.GEMINI: "🔵",
                        ProviderType.GROK: "⚫",
                    }
                    options.append(
                        discord.SelectOption(
                            label=provider_type.value.capitalize(),
                            value=provider_type.value,
                            emoji=emoji_map.get(provider_type, "🤖"),
                            default=(provider_type == discordClient.provider_manager.current_provider),
                        )
                    )
                super().__init__(placeholder="Select a provider...", options=options, min_values=1, max_values=1)

            async def callback(self, interaction: discord.Interaction):
                selected = ProviderType(self.values[0])
                discordClient.switch_provider(selected)
                await interaction.response.send_message(
                    f"✅ Switched to **{selected.value}** provider", ephemeral=True
                )

        view = discord.ui.View()
        view.add_item(ProviderSelect())
        info = discordClient.get_current_provider_info()
        embed = discord.Embed(
            title="🤖 AI Provider Settings",
            description=f"**Current:** {info['provider']} / {info['current_model']}",
            color=discord.Color.blue(),
        )
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    @discordClient.tree.command(name="switchpersona", description="Switch AI personality")
    async def switchpersona(interaction: discord.Interaction, persona: str):
        user_id = str(interaction.user.id)
        try:
            available_personas = personas.get_available_personas(user_id)
            if persona not in available_personas:
                await interaction.response.send_message(
                    f"❌ Invalid persona. Available: {', '.join(available_personas)}",
                    ephemeral=True,
                )
                return

            if personas.is_jailbreak_persona(persona):
                try:
                    personas.get_persona_prompt(persona, user_id)
                except PermissionError:
                    await interaction.response.send_message(
                        f"❌ You don't have permission for the '{persona}' persona.",
                        ephemeral=True,
                    )
                    return
                await interaction.response.send_message(
                    f"⚠️ **WARNING**: '{persona}' is a jailbreak persona. Use at your own risk.",
                    ephemeral=False,
                )
                logger.warning(f"User {user_id} activated jailbreak persona: {persona}")
            else:
                await interaction.response.defer(ephemeral=False)

            await discordClient.switch_persona(persona, user_id)
            message = f"🎭 Switched to **{persona}** persona"
            if hasattr(interaction, "followup"):
                await interaction.followup.send(message)
            else:
                await interaction.channel.send(message)
        except Exception as e:
            logger.error(f"Error switching persona: {e}")
            await interaction.response.send_message(
                f"❌ Failed to switch persona: {str(e)}", ephemeral=True
            )

    @discordClient.tree.command(name="private", description="Toggle private responses")
    async def private(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=False)
        discordClient.isPrivate = not discordClient.isPrivate
        mode = "private" if discordClient.isPrivate else "public"
        await interaction.followup.send(f"> **INFO: Switched to {mode} mode.**")

    @discordClient.tree.command(name="replyall", description="Toggle replyAll mode")
    async def replyall(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=False)
        discordClient.is_replying_all = not discordClient.is_replying_all
        state = "ON" if discordClient.is_replying_all else "OFF"
        await interaction.followup.send(f"> **INFO: ReplyAll mode is now {state}.**")

    @discordClient.tree.command(name="help", description="Show all available commands")
    async def help(interaction: discord.Interaction):
        embed = discord.Embed(
            title="🤖 CLCT — Help",
            description=(
                "I respond when you **@mention** me or **reply** to my messages.\n"
                "I also participate in threads where I've been active."
            ),
            color=discord.Color.blue(),
        )

        commands = [
            ("💬 **Chat**", [
                ("@CLCT [message]", "Mention me to chat"),
                ("Reply to my message", "Continue the conversation"),
                ("/chat [message]", "Slash command chat"),
                ("/reset", "Clear this channel's context"),
                ("/resetall", "Clear ALL contexts (admin)"),
            ]),
            ("📊 **Info**", [
                ("/status", "Show bot status, RAG stats & metrics"),
                ("/help", "Show this help"),
            ]),
            ("🎭 **Personas**", [
                ("/switchpersona [name]", "Change personality"),
            ]),
            ("⚙️ **Settings**", [
                ("/private", "Toggle private responses"),
                ("/replyall", "Toggle reply-all mode"),
                ("/provider", "Switch AI provider (legacy)"),
            ]),
            ("📚 **Knowledge Base** (Admin)", [
                ("@Bot reload knowledge", "Reload all knowledge domains"),
                ("@Bot scan knowledge", "View knowledge base status"),
                ("@Bot kb status", "View knowledge base status"),
            ]),
            ("👍 **Feedback**", [
                ("React 👍/👎", "Rate bot responses for quality tracking"),
            ]),
        ]

        for category, cmds in commands:
            value = "\n".join([f"`{cmd}` — {desc}" for cmd, desc in cmds])
            embed.add_field(name=category, value=value, inline=False)

        await interaction.response.send_message(embed=embed, ephemeral=False)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    discordClient.run(os.getenv("DISCORD_BOT_TOKEN"))