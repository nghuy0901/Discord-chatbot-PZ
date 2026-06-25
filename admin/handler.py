"""
AdminAgentHandler — Main entry point for admin agentic features.

Orchestrates:
1. Admin permission check
2. Intent detection (keyword → LLM parsing)
3. Action execution (schedule CRUD, announcements)
4. Response generation via LLM

Designed for minimal integration footprint:
- bot.py calls `await admin_handler.handle(message)` before normal response
- Returns True if handled (admin command), False if not
"""

import os
import logging
from typing import Optional

import discord

from admin.intent_parser import (
    detect_admin_intent,
    is_admin_command,
    parse_intent_with_llm,
    ParsedIntent,
    INTENT_SCHEDULE,
    INTENT_LIST,
    INTENT_CANCEL,
    INTENT_ANNOUNCE,
    INTENT_UNKNOWN,
)
from admin.scheduler import ScheduleManager

logger = logging.getLogger(__name__)

# Admin user IDs
ADMIN_USER_IDS: set = set(
    os.getenv("ADMIN_USER_IDS", "").split(",")
) - {""}


class AdminAgentHandler:
    """
    Admin agent that processes admin commands as natural language.

    Usage:
        handler = AdminAgentHandler()
        handled = await handler.handle(message, cleaned_content)
        if handled:
            return  # was an admin command, don't process further
    """

    def __init__(self):
        self.scheduler = ScheduleManager()

    def is_admin(self, user_id: str) -> bool:
        """Check if the user is an admin."""
        return user_id in ADMIN_USER_IDS

    async def handle(
        self,
        message: discord.Message,
        content: str,
    ) -> bool:
        """
        Handle an admin command from a message.

        Args:
            message: Discord message object.
            content: Cleaned message content (without @mention).

        Returns:
            True if the message was handled as an admin command.
            False if not an admin command (let normal flow handle it).
        """
        user_id = str(message.author.id)

        # 1. Admin check
        if not self.is_admin(user_id):
            return False

        # 2. Detect admin intent (fast keyword check)
        intent = detect_admin_intent(content)
        if intent is None:
            # Check with broader trigger detection
            if not is_admin_command(content):
                return False
            intent = INTENT_UNKNOWN

        logger.info(
            f"🛡️ Admin command detected: intent={intent}, "
            f"user={message.author.display_name}"
        )

        # 3. Show typing while processing
        async with message.channel.typing():
            try:
                # Parse details with LLM
                parsed = await parse_intent_with_llm(content, intent)

                # 4. Execute based on intent
                response = await self._execute_intent(message, parsed)

                # 5. Send response
                if response:
                    await message.reply(response, mention_author=False)
                return True

            except Exception as e:
                logger.error(f"Admin handler error: {e}", exc_info=True)
                await message.reply(
                    f"❌ Lỗi xử lý lệnh admin: {str(e)[:300]}",
                    mention_author=False,
                )
                return True

    async def _execute_intent(
        self,
        message: discord.Message,
        parsed: ParsedIntent,
    ) -> str:
        """
        Execute the parsed intent and return a response string.

        Args:
            message: Discord message for context.
            parsed: Parsed intent with extracted details.

        Returns:
            Response string to send back to the admin.
        """
        if parsed.intent == INTENT_SCHEDULE:
            return await self._handle_schedule(message, parsed)
        elif parsed.intent == INTENT_LIST:
            return await self._handle_list(message, parsed)
        elif parsed.intent == INTENT_CANCEL:
            return await self._handle_cancel(message, parsed)
        elif parsed.intent == INTENT_ANNOUNCE:
            return await self._handle_announce(message, parsed)
        else:
            return (
                "🤔 Mình hiểu bạn muốn dùng tính năng admin, "
                "nhưng chưa rõ yêu cầu cụ thể.\n\n"
                "**Các lệnh admin hỗ trợ:**\n"
                "• `đặt lịch <mô tả> lúc <giờ>` — Tạo nhắc nhở\n"
                "• `xem lịch` — Xem danh sách lịch\n"
                "• `hủy lịch <id>` — Hủy lịch\n"
                "• `thông báo <nội dung>` — Gửi thông báo"
            )

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    async def _handle_schedule(
        self, message: discord.Message, parsed: ParsedIntent
    ) -> str:
        """Handle schedule creation."""
        guild_id = str(message.guild.id) if message.guild else "0"
        channel_id = parsed.channel_id or str(message.channel.id)
        creator_id = str(message.author.id)

        description = parsed.description or parsed.raw_message
        if not description or description == parsed.raw_message:
            # Use the raw message as description, cleaned
            description = parsed.raw_message

        try:
            schedule = await self.scheduler.create_schedule(
                guild_id=guild_id,
                channel_id=channel_id,
                creator_id=creator_id,
                description=description,
                time_str=parsed.time,
                date_str=parsed.date,
                recurrence=parsed.recurrence or "once",
                day_of_week=parsed.day_of_week,
            )

            # Format confirmation
            cron = schedule.get("cron_expression", "?")
            readable = ScheduleManager.format_cron_readable(cron)
            next_run = schedule.get("next_run", "?")

            # Format next_run
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(next_run)
                next_run_display = dt.strftime("%d/%m/%Y %H:%M")
            except (ValueError, TypeError):
                next_run_display = str(next_run)

            return (
                f"✅ **Đã tạo lịch thành công!**\n\n"
                f"📋 **Nội dung**: {description}\n"
                f"⏰ **Lịch trình**: {readable}\n"
                f"📅 **Lần tiếp theo**: {next_run_display}\n"
                f"🆔 **ID**: `{schedule.get('id', '?')[:8]}`\n"
                f"📍 **Kênh**: <#{channel_id}>"
            )

        except ValueError as e:
            return f"❌ Lỗi tạo lịch: {e}"
        except Exception as e:
            logger.error(f"Schedule creation failed: {e}", exc_info=True)
            return f"❌ Không thể tạo lịch: {str(e)[:200]}"

    async def _handle_list(
        self, message: discord.Message, parsed: ParsedIntent
    ) -> str:
        """Handle listing schedules."""
        guild_id = str(message.guild.id) if message.guild else "0"

        schedules = await self.scheduler.list_schedules(guild_id)

        if not schedules:
            return "📅 Hiện không có lịch hẹn nào đang hoạt động."

        lines = [f"📅 **Danh sách lịch hẹn** ({len(schedules)} lịch):\n"]
        for s in schedules:
            info = ScheduleManager.format_schedule_info(s)
            lines.append(f"• {info}")

        return "\n".join(lines)

    async def _handle_cancel(
        self, message: discord.Message, parsed: ParsedIntent
    ) -> str:
        """Handle schedule cancellation."""
        schedule_id = parsed.schedule_id

        if not schedule_id:
            # Try to find ID in the raw message
            import re
            id_match = re.search(r"[a-f0-9]{8}", parsed.raw_message.lower())
            if id_match:
                schedule_id = id_match.group(0)
            else:
                return (
                    "❌ Cần cung cấp ID lịch để hủy.\n"
                    "Dùng `xem lịch` để xem danh sách và ID."
                )

        # Find the full ID by prefix match
        guild_id = str(message.guild.id) if message.guild else "0"
        schedules = await self.scheduler.list_schedules(guild_id)

        matching = [
            s for s in schedules
            if s.get("id", "").startswith(schedule_id)
        ]

        if not matching:
            return f"❌ Không tìm thấy lịch với ID `{schedule_id}`"

        if len(matching) > 1:
            lines = ["⚠️ Tìm thấy nhiều lịch khớp với ID, hãy chỉ rõ hơn:\n"]
            for s in matching:
                lines.append(f"• {ScheduleManager.format_schedule_info(s)}")
            return "\n".join(lines)

        # Cancel the unique match
        full_id = matching[0].get("id")
        success = await self.scheduler.cancel_schedule(full_id)

        if success:
            desc = matching[0].get("description", "?")
            return f"✅ Đã hủy lịch: **{desc}** (`{full_id[:8]}`)"
        else:
            return f"❌ Không thể hủy lịch `{schedule_id}`"

    async def _handle_announce(
        self, message: discord.Message, parsed: ParsedIntent
    ) -> str:
        """Handle announcement (immediate send to channel)."""
        content = parsed.description or parsed.raw_message

        # Remove the trigger word "thông báo" from the content
        for trigger in ["thông báo", "announce"]:
            if content.lower().startswith(trigger):
                content = content[len(trigger):].strip()

        if not content:
            return "❌ Cần nội dung thông báo. Ví dụ: `thông báo Hôm nay nghỉ họp`"

        # Send announcement embed
        embed = discord.Embed(
            title="📢 Thông Báo",
            description=content,
            color=discord.Color.gold(),
        )
        embed.set_footer(
            text=f"— {message.author.display_name}",
        )

        try:
            await message.channel.send(embed=embed)
            return "✅ Đã gửi thông báo!"
        except Exception as e:
            return f"❌ Không thể gửi thông báo: {e}"
