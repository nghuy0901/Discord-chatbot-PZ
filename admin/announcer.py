"""
Announcer — Send schedule notifications to Discord channels.

Handles:
- Normal reminders (on-time)
- Missed reminders (bot was offline)
- Rich Discord embeds with @admin tags
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

import discord

logger = logging.getLogger(__name__)


async def send_reminder(
    client: discord.Client,
    schedule: Dict[str, Any],
    note: Optional[str] = None,
) -> bool:
    """
    Send a schedule reminder to the target Discord channel.

    Args:
        client: Discord client instance.
        schedule: Schedule dict from database.
        note: Optional note to append (e.g. "trễ 5 phút").

    Returns:
        True if sent successfully.
    """
    channel_id = schedule.get("channel_id")
    if not channel_id:
        logger.warning(f"Schedule {schedule.get('id')} has no channel_id")
        return False

    try:
        channel = client.get_channel(int(channel_id))
        if channel is None:
            channel = await client.fetch_channel(int(channel_id))

        creator_id = schedule.get("creator_id")
        description = schedule.get("description", "Nhắc nhở")
        
        # Build conversational text via LLM
        mention = f"<@{creator_id}>" if creator_id else ""
        
        system_prompt = (
            "Bạn là trợ lý Discord năng động. Nhiệm vụ của bạn là viết một câu thông báo đến giờ hẹn/nhắc nhở "
            "thật tự nhiên, sinh động và có thể pha chút hài hước (vd: 'Gogogo đến giờ họp rồi mọi người ơi').\n\n"
            "YÊU CẦU BẮT BUỘC:\n"
            "1. Phải giữ NGHIÊM NGẶT đúng các tag (viết y chang dạng <@ID> hoặc @username) có trong nội dung.\n"
            "2. Viết liền thành một tin nhắn (chỉ dài 1-2 câu).\n"
            "3. KHÔNG phản hồi bất kỳ thứ gì ngoài nội dung tin nhắn đó."
        )
        
        user_prompt = f"Thông báo nội dung sau: '{description}'\nNhớ gọi tên (tag) người tạo: {mention}"

        try:
            from src.ollama_provider import chat_completion
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            generated_msg = await chat_completion(messages=messages, temperature=0.7)
            if not generated_msg or len(generated_msg.strip()) < 5:
                raise ValueError("LLM generated empty response")
                
            content = generated_msg.strip()
        except Exception as e:
            logger.warning(f"Failed to generate LLM announcement, using fallback: {e}")
            content = f"**Gogogo đến giờ rồi mọi người ơi!** {mention}\n **Nội dung:** {description}"

        if note:
            content += f"\n*{note}*"

        await channel.send(content=content)
        logger.info(
            f"✅ Reminder sent: '{schedule.get('description')}' → "
            f"#{channel.name if hasattr(channel, 'name') else channel_id}"
        )
        return True

    except discord.NotFound:
        logger.warning(f"Channel {channel_id} not found — skipping reminder")
        return False
    except discord.Forbidden:
        logger.warning(f"No permission to send to channel {channel_id}")
        return False
    except Exception as e:
        logger.error(f"Failed to send reminder: {e}")
        return False


async def send_missed_reminder(
    client: discord.Client,
    schedule: Dict[str, Any],
    note: str = "⚠️ Nhắc nhở bị trễ do bot offline",
) -> bool:
    """
    Send a missed/late reminder notification.

    Args:
        client: Discord client instance.
        schedule: Schedule dict from database.
        note: Note explaining the delay.

    Returns:
        True if sent successfully.
    """
    return await send_reminder(client, schedule, note=note)
