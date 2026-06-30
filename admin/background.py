"""
Background Task — Scheduler loop that checks and triggers due schedules.

Runs as a background asyncio task, polling Supabase every 30 seconds
for schedules whose next_run <= NOW().

Handles:
- Normal on-time triggers
- Missed schedules (bot was offline)
- Crash recovery (all data in Supabase, survives restarts)
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import discord

from admin.scheduler import ScheduleManager
from admin import announcer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POLL_INTERVAL = 30  # seconds between schedule checks
TIMEZONE = ZoneInfo(os.getenv("ADMIN_TIMEZONE", "Asia/Ho_Chi_Minh"))


async def start_admin_background_tasks(client: discord.Client) -> None:
    """
    Start the admin background scheduler loop.
    
    Called from bot.py on_ready event.

    Args:
        client: The Discord client instance (needed to send messages).
    """
    logger.info("🔄 Starting admin background scheduler...")
    asyncio.create_task(_scheduler_loop(client))


async def _scheduler_loop(client: discord.Client) -> None:
    """
    Main scheduler loop — runs indefinitely.
    
    Every POLL_INTERVAL seconds:
    1. Query Supabase for due schedules (next_run <= NOW)
    2. Execute each due schedule (send notification)
    3. Advance next_run or deactivate (for one-time schedules)
    """
    scheduler = ScheduleManager()

    # Initial delay to let bot fully start up
    await asyncio.sleep(5)
    logger.info(f"✅ Background scheduler active (poll every {POLL_INTERVAL}s)")

    while True:
        try:
            due_schedules = await scheduler.get_due_schedules()

            if due_schedules:
                logger.info(f"⏰ {len(due_schedules)} schedule(s) due")

            for schedule in due_schedules:
                try:
                    await _handle_due_schedule(
                        client, scheduler, schedule
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to handle schedule "
                        f"{schedule.get('id', '?')}: {e}"
                    )

        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")

        await asyncio.sleep(POLL_INTERVAL)


async def _handle_due_schedule(
    client: discord.Client,
    scheduler: ScheduleManager,
    schedule: dict,
) -> None:
    """
    Handle a single due schedule:
    - Detect if missed (was offline)
    - Send appropriate notification
    - Advance to next run time
    """
    now = datetime.now(tz=TIMEZONE)
    next_run_str = schedule.get("next_run", "")
    schedule_id = schedule.get("id", "?")

    # Parse next_run
    try:
        next_run = datetime.fromisoformat(next_run_str)
        if next_run.tzinfo is None:
            next_run = next_run.replace(tzinfo=TIMEZONE)
    except (ValueError, TypeError):
        logger.warning(f"Invalid next_run for schedule {schedule_id}: {next_run_str}")
        next_run = now

    missed_duration = now - next_run

    # Determine notification type based on how late we are
    if missed_duration > timedelta(hours=24):
        # Bot was offline for a long time — send missed reminder
        note = f"⚠️ Trễ {missed_duration.days} ngày do bot offline"
        await announcer.send_missed_reminder(client, schedule, note=note)

    elif missed_duration > timedelta(minutes=5):
        # Slightly late — send with late note
        minutes_late = int(missed_duration.total_seconds() / 60)
        note = f"⚠️ Nhắc trễ {minutes_late} phút"
        await announcer.send_reminder(client, schedule, note=note)

    else:
        # On time — normal reminder
        await announcer.send_reminder(client, schedule)

    # Advance schedule to next run
    cron_expr = schedule.get("cron_expression", "")
    success = await scheduler.advance_schedule(schedule_id, cron_expr)

    if success:
        logger.info(f"✅ Schedule {schedule_id[:8]} advanced to next run")
    else:
        logger.warning(f"⚠️ Failed to advance schedule {schedule_id[:8]}")
