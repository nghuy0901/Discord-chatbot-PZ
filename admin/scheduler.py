"""
Scheduler — Schedule CRUD operations + cron expression management.

MCP-ready: Each public method is designed as a standalone async function
that can be wrapped as an MCP tool in the future.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo

from croniter import croniter

from admin.db import AdminDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timezone config
# ---------------------------------------------------------------------------
TIMEZONE = ZoneInfo(os.getenv("ADMIN_TIMEZONE", "Asia/Ho_Chi_Minh"))

# ---------------------------------------------------------------------------
# Day of week mapping (Vietnamese → cron)
# ---------------------------------------------------------------------------
DAY_MAP = {
    "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
    "friday": 5, "saturday": 6, "sunday": 0,
    # Vietnamese aliases
    "thứ hai": 1, "thứ 2": 1, "t2": 1,
    "thứ ba": 2, "thứ 3": 2, "t3": 2,
    "thứ tư": 3, "thứ 4": 3, "t4": 3,
    "thứ năm": 4, "thứ 5": 4, "t5": 4,
    "thứ sáu": 5, "thứ 6": 5, "t6": 5,
    "thứ bảy": 6, "thứ 7": 6, "t7": 6,
    "chủ nhật": 0, "cn": 0,
}


class ScheduleManager:
    """
    Manage scheduled reminders with cron expressions.
    
    Supports:
    - One-time schedules
    - Recurring: daily, weekly, monthly
    - Vietnamese day-of-week parsing
    """

    def __init__(self):
        self.db = AdminDB()

    # ---- Create ----

    async def create_schedule(
        self,
        guild_id: str,
        channel_id: str,
        creator_id: str,
        description: str,
        time_str: Optional[str] = None,
        date_str: Optional[str] = None,
        recurrence: str = "once",
        day_of_week: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new schedule from parsed intent data.

        Args:
            guild_id: Discord guild ID
            channel_id: Target channel for notification
            creator_id: Admin user ID who created this
            description: What the reminder is about
            time_str: Time in HH:MM format (24h)
            date_str: Date in YYYY-MM-DD format (for one-time)
            recurrence: "once", "daily", "weekly", "monthly"
            day_of_week: Day name (for weekly recurrence)

        Returns:
            Created schedule dict

        Raises:
            ValueError: If time/date parsing fails
        """
        # Parse time
        hour, minute = self._parse_time(time_str)

        # Build cron expression
        cron_expr = self._build_cron(
            hour=hour,
            minute=minute,
            recurrence=recurrence,
            day_of_week=day_of_week,
            date_str=date_str,
        )

        # Compute first next_run
        now = datetime.now(tz=TIMEZONE)
        next_run = self._compute_next_run(cron_expr, now)

        data = {
            "guild_id": guild_id,
            "channel_id": channel_id,
            "creator_id": creator_id,
            "description": description,
            "cron_expression": cron_expr,
            "next_run": next_run.isoformat(),
        }

        schedule = await self.db.create_schedule(data)
        logger.info(
            f"📅 Schedule created: '{description}' | "
            f"cron={cron_expr} | next_run={next_run}"
        )
        return schedule

    # ---- Read ----

    async def list_schedules(
        self, guild_id: str, active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """List all schedules for a guild."""
        return await self.db.list_schedules(guild_id, active_only)

    async def get_due_schedules(self) -> List[Dict[str, Any]]:
        """Get all schedules that are due for execution."""
        now = datetime.now(tz=TIMEZONE)
        return await self.db.get_due_schedules(now)

    # ---- Update ----

    async def advance_schedule(self, schedule_id: str, cron_expr: str) -> bool:
        """
        Advance a schedule to its next run time after execution.

        For one-time schedules (contains specific date), deactivates instead.
        """
        now = datetime.now(tz=TIMEZONE)

        # Check if this is a one-time schedule
        if self._is_one_time(cron_expr):
            return await self.db.cancel_schedule(schedule_id)

        next_run = self._compute_next_run_after(cron_expr, now)
        return await self.db.update_next_run(schedule_id, next_run)

    # ---- Delete ----

    async def cancel_schedule(self, schedule_id: str) -> bool:
        """Cancel (soft-delete) a schedule."""
        return await self.db.cancel_schedule(schedule_id)

    # ---- Cron helpers ----

    def _parse_time(self, time_str: Optional[str]) -> tuple:
        """Parse time string (HH:MM) into (hour, minute)."""
        if not time_str:
            # Default to current time + 1 hour
            now = datetime.now(tz=TIMEZONE)
            future = now + timedelta(hours=1)
            return future.hour, 0

        parts = time_str.replace("h", ":").replace("H", ":").split(":")
        parts = [p for p in parts if p.strip()]  # Remove empty parts (e.g. "10h" → "10:")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0

        if not (0 <= hour <= 23):
            raise ValueError(f"Invalid hour: {hour}")
        if not (0 <= minute <= 59):
            raise ValueError(f"Invalid minute: {minute}")

        return hour, minute

    def _build_cron(
        self,
        hour: int,
        minute: int,
        recurrence: str = "once",
        day_of_week: Optional[str] = None,
        date_str: Optional[str] = None,
    ) -> str:
        """
        Build a cron expression from parsed parameters.

        Format: minute hour day_of_month month day_of_week
        """
        if recurrence == "daily":
            return f"{minute} {hour} * * *"

        elif recurrence == "weekly":
            dow = 6  # Default Saturday
            if day_of_week:
                dow_lower = day_of_week.lower().strip()
                dow = DAY_MAP.get(dow_lower, 6)
            return f"{minute} {hour} * * {dow}"

        elif recurrence == "monthly":
            # Default: 1st of each month
            day = 1
            if date_str:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    day = dt.day
                except ValueError:
                    pass
            return f"{minute} {hour} {day} * *"

        else:  # "once"
            if date_str:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    return f"{minute} {hour} {dt.day} {dt.month} *"
                except ValueError:
                    pass
            # No date specified: schedule for today/tomorrow
            now = datetime.now(tz=TIMEZONE)
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return f"{minute} {hour} {target.day} {target.month} *"

    def _compute_next_run(
        self, cron_expr: str, after: datetime
    ) -> datetime:
        """Compute the next run time from a cron expression."""
        cron = croniter(cron_expr, after)
        next_dt = cron.get_next(datetime)
        # Ensure timezone
        if next_dt.tzinfo is None:
            next_dt = next_dt.replace(tzinfo=TIMEZONE)
        return next_dt

    def _compute_next_run_after(
        self, cron_expr: str, after: datetime
    ) -> datetime:
        """
        Compute the next run time STRICTLY after the given time.
        Used after executing a schedule to find the next occurrence.
        """
        return self._compute_next_run(cron_expr, after)

    def _is_one_time(self, cron_expr: str) -> bool:
        """
        Check if a cron expression is one-time (specific month+day, no wildcards).
        e.g. "0 10 15 4 *" (April 15th at 10:00) — one-time
        e.g. "0 10 * * *" (daily at 10:00) — recurring
        """
        parts = cron_expr.split()
        if len(parts) != 5:
            return False
        # One-time if both day_of_month AND month are specific (not *)
        return parts[2] != "*" and parts[3] != "*"

    # ---- Formatting helpers ----

    @staticmethod
    def format_schedule_info(schedule: Dict[str, Any]) -> str:
        """Format a schedule dict into a human-readable string."""
        desc = schedule.get("description", "No description")
        cron = schedule.get("cron_expression", "?")
        next_run = schedule.get("next_run", "?")
        sid = schedule.get("id", "?")[:8]

        # Parse next_run for display
        if isinstance(next_run, str):
            try:
                dt = datetime.fromisoformat(next_run)
                next_run = dt.strftime("%d/%m/%Y %H:%M")
            except ValueError:
                pass

        return f"`{sid}` — **{desc}** | Cron: `{cron}` | Next: {next_run}"

    @staticmethod
    def format_cron_readable(cron_expr: str) -> str:
        """Convert cron expression to Vietnamese readable text."""
        parts = cron_expr.split()
        if len(parts) != 5:
            return cron_expr

        minute, hour, dom, month, dow = parts

        time_str = f"{hour}:{minute.zfill(2)}"

        if dom == "*" and month == "*" and dow == "*":
            return f"Hàng ngày lúc {time_str}"
        elif dom == "*" and month == "*" and dow != "*":
            day_names = {
                "0": "Chủ Nhật", "1": "Thứ 2", "2": "Thứ 3",
                "3": "Thứ 4", "4": "Thứ 5", "5": "Thứ 6", "6": "Thứ 7",
            }
            day = day_names.get(dow, f"ngày {dow}")
            return f"Hàng tuần vào {day} lúc {time_str}"
        elif month == "*" and dow == "*":
            return f"Hàng tháng ngày {dom} lúc {time_str}"
        else:
            return f"Ngày {dom}/{month} lúc {time_str}"
