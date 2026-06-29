"""
Admin DB — Supabase REST client for admin_schedules table.

Uses supabase-py to interact with the admin_schedules table via REST API.
Service role key required for full CRUD access (bypasses RLS).
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Lazy-init Supabase client
_client = None


def _get_client():
    """Get or create Supabase client (lazy singleton)."""
    global _client
    if _client is None:
        try:
            from supabase import create_client
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_KEY")
            if not url or not key:
                raise ValueError(
                    "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env"
                )
            _client = create_client(url, key)
            logger.info("✅ Supabase client initialized for admin module.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            raise
    return _client


class AdminDB:
    """
    Supabase REST client for admin_schedules CRUD.

    MCP-ready: Each method is a standalone async-compatible function
    that can be easily wrapped as an MCP tool.
    """

    TABLE = "admin_schedules"

    # ---- Create ----

    async def create_schedule(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new schedule.

        Args:
            data: dict with keys: guild_id, channel_id, creator_id,
                  description, cron_expression, next_run

        Returns:
            The created schedule record.
        """
        client = _get_client()
        try:
            result = client.table(self.TABLE).insert(data).execute()
            if result.data:
                logger.info(f"📅 Schedule created: {result.data[0].get('id', '?')}")
                return result.data[0]
            raise ValueError("Insert returned no data")
        except Exception as e:
            logger.error(f"Failed to create schedule: {e}")
            raise

    # ---- Read ----

    async def get_due_schedules(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get all active schedules whose next_run <= now.

        Returns:
            List of schedule dicts that are due for execution.
        """
        client = _get_client()
        if now is None:
            now = datetime.now().astimezone()

        try:
            result = (
                client.table(self.TABLE)
                .select("*")
                .eq("is_active", True)
                .lte("next_run", now.isoformat())
                .order("next_run")
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get due schedules: {e}")
            return []

    async def list_schedules(
        self, guild_id: str, active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List all schedules for a guild.

        Args:
            guild_id: Discord guild ID.
            active_only: If True, only return active schedules.

        Returns:
            List of schedule dicts.
        """
        client = _get_client()
        try:
            query = (
                client.table(self.TABLE)
                .select("*")
                .eq("guild_id", guild_id)
                .order("next_run")
            )
            if active_only:
                query = query.eq("is_active", True)
            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to list schedules: {e}")
            return []

    async def get_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Get a single schedule by ID."""
        client = _get_client()
        try:
            result = (
                client.table(self.TABLE)
                .select("*")
                .eq("id", schedule_id)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get schedule {schedule_id}: {e}")
            return None

    # ---- Update ----

    async def update_next_run(
        self, schedule_id: str, next_run: datetime
    ) -> bool:
        """
        Advance a schedule's next_run after execution.

        Args:
            schedule_id: UUID of the schedule.
            next_run: New next_run datetime.

        Returns:
            True if update succeeded.
        """
        client = _get_client()
        try:
            result = (
                client.table(self.TABLE)
                .update({
                    "next_run": next_run.isoformat(),
                    "updated_at": datetime.now().astimezone().isoformat(),
                })
                .eq("id", schedule_id)
                .execute()
            )
            return bool(result.data)
        except Exception as e:
            logger.error(f"Failed to update next_run for {schedule_id}: {e}")
            return False

    # ---- Delete (soft) ----

    async def cancel_schedule(self, schedule_id: str) -> bool:
        """
        Cancel (soft-delete) a schedule by setting is_active = False.

        Args:
            schedule_id: UUID of the schedule to cancel.

        Returns:
            True if cancelled successfully.
        """
        client = _get_client()
        try:
            result = (
                client.table(self.TABLE)
                .update({
                    "is_active": False,
                    "updated_at": datetime.now().astimezone().isoformat(),
                })
                .eq("id", schedule_id)
                .execute()
            )
            if result.data:
                logger.info(f"🚫 Schedule cancelled: {schedule_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cancel schedule {schedule_id}: {e}")
            return False
