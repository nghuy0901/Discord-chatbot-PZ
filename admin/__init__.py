"""
Admin Module — Agentic features for Discord bot NomNom.

Features:
- Schedule management (đặt lịch, xem lịch, hủy lịch)
- Announcements (thông báo kênh)
- Background task scheduler

Only accessible by admin users defined in ADMIN_USER_IDS.
Uses Supabase REST API for persistence.
"""

from admin.handler import AdminAgentHandler
from admin.background import start_admin_background_tasks

__all__ = ["AdminAgentHandler", "start_admin_background_tasks"]
