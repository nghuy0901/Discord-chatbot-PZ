"""
Intent Parser — Detects admin intents from Vietnamese natural language.

Two-stage parsing:
1. Keyword heuristic (fast path) → detect intent type
2. LLM structured extraction → parse details (time, recurrence, etc.)
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent types
# ---------------------------------------------------------------------------
INTENT_SCHEDULE = "schedule_meeting"
INTENT_LIST = "list_schedules"
INTENT_CANCEL = "cancel_schedule"
INTENT_ANNOUNCE = "announcement"
INTENT_UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# Keyword detection (Stage 1 — fast path)
# ---------------------------------------------------------------------------
ADMIN_KEYWORDS: Dict[str, str] = {
    # Schedule
    "đặt lịch": INTENT_SCHEDULE,
    "hẹn lịch": INTENT_SCHEDULE,
    "lịch họp": INTENT_SCHEDULE,
    "nhắc nhở": INTENT_SCHEDULE,
    "tạo lịch": INTENT_SCHEDULE,
    "lên lịch": INTENT_SCHEDULE,
    "đặt hẹn": INTENT_SCHEDULE,
    "nhắc tôi": INTENT_SCHEDULE,
    "nhắc lúc": INTENT_SCHEDULE,
    # List
    "xem lịch": INTENT_LIST,
    "danh sách lịch": INTENT_LIST,
    "lịch sắp tới": INTENT_LIST,
    "có lịch gì": INTENT_LIST,
    # Cancel
    "hủy lịch": INTENT_CANCEL,
    "xóa lịch": INTENT_CANCEL,
    "bỏ lịch": INTENT_CANCEL,
    # Announce
    "thông báo": INTENT_ANNOUNCE,
    "announce": INTENT_ANNOUNCE,
}

# Admin command trigger phrases — these distinguish admin intent from normal chat
ADMIN_TRIGGERS = [
    "đặt lịch", "hẹn lịch", "lịch họp", "nhắc nhở", "tạo lịch",
    "lên lịch", "xem lịch", "hủy lịch", "xóa lịch", "thông báo",
    "danh sách lịch", "đặt hẹn", "nhắc tôi", "nhắc lúc", "bỏ lịch",
    "lịch sắp tới", "có lịch gì", "announce",
]


@dataclass
class ParsedIntent:
    """Result of intent parsing."""
    intent: str = INTENT_UNKNOWN
    description: str = ""
    time: Optional[str] = None          # e.g. "10:00"
    date: Optional[str] = None          # e.g. "2026-04-20"
    recurrence: Optional[str] = None    # "once", "daily", "weekly", "monthly"
    day_of_week: Optional[str] = None   # "monday"..."sunday"
    channel_id: Optional[str] = None    # target channel
    schedule_id: Optional[str] = None   # for cancel
    raw_message: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "description": self.description,
            "time": self.time,
            "date": self.date,
            "recurrence": self.recurrence,
            "day_of_week": self.day_of_week,
            "channel_id": self.channel_id,
            "schedule_id": self.schedule_id,
            "confidence": self.confidence,
        }


def detect_admin_intent(message: str) -> Optional[str]:
    """
    Stage 1: Fast keyword-based intent detection.

    Returns intent type string or None if not an admin command.
    """
    lower = message.lower().strip()
    for keyword, intent in ADMIN_KEYWORDS.items():
        if keyword in lower:
            return intent
    return None


def is_admin_command(message: str) -> bool:
    """Check if a message looks like an admin command."""
    lower = message.lower().strip()
    return any(trigger in lower for trigger in ADMIN_TRIGGERS)


async def parse_intent_with_llm(
    message: str,
    intent_hint: str,
) -> ParsedIntent:
    """
    Stage 2: Use LLM to extract structured details from the message.

    Args:
        message: The admin's natural language message.
        intent_hint: The intent detected by keyword matching.

    Returns:
        ParsedIntent with extracted fields.
    """
    prompt_path = os.path.join(
        os.path.dirname(__file__), "prompts", "schedule_parser.txt"
    )

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        system_prompt = _DEFAULT_PARSER_PROMPT

    from datetime import datetime
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(os.getenv("ADMIN_TIMEZONE", "Asia/Ho_Chi_Minh"))
    current_time_str = datetime.now(tz=tz).strftime("%Y-%m-%d %H:%M:%S")

    # Build the LLM request
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Thời điểm hiện tại: {current_time_str}\nIntent hint: {intent_hint}\nMessage: {message}"},
    ]

    try:
        from src.ollama_provider import chat_completion
        response = await chat_completion(
            messages=messages,
            temperature=0.1,  # Low temperature for structured output
        )

        # Try to extract JSON from response
        parsed = _extract_json(response)
        if parsed:
            return ParsedIntent(
                intent=parsed.get("intent", intent_hint),
                description=parsed.get("description", ""),
                time=parsed.get("time"),
                date=parsed.get("date"),
                recurrence=parsed.get("recurrence", "once"),
                day_of_week=parsed.get("day_of_week"),
                channel_id=parsed.get("channel_id"),
                schedule_id=parsed.get("schedule_id"),
                raw_message=message,
                confidence=0.9,
            )
        else:
            logger.warning(f"LLM response not valid JSON: {response[:200]}")
            return ParsedIntent(
                intent=intent_hint,
                description=message,
                raw_message=message,
                confidence=0.5,
            )

    except Exception as e:
        logger.error(f"LLM intent parsing failed: {e}")
        return ParsedIntent(
            intent=intent_hint,
            description=message,
            raw_message=message,
            confidence=0.3,
        )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM response text."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    import re
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } in text
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# Fallback prompt if file not found
_DEFAULT_PARSER_PROMPT = """Bạn là một parser chuyên trích xuất thông tin lịch hẹn từ tin nhắn tiếng Việt.

Trả về JSON với các trường sau:
{
  "intent": "schedule_meeting" | "list_schedules" | "cancel_schedule" | "announcement",
  "description": "mô tả ngắn gọn",
  "time": "HH:MM" (24h format),
  "date": "YYYY-MM-DD" (nếu có),
  "recurrence": "once" | "daily" | "weekly" | "monthly",
  "day_of_week": "monday" | "tuesday" | ... | "sunday" (nếu weekly),
  "channel_id": null,
  "schedule_id": null
}

Chỉ trả về JSON, không giải thích gì thêm."""
