"""
Query Preprocessor (A2) — cleans and enriches user queries
before they are sent to the RAG retrieval pipeline.

Features:
- Remove Discord formatting (@mentions, emojis, URLs)
- Expand common abbreviations (Vietnamese + English + gaming)
- Detect language (vi/en)
- Add domain context if a knowledge domain is detected
"""

import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abbreviation/slang expansion maps
# ---------------------------------------------------------------------------
ABBREVIATIONS = {
    # Vietnamese common
    "ko": "không",
    "k": "không",
    "dc": "được",
    "đc": "được",
    "ng": "người",
    "ntn": "như thế nào",
    "sao": "tại sao",
    "j": "gì",
    "gì v": "gì vậy",
    "đi": "đi",
    "r": "rồi",
    "oy": "rồi",
    "bt": "biết",
    "bik": "biết",
    "lm": "làm",
    "lm sao": "làm sao",
    "tl": "trả lời",
    "đang": "đang",
    "nma": "nhưng mà",
    "vs": "với",
    "mk": "mình",
    "ik": "đi",
    "cx": "cũng",
    "mn": "mọi người",
    "ae": "anh em",
    # Gaming common
    "gg": "good game",
    "wp": "well played",
    "afk": "away from keyboard",
    "npc": "non-player character",
    "hp": "health points",
    "xp": "experience points",
    "dmg": "damage",
    "dps": "damage per second",
    "op": "overpowered",
    "nerf": "nerf (giảm sức mạnh)",
    "buff": "buff (tăng sức mạnh)",
    "pz": "Project Zomboid",
    "mc": "Minecraft",
}

# Discord formatting patterns
DISCORD_MENTION_PATTERN = re.compile(r"<@!?\d+>")
DISCORD_CHANNEL_PATTERN = re.compile(r"<#\d+>")
DISCORD_ROLE_PATTERN = re.compile(r"<@&\d+>")
DISCORD_EMOJI_PATTERN = re.compile(r"<a?:\w+:\d+>")
URL_PATTERN = re.compile(r"https?://\S+")


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------
class QueryPreprocessor:
    """
    Preprocesses user queries before RAG retrieval.
    Cleans Discord formatting, expands abbreviations,
    and enriches queries with domain context.
    """

    def __init__(self):
        self._abbreviations = ABBREVIATIONS.copy()

    def preprocess(
        self,
        raw_query: str,
        channel_name: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """
        Full preprocessing pipeline.

        Args:
            raw_query: Raw user message text.
            channel_name: Discord channel name for domain hinting.

        Returns:
            Tuple of (cleaned_query, metadata_dict)
            metadata_dict contains: language, domain, original_length, etc.
        """
        metadata = {
            "original_length": len(raw_query),
            "language": "unknown",
            "domain": None,
            "was_cleaned": False,
        }

        # Step 1: Remove Discord formatting
        cleaned = self._remove_discord_formatting(raw_query)
        if cleaned != raw_query:
            metadata["was_cleaned"] = True

        # Step 2: Basic cleanup
        cleaned = cleaned.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)  # collapse whitespace

        # Step 3: Detect language
        metadata["language"] = self._detect_language(cleaned)

        # Step 4: Expand abbreviations
        cleaned = self._expand_abbreviations(cleaned)

        # Step 5: Detect domain from keywords
        metadata["domain"] = self._detect_domain(cleaned, channel_name)

        # Step 6: Add domain context prefix (if domain detected)
        if metadata["domain"]:
            cleaned = self._add_domain_context(cleaned, metadata["domain"])

        logger.debug(
            f"Query preprocessed: '{raw_query[:60]}' → '{cleaned[:60]}' "
            f"(lang={metadata['language']}, domain={metadata['domain']})"
        )

        return cleaned, metadata

    def _remove_discord_formatting(self, text: str) -> str:
        """Remove Discord-specific formatting from text."""
        text = DISCORD_MENTION_PATTERN.sub("", text)
        text = DISCORD_CHANNEL_PATTERN.sub("", text)
        text = DISCORD_ROLE_PATTERN.sub("", text)
        text = DISCORD_EMOJI_PATTERN.sub("", text)
        text = URL_PATTERN.sub("[link]", text)
        return text

    def _detect_language(self, text: str) -> str:
        """Simple language detection based on Vietnamese diacritics."""
        # Vietnamese diacritics: ắằẳẵặấầẩẫậéèẻẽẹêếềểễệ...
        viet_chars = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")
        viet_count = sum(1 for c in text.lower() if c in viet_chars)
        if viet_count >= 2 or (viet_count >= 1 and len(text) < 20):
            return "vi"
        return "en"

    def _expand_abbreviations(self, text: str) -> str:
        """Expand known abbreviations/slang."""
        words = text.split()
        expanded = []
        for word in words:
            lower = word.lower().strip(".,!?")
            if lower in self._abbreviations:
                expanded.append(self._abbreviations[lower])
            else:
                expanded.append(word)
        return " ".join(expanded)

    def _detect_domain(
        self, text: str, channel_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Detect the knowledge domain from keywords in text or channel name.
        Returns domain key or None.
        """
        text_lower = text.lower()
        channel = (channel_name or "").lower()

        # Domain keyword maps
        domain_keywords = {
            "pz": [
                "project zomboid", "zomboid", "pz", "zombie",
                "carpentry", "foraging", "trapping",
                "moodle", "trait", "occupation", "knox",
                "crafting", "metalwork", "tailoring",
                "first aid", "farming", "fishing",
                "barricade", "generator", "vehicle",
            ],
        }

        for domain, keywords in domain_keywords.items():
            # Check channel name first
            if any(kw in channel for kw in keywords):
                return domain
            # Check message content
            if any(kw in text_lower for kw in keywords):
                return domain

        return None

    def _add_domain_context(self, text: str, domain: str) -> str:
        """Add domain-specific context prefix to improve retrieval."""
        prefixes = {
            "pz": "Project Zomboid:",
        }
        prefix = prefixes.get(domain, "")
        if prefix and not text.lower().startswith(prefix.lower()):
            return f"{prefix} {text}"
        return text

    def add_abbreviation(self, abbr: str, expansion: str) -> None:
        """Add a custom abbreviation mapping."""
        self._abbreviations[abbr.lower()] = expansion


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_preprocessor: Optional[QueryPreprocessor] = None


def get_preprocessor() -> QueryPreprocessor:
    """Return the singleton QueryPreprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = QueryPreprocessor()
    return _preprocessor
