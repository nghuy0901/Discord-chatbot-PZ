"""
Output sanitisation filters.

NomNom runs on a Chinese-origin model (Qwen) that occasionally emits stray
CJK characters (e.g. "全部") despite the system-prompt language rule. The prompt
alone cannot guarantee compliance, so this module is a deterministic safety net:
it strips CJK/Japanese/fullwidth characters from the model's output before it
reaches Discord, honouring the prompt's absolute "no Chinese" contract.

Vietnamese (Latin + diacritics) and emoji (astral plane) are untouched.
"""

import logging
import re

logger = logging.getLogger(__name__)

# CJK / Japanese / fullwidth ranges that must never appear in NomNom's
# Vietnamese/English output. Emoji live above the BMP and are not matched.
_CJK_PATTERN = re.compile(
    "["
    "　-〿"  # CJK symbols & punctuation 、。「」 + ideographic space
    "぀-ヿ"  # Hiragana + Katakana (Japanese)
    "㐀-䶿"  # CJK Unified Ideographs Extension A
    "一-鿿"  # CJK Unified Ideographs (Han)
    "豈-﫿"  # CJK Compatibility Ideographs
    "＀-￯"  # Halfwidth & Fullwidth Forms （）！？ １２３ etc.
    "]+"
)

# Tidy up the whitespace/punctuation left behind once characters are removed.
_MULTISPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:…)\]])")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


def contains_cjk(text: str) -> bool:
    """Return True if the text contains any CJK/Japanese/fullwidth character."""
    return bool(text) and _CJK_PATTERN.search(text) is not None


def strip_chinese(text: str) -> str:
    """Remove CJK/Japanese/fullwidth characters and tidy the surrounding text.

    Returns the input unchanged when it contains no such characters, so normal
    Vietnamese/English replies pass through untouched.
    """
    if not text or not _CJK_PATTERN.search(text):
        return text

    cleaned = _CJK_PATTERN.sub("", text)
    cleaned = _MULTISPACE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = _TRAILING_WS.sub("", cleaned)
    logger.warning(
        "Stripped CJK characters from model output (language-purity safety net)."
    )
    return cleaned
