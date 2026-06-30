"""Tests for the CJK output safety net (utils/text_filters.py)."""

from utils.text_filters import strip_chinese, contains_cjk


def test_strips_chinese_word_midsentence():
    src = "Sát thương không quyết định全部 — tốc độ quan trọng hơn"
    out = strip_chinese(src)
    assert not contains_cjk(out)
    assert "全" not in out and "部" not in out
    assert "Sát thương không quyết định" in out
    assert "tốc độ quan trọng hơn" in out
    # No doubled-space artifact where the word was removed.
    assert "  " not in out


def test_strips_japanese_and_fullwidth():
    assert not contains_cjk(strip_chinese("テスト１２３ですね"))


def test_preserves_vietnamese_and_punctuation():
    src = "Rìu là vũ khí mạnh nhất, đúng không?"
    assert strip_chinese(src) == src


def test_preserves_emoji():
    src = "Chào cậu 😎🔪✅"
    assert strip_chinese(src) == src


def test_noop_on_clean_text_returns_same_object():
    src = "Hello, world!"
    assert strip_chinese(src) is src


def test_handles_empty_and_none():
    assert strip_chinese("") == ""
    assert strip_chinese(None) is None
