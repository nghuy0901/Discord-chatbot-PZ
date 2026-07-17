from pathlib import Path


def test_prompt_forbids_general_knowledge_fallback():
    text = Path("prompts/templates/rag_instructions.txt").read_text(
        encoding="utf-8"
    ).lower()
    assert "may use your general knowledge" not in text
    assert "do not answer from general knowledge" in text
    assert "ask a clarifying question" in text
    assert "approved retrieved evidence" in text


def test_prompt_allows_direct_certainty_translation_without_vague_numeric_inference():
    text = Path("prompts/templates/rag_instructions.txt").read_text(
        encoding="utf-8"
    ).lower()

    assert "guarantees" in text and "100%" in text
    assert "likely" in text and "do not" in text
