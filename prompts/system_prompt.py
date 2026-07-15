"""
System Prompt Builder — Loads, assembles, and caches the system prompt
from modular template sections stored as .txt files in prompts/templates/.

Each section is a standalone .txt file that can be edited independently
without touching Python code. Sections are concatenated in SECTION_ORDER
to form the final template.

Placeholders supported:
  {current_date}   — injected at build time (UTC timestamp)
  {domain_prompt}  — domain-specific instructions from knowledge/prompts/
  {rag_context}    — retrieved RAG context block
"""

import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Section load order — defines the structure of the final prompt.
# To add a new section: create a .txt file in templates/ and add its
# stem name here in the desired position.
SECTION_ORDER: List[str] = [
    "base_identity",
    "conversation_rules",
    "identity_rules",
    "game_domain_rules",
    "rag_instructions",
    "citation_rules",
    "formatting_rules",
]

_cached_template: Optional[str] = None


def _load_template() -> str:
    """
    Load and concatenate all template sections from .txt files.
    Result is cached after first call; use reload_templates() to refresh.
    """
    global _cached_template
    if _cached_template is not None:
        return _cached_template

    sections: List[str] = []
    for section_name in SECTION_ORDER:
        path = TEMPLATES_DIR / f"{section_name}.txt"
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                sections.append(content)
                logger.debug(f"Loaded prompt section: {section_name} ({len(content)} chars)")
        else:
            logger.warning(f"Prompt section file not found: {path}")

    # Combine sections, then append dynamic placeholders
    _cached_template = "\n\n".join(sections) + "\n\n{domain_prompt}\n\n{rag_context}"

    logger.info(
        f"System prompt template assembled: {len(SECTION_ORDER)} sections, "
        f"{len(_cached_template)} chars total"
    )
    return _cached_template


def build_system_prompt(
    rag_context: str = "",
    domain_prompt: str = "",
) -> str:
    """
    Build the final system prompt with runtime values.

    Args:
        rag_context: Formatted RAG retrieval results block.
        domain_prompt: Domain-specific instructions (from knowledge/prompts/).

    Returns:
        The fully formatted system prompt string ready for LLM consumption.
    """
    template = _load_template()
    prompt = template.format(
        current_date=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        rag_context=rag_context,
        domain_prompt=domain_prompt,
    )
    if rag_context:
        prompt += (
            "\n\n# Required evidence-backed answer form\n"
            "- Answer the exact question in at most 3 short bullets.\n"
            "- Every factual bullet must end with one or more valid [n] source markers.\n"
            "- Omit any claim that is not explicit in its cited source.\n"
            "- Do not infer comparisons, recommendations, negative claims, or missing effects.\n"
            "- Do not add an introduction, conclusion, tip, or follow-up offer."
        )
    return prompt


def reload_templates() -> None:
    """
    Force reload all template sections from disk.
    Useful for hot-reloading prompts without restarting the bot.
    """
    global _cached_template
    _cached_template = None
    logger.info("System prompt template cache cleared — will reload on next build.")


def get_section_names() -> List[str]:
    """Return the ordered list of section names (for introspection/debugging)."""
    return list(SECTION_ORDER)
