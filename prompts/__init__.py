"""
Prompts package — modular system prompt management.

Provides build_system_prompt() to assemble the final system prompt
from individual template sections stored as .txt files.
"""

from prompts.system_prompt import build_system_prompt, reload_templates

__all__ = ["build_system_prompt", "reload_templates"]
