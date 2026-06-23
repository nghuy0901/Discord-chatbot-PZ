import hashlib
import os


def prompt_version(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]


def current_prompt_cache_version() -> str:
    explicit = os.getenv("PROMPT_VERSION")
    if explicit:
        return explicit

    try:
        from prompts.system_prompt import _load_template

        return prompt_version(_load_template())
    except Exception:
        return "v1"
