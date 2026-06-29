"""Fail-fast startup configuration validation (audit H9 / M10).

A misconfigured deployment should refuse to boot with a clear error, instead of
starting "fine" and then failing on every request (and leaking the misconfig in
the error text). Both entry points call :func:`validate_runtime_config`:
- the Discord bot in ``src.bot.run_discord_bot`` (hard fail), and
- the FastAPI app in its startup event (hard fail in production, warn in dev).
"""

import os
import logging

logger = logging.getLogger(__name__)

# Placeholder secrets that ship in .env.example and must never reach production.
_INSECURE_LLM_KEYS = {"", "local-dev-key", "local", "changeme", "default"}


def is_production() -> bool:
    return os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "")).strip().lower() in {
        "prod",
        "production",
    }


def validate_runtime_config(
    *, require_discord: bool = False, require_api: bool = False
) -> None:
    """Validate required configuration; raise ``RuntimeError`` on any problem."""
    errors = []

    provider = (os.getenv("LLM_PROVIDER") or "").lower().strip()
    try:
        # Canonical provider/credential validation (raises on missing/invalid).
        from src.llm.factory import build_llm_config

        build_llm_config()
    except Exception as e:
        errors.append(str(e))

    if not os.getenv("POSTGRES_URL"):
        errors.append("POSTGRES_URL is required")

    if require_discord and not os.getenv("DISCORD_BOT_TOKEN"):
        errors.append("DISCORD_BOT_TOKEN is required to start the Discord bot")

    if require_api:
        try:
            from api.auth import require_configured_api_key

            require_configured_api_key()
        except Exception as e:
            errors.append(str(e))

    # Production only: refuse to boot with shipped placeholder credentials (M10).
    if is_production():
        if "postgres:postgres@" in os.getenv("POSTGRES_URL", ""):
            errors.append("POSTGRES_URL uses the default postgres:postgres credentials")
        if provider in {"openai", "gemini"} and os.getenv("LLM_API_KEY", "") in _INSECURE_LLM_KEYS:
            errors.append("LLM_API_KEY is a placeholder; a real key is required in production")
        if (
            os.getenv("EMBEDDING_PROVIDER", "").lower() in {"openai", "gemini"}
            and os.getenv("EMBEDDING_API_KEY", "") in _INSECURE_LLM_KEYS
        ):
            errors.append("EMBEDDING_API_KEY is a placeholder; a real key is required in production")

    if errors:
        raise RuntimeError(
            "Startup configuration is invalid (refusing to boot):\n  - "
            + "\n  - ".join(errors)
        )
    logger.info("Startup configuration validated (production=%s).", is_production())
