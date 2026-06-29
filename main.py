#!/usr/bin/env python3
"""
NomNom Discord Bot — Main entry point.

Validates environment, initialises subsystems, and launches the bot.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from src.bot import run_discord_bot
from src.log import logger

load_dotenv()


def validate_environment() -> bool:
    """Validate required environment variables and report available services."""
    # Required
    required_vars = ["DISCORD_BOT_TOKEN"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Please check your .env file (see .env.example)")
        return False

    # LLM provider
    llm_provider = os.getenv("LLM_PROVIDER", "not configured")
    llm_model = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "vllm-local"))
    llm_url = os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", ""))
    logger.info(f"LLM: provider={llm_provider}, model={llm_model}, base_url={llm_url or 'default'}")

    # RAG
    if os.getenv("ENABLE_RAG", "true").lower() == "true":
        pg_url = os.getenv("POSTGRES_URL", "")
        if pg_url:
            is_supabase = "supabase" in pg_url
            logger.info(f"RAG: enabled ({'Supabase' if is_supabase else 'local'} PostgreSQL)")
        else:
            logger.warning("RAG: enabled but POSTGRES_URL not set — will use default")
    else:
        logger.info("RAG: disabled")

    # Legacy providers (optional)
    providers = []
    if os.getenv("OPENAI_KEY"):
        providers.append("OpenAI")
    if os.getenv("CLAUDE_KEY"):
        providers.append("Claude")
    if os.getenv("GEMINI_KEY"):
        providers.append("Gemini")
    if os.getenv("GROK_KEY"):
        providers.append("Grok")
    if providers:
        logger.info(f"Legacy providers: {', '.join(providers)}")

    return True


def main():
    """Main entry point."""
    logger.info("Starting NomNom Discord Bot …")

    if not validate_environment():
        sys.exit(1)

    try:
        run_discord_bot()
    except KeyboardInterrupt:
        logger.info("Shutting down …")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise


if __name__ == "__main__":
    main()
