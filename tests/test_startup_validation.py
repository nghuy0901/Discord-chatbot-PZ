"""Fail-fast startup config validation (audit H9/M10)."""

import pytest

from src.startup import validate_runtime_config


def _base_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_MODEL", "vllm-local")
    monkeypatch.setenv("LLM_BASE_URL", "http://vllm:8000/v1")
    monkeypatch.setenv("POSTGRES_URL", "postgresql://u:p@db:5432/x")
    monkeypatch.setenv("APP_ENV", "development")


def test_valid_dev_config_passes(monkeypatch):
    _base_env(monkeypatch)
    validate_runtime_config()  # no raise


def test_missing_llm_provider_raises(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(RuntimeError):
        validate_runtime_config()


def test_missing_postgres_raises(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    with pytest.raises(RuntimeError):
        validate_runtime_config()


def test_missing_discord_token_raises(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        validate_runtime_config(require_discord=True)


def test_production_rejects_default_postgres(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv(
        "POSTGRES_URL", "postgresql://postgres:postgres@postgres:5432/postgres"
    )
    with pytest.raises(RuntimeError):
        validate_runtime_config()


def test_production_rejects_placeholder_openai_key(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "local-dev-key")
    monkeypatch.setenv("POSTGRES_URL", "postgresql://u:realpass@db:5432/x")
    with pytest.raises(RuntimeError):
        validate_runtime_config()
