import pytest

from api.auth import require_configured_api_key


def test_missing_api_key_config_raises(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API_KEY is required"):
        require_configured_api_key()


def test_default_api_key_is_rejected(monkeypatch):
    monkeypatch.setenv("API_KEY", "PZ-Default-Key-123")
    with pytest.raises(RuntimeError, match="default API_KEY"):
        require_configured_api_key()
