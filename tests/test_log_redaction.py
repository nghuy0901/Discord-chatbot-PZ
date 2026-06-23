from src.log import redact_sensitive


def test_redacts_sensitive_env_values(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-api-key-123")
    assert "secret-api-key-123" not in redact_sensitive("key=secret-api-key-123")


def test_redacts_bearer_tokens():
    assert redact_sensitive("Authorization: Bearer abc.def.ghi") == "Authorization: Bearer [redacted]"
