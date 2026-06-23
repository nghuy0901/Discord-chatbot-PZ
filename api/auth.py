import os


DEFAULT_KEYS = {"PZ-Default-Key-123", "changeme", "default"}


def require_configured_api_key() -> str:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is required for NomNom API")
    if api_key in DEFAULT_KEYS:
        raise RuntimeError("Refusing to start with default API_KEY")
    if len(api_key) < 32:
        raise RuntimeError("API_KEY must be at least 32 characters")
    return api_key
