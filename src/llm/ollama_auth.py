from typing import Dict, Optional
from urllib.parse import urlparse


def build_ollama_headers(api_key: Optional[str]) -> Dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def validate_ollama_auth(base_url: Optional[str], api_key: Optional[str]) -> None:
    hostname = urlparse(base_url or "").hostname or ""
    if hostname == "ollama.com" or hostname.endswith(".ollama.com"):
        if not api_key:
            raise RuntimeError(
                "OLLAMA_API_KEY (or LLM_API_KEY) is required for Ollama Cloud"
            )
