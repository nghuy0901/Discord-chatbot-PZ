from pathlib import Path


def test_legacy_llamaindex_ollama_receives_auth_headers():
    source = Path("llm_provider.py").read_text(encoding="utf-8")

    assert "build_ollama_headers" in source
    assert "headers = build_ollama_headers" in source
    assert "headers = headers" in source
