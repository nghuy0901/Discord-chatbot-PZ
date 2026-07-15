"""Docker wiring for the local vLLM embedding service."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compose_defines_vllm_embedding_service():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "vllm-embeddings:" in compose
    assert "vllm/vllm-openai" in compose
    assert "--model" in compose
    assert "BAAI/bge-m3" in compose
    assert "BgeM3EmbeddingModel" in compose
    assert "--gpu-memory-utilization" in compose
    assert "${VLLM_GPU_MEMORY_UTILIZATION:-0.70}" in compose
    assert "--convert" in compose
    assert "embed" in compose
    assert "--pooler-config" in compose
    assert '"task":"embed"' in compose


def test_env_example_points_embeddings_at_compose_service():
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "EMBEDDING_PROVIDER=openai_compatible" in env_example
    assert "EMBEDDING_MODEL=BAAI/bge-m3" in env_example
    assert "EMBEDDING_BASE_URL=http://vllm-embeddings:8000/v1" in env_example
    assert "VLLM_GPU_MEMORY_UTILIZATION=0.70" in env_example
