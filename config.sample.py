# Sample configuration file
# Copy this to config.py and fill in your actual values

# Discord Bot Token - Get this from Discord Developer Portal
DISCORD_TOKEN = "your_discord_bot_token_here"

# System prompt key for fallback responses
FALLBACK_SYSTEM_PROMPT = "default"

# Add any other configuration variables your LLM provider needs
# For example:
# OPENAI_API_KEY = "your_openai_api_key_here"
# MODEL_NAME = "gpt-3.5-turbo"

# Embedding provider for vector search / hybrid RAG.
# Supported values: "openai", "openai_compatible", "gemini"
# EMBEDDING_PROVIDER = "openai_compatible"
# EMBEDDING_MODEL = "BAAI/bge-m3"
# Use http://vllm-embeddings:8000/v1 inside docker compose.
# Use http://localhost:8001/v1 from the host machine.
# EMBEDDING_BASE_URL = "http://vllm-embeddings:8000/v1"
# EMBEDDING_API_KEY = "local-dev-key"
# EMBEDDING_DIMENSION = 1024
# EMBEDDING_OPENAI_COMPAT_RAW_TEXT = true  # Required for BGE-M3/vLLM.
# EMBEDDING_COLLECTION_VERSION = "v2"
# KB_COLLECTION = "knowledge_base_v2"  # Promote only after benchmark review.

# Managed scale-out profile (leave disabled until a provider/key/budget is approved):
# EMBEDDING_PROVIDER = "openai"
# EMBEDDING_MODEL = "text-embedding-3-small"
# EMBEDDING_DIMENSION = 1024
# EMBEDDING_BASE_URL = "https://api.openai.com/v1"
# EMBEDDING_API_KEY = "<approved secret>"
