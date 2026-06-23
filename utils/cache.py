"""
Redis caching layer for RAG queries and LLM responses.
Provides fast in-memory query retrieval and graceful fallback if Redis is down.
"""

import os
import logging
import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

# Redis configurations
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_ENABLED = os.getenv("REDIS_CACHE_ENABLED", "true").lower() == "true"
DEFAULT_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))  # 1 hour

_cache_instance = None


@dataclass(frozen=True)
class CacheVersion:
    kb_version: str
    prompt_version: str
    llm_model: str
    embedding_model: str
    retrieval_config_version: str


def _current_kb_version() -> str:
    try:
        from knowledge.manager import get_knowledge_manager

        kb_version = getattr(get_knowledge_manager(), "kb_version", "")
        if kb_version:
            return kb_version
    except Exception:
        pass
    return os.getenv("KB_VERSION", "v1")


def _current_prompt_version() -> str:
    try:
        from src.observability.prompts import current_prompt_cache_version

        return current_prompt_cache_version()
    except Exception:
        return os.getenv("PROMPT_VERSION", "v1")


def _current_llm_model() -> str:
    provider = os.getenv("LLM_PROVIDER", "")
    model = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", ""))
    return f"{provider}:{model}" if provider else model


def _current_embedding_model() -> str:
    try:
        from rag.embedding_registry import current_embedding_signature

        signature = current_embedding_signature()
        parts = [
            signature.provider,
            signature.model,
            str(signature.dimension),
            signature.collection_version,
        ]
        return ":".join(part for part in parts if part)
    except Exception:
        provider = os.getenv("EMBEDDING_PROVIDER", "")
        model = os.getenv("EMBEDDING_MODEL", "")
        return f"{provider}:{model}" if provider else model


def default_cache_version(
    *,
    kb_version: Optional[str] = None,
    prompt_version: Optional[str] = None,
    llm_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    retrieval_config_version: Optional[str] = None,
) -> CacheVersion:
    return CacheVersion(
        kb_version=kb_version or _current_kb_version(),
        prompt_version=prompt_version or _current_prompt_version(),
        llm_model=llm_model or _current_llm_model(),
        embedding_model=embedding_model or _current_embedding_model(),
        retrieval_config_version=(
            retrieval_config_version
            or os.getenv("RETRIEVAL_CONFIG_VERSION", "v1")
        ),
    )


class QueryCache:
    """Caching helper using Redis with automatic serialization and graceful bypass."""

    def __init__(self, url: str = REDIS_URL, default_ttl: int = DEFAULT_TTL):
        self.url = url
        self.default_ttl = default_ttl
        self.redis = None
        self.is_connected = False

        if CACHE_ENABLED:
            self._init_redis()

    @property
    def _redis(self):
        return self.redis

    def _init_redis(self) -> None:
        """Initialize the async redis client."""
        try:
            import redis.asyncio as redis
            self.redis = redis.from_url(
                self.url,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
                retry_on_timeout=True
            )
            self.is_connected = True
            logger.info(f"🔌 Redis client initialised for url={self.url}")
        except ImportError:
            logger.warning("⚠️ Redis python package not installed. Caching is disabled.")
            self.is_connected = False
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to Redis: {e}. Caching is disabled.")
            self.is_connected = False

    def _make_key(
        self,
        query: str,
        domain: Optional[str],
        version: Optional[CacheVersion] = None,
        request_context: Optional[Any] = None,
    ) -> str:
        """Generate a versioned SHA-256 cache key."""
        version = version or getattr(request_context, "cache_version", None)
        version = version or default_cache_version()
        normalized = "|".join([
            query.strip().lower(),
            domain or "all",
            version.kb_version,
            version.prompt_version,
            version.llm_model,
            version.embedding_model,
            version.retrieval_config_version,
        ])
        query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"rag:cache:v2:{version.kb_version}:{query_hash}"

    def make_key_for_test(self, query: str, domain: str, version: CacheVersion) -> str:
        return self._make_key(query, domain, version)

    async def get(
        self,
        query: str,
        domain: Optional[str] = None,
        version: Optional[CacheVersion] = None,
        request_context: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a cached item. Returns None on cache miss or error."""
        if not CACHE_ENABLED or not self.is_connected or not self.redis:
            return None

        key = self._make_key(query, domain, version, request_context)
        try:
            data = await self.redis.get(key)
            if data:
                logger.info(f"🚀 Cache HIT for query='{query[:40]}...' in domain='{domain}'")
                return json.loads(data)
        except Exception as e:
            logger.warning(f"⚠️ Redis GET error: {e}")
            # Try to reconnect on failure
            self.is_connected = False

        return None

    async def set(
        self,
        query: str,
        domain: Optional[str],
        result: Dict[str, Any],
        ttl: Optional[int] = None,
        version: Optional[CacheVersion] = None,
        request_context: Optional[Any] = None,
    ) -> bool:
        """Cache an item. Returns True on success, False on error."""
        if not CACHE_ENABLED or not self.is_connected or not self.redis:
            return False

        key = self._make_key(query, domain, version, request_context)
        expire = ttl if ttl is not None else self.default_ttl
        try:
            serialized = json.dumps(result, ensure_ascii=False)
            await self.redis.set(key, serialized, ex=expire)
            logger.debug(f"💾 Cached result for key={key} (TTL={expire}s)")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis SET error: {e}")
            self.is_connected = False
            return False

    async def invalidate_domain(
        self,
        domain: str,
        kb_version: Optional[str] = None,
    ) -> int:
        """Invalidate all cached queries belonging to a specific domain."""
        if not CACHE_ENABLED or not self.redis:
            return 0

        count = 0
        try:
            # Check connection status first
            if not self.is_connected:
                self._init_redis()
                if not self.is_connected:
                    return 0

            # Keys intentionally keep domain inside the hash, so invalidation
            # removes the old KB namespace rather than a single domain prefix.
            pattern = f"rag:cache:v2:{kb_version}:*" if kb_version else "rag:cache:v2:*"
            async for key in self.redis.scan_iter(pattern):
                await self.redis.delete(key)
                count += 1

            if count > 0:
                logger.info(f"🔄 Invalidated {count} cached items for domain '{domain}'")
        except Exception as e:
            logger.warning(f"⚠️ Redis invalidation error for domain '{domain}': {e}")
            self.is_connected = False

        return count

    async def flush_all(self) -> bool:
        """Clear the entire Redis database."""
        if not CACHE_ENABLED or not self.redis:
            return False
        try:
            await self.redis.flushdb()
            logger.info("🔄 Flushed entire Redis cache database.")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis flushdb error: {e}")
            self.is_connected = False
            return False


def get_query_cache() -> QueryCache:
    """Get the singleton QueryCache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryCache()
    return _cache_instance
