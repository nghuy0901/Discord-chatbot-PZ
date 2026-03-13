"""
Domain Router (D2) — Routes user queries to the correct knowledge domain
and provides combined retrieval from both knowledge base and chat history.

Determines which knowledge domains are relevant, searches them,
and merges results with chat history for a comprehensive RAG context.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

from knowledge.manager import get_knowledge_manager
from rag.query_preprocessor import get_preprocessor

logger = logging.getLogger(__name__)

# Minimum similarity to include KB results
KB_MIN_SIMILARITY: float = 0.35


# ---------------------------------------------------------------------------
# Domain Router
# ---------------------------------------------------------------------------
class DomainRouter:
    """
    Routes queries to relevant knowledge domains and merges
    knowledge base results with chat history retrieval results.
    """

    def __init__(self):
        self._kb_manager = None

    @property
    def kb_manager(self):
        if self._kb_manager is None:
            self._kb_manager = get_knowledge_manager()
        return self._kb_manager

    def route(
        self,
        query: str,
        detected_domain: Optional[str] = None,
        channel_name: Optional[str] = None,
    ) -> List[str]:
        """
        Determine which knowledge domains to search.

        Args:
            query: Preprocessed user query.
            detected_domain: Domain detected by query preprocessor.
            channel_name: Discord channel name for hints.

        Returns:
            List of domain names to search (can be empty).
        """
        domains = []

        # 1. Use preprocessor-detected domain
        if detected_domain and detected_domain in self.kb_manager.domains:
            domains.append(detected_domain)

        # 2. If no specific domain detected, search all loaded domains
        if not domains and self.kb_manager.domains:
            domains = list(self.kb_manager.domains.keys())

        return domains

    def search_knowledge(
        self,
        query: str,
        domains: Optional[List[str]] = None,
        top_k: int = 5,
        threshold: float = KB_MIN_SIMILARITY,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Search knowledge base across specified domains.

        Args:
            query: Search query.
            domains: List of domains to search (None = all).
            top_k: Max results per domain.
            threshold: Minimum similarity score.

        Returns:
            Tuple of (results_list, primary_domain_matched)
        """
        all_results = []
        primary_domain = None

        if not domains:
            # Search all domains
            results = self.kb_manager.search(
                query=query,
                domain=None,
                k=top_k,
                score_threshold=threshold,
            )
            all_results.extend(results)
        else:
            for domain in domains:
                results = self.kb_manager.search(
                    query=query,
                    domain=domain,
                    k=top_k,
                    score_threshold=threshold,
                )
                all_results.extend(results)

        # Sort by similarity descending
        all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        # Determine primary domain from top result
        if all_results:
            primary_domain = all_results[0].get("domain")

        return all_results[:top_k], primary_domain

    def get_domain_prompt(self, domain: Optional[str]) -> str:
        """Get the domain-specific prompt addition."""
        if not domain:
            return ""
        return self.kb_manager.get_domain_prompt(domain)


# ---------------------------------------------------------------------------
# Format knowledge results for prompt
# ---------------------------------------------------------------------------
def format_kb_results_for_prompt(
    results: List[Dict[str, Any]],
    max_chars: int = 2000,
) -> str:
    """
    Format knowledge base results into a compact block for prompt injection.
    """
    if not results:
        return ""

    lines = ["[Retrieved Knowledge Base — Relevant static documents]"]
    total_chars = 0

    for i, r in enumerate(results, 1):
        source = r.get("source", "unknown")
        domain = r.get("domain", "")
        similarity = r.get("similarity", 0.0)
        content = r.get("content", "")

        # Truncate very long chunks
        if len(content) > 600:
            content = content[:597] + "…"

        citation = (
            f"[KB-{i}] ({similarity:.0%} match) [{domain}] {source}\n"
            f"    {content}"
        )

        if total_chars + len(citation) > max_chars:
            lines.append(f"... ({len(results) - i + 1} more KB results omitted)")
            break

        lines.append(citation)
        total_chars += len(citation)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_router: Optional[DomainRouter] = None


def get_domain_router() -> DomainRouter:
    """Return the singleton DomainRouter."""
    global _router
    if _router is None:
        _router = DomainRouter()
    return _router
