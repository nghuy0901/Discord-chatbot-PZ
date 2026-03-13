"""
Knowledge Base Module — Multi-domain RAG-enhanced knowledge system.

Components:
- manager.py: Load, chunk, embed, and manage knowledge documents
- domain_router.py: Route queries to relevant knowledge domains
"""

from knowledge.manager import get_knowledge_manager, KnowledgeManager
from knowledge.domain_router import get_domain_router, DomainRouter

__all__ = [
    "get_knowledge_manager",
    "KnowledgeManager",
    "get_domain_router",
    "DomainRouter",
]
