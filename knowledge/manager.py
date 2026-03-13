"""
Knowledge Manager (D2 + C2 + C3) — Load, chunk, embed, and manage
multi-domain static knowledge base documents.

Features:
- Multi-domain support (pz, server_rules, general, etc.)
- Sentence-based chunking with overlap (C2)
- Hot-reload via @mention commands (C3)
- Domain-specific prompts
- PGVector storage for similarity search
"""

import os
import re
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KNOWLEDGE_DIR = os.getenv(
    "KNOWLEDGE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs"),
)
PROMPTS_DIR = os.getenv(
    "KNOWLEDGE_PROMPTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts"),
)

# Chunking settings (C2)
CHUNK_SIZE: int = int(os.getenv("KB_CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("KB_CHUNK_OVERLAP", "50"))

# Knowledge collection name (separate from chat history)
KB_COLLECTION = os.getenv("KB_COLLECTION", "knowledge_base")

SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst"}


# ---------------------------------------------------------------------------
# Sentence-based text chunker (C2)
# ---------------------------------------------------------------------------
class SentenceChunker:
    """
    Split text into chunks based on sentences, with overlap.
    Prioritizes splitting at paragraph → sentence → word boundaries.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)

            # If single sentence exceeds chunk_size, force-split it
            if sentence_len > self.chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # Split long sentence by words
                words = sentence.split()
                word_chunk = []
                word_len = 0
                for word in words:
                    if word_len + len(word) + 1 > self.chunk_size and word_chunk:
                        chunks.append(" ".join(word_chunk))
                        # Overlap: keep last few words
                        overlap_words = []
                        ol = 0
                        for w in reversed(word_chunk):
                            if ol + len(w) + 1 > self.overlap:
                                break
                            overlap_words.insert(0, w)
                            ol += len(w) + 1
                        word_chunk = overlap_words
                        word_len = ol
                    word_chunk.append(word)
                    word_len += len(word) + 1
                if word_chunk:
                    current_chunk = word_chunk
                    current_len = word_len
                else:
                    current_chunk = []
                    current_len = 0
                continue

            # Normal case: add sentence if it fits
            if current_len + sentence_len + 1 > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Overlap: keep sentences from end of current chunk
                overlap_sents = []
                ol = 0
                for s in reversed(current_chunk):
                    if ol + len(s) + 1 > self.overlap:
                        break
                    overlap_sents.insert(0, s)
                    ol += len(s) + 1
                current_chunk = overlap_sents
                current_len = ol

            current_chunk.append(sentence)
            current_len += sentence_len + 1

        # Flush remaining
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Split on sentence boundaries: . ! ? followed by space or newline
        # Also split on double newlines (paragraphs)
        text = re.sub(r"\n{2,}", "\n\n", text)
        parts = re.split(r"\n\n", text)

        sentences = []
        for part in parts:
            # Split sentences within paragraphs
            sents = re.split(r"(?<=[.!?])\s+", part.strip())
            sentences.extend(sents)
            sentences.append("")  # paragraph break marker

        return [s for s in sentences if s]


# ---------------------------------------------------------------------------
# Domain info
# ---------------------------------------------------------------------------
class KnowledgeDomain:
    """Represents a knowledge domain with its documents and prompt."""

    def __init__(self, name: str, docs_path: str, prompt_path: Optional[str] = None):
        self.name = name
        self.docs_path = docs_path
        self.prompt_path = prompt_path
        self.prompt_text: str = ""
        self.doc_count: int = 0
        self.chunk_count: int = 0
        self.file_hashes: Dict[str, str] = {}  # filename → md5 hash
        self.last_loaded: float = 0

    def load_prompt(self) -> str:
        """Load domain-specific prompt from file."""
        if self.prompt_path and os.path.isfile(self.prompt_path):
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self.prompt_text = f.read().strip()
        return self.prompt_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "docs_path": self.docs_path,
            "doc_count": self.doc_count,
            "chunk_count": self.chunk_count,
            "last_loaded": self.last_loaded,
            "has_prompt": bool(self.prompt_text),
        }


# ---------------------------------------------------------------------------
# Knowledge Manager
# ---------------------------------------------------------------------------
class KnowledgeManager:
    """
    Manages multi-domain static knowledge base.
    Loads documents, chunks them, embeds into PGVector,
    and provides search interface.
    """

    def __init__(
        self,
        docs_dir: str = KNOWLEDGE_DIR,
        prompts_dir: str = PROMPTS_DIR,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.docs_dir = docs_dir
        self.prompts_dir = prompts_dir
        self.chunker = SentenceChunker(chunk_size, chunk_overlap)
        self.domains: Dict[str, KnowledgeDomain] = {}
        self._vectorstore = None
        self._initialized = False

    # ----- Initialization -----

    def _get_vectorstore(self):
        """Get or create the knowledge base PGVector store (separate collection)."""
        if self._vectorstore is None:
            from rag.db import get_embeddings, PGVECTOR_CONNECTION
            from langchain_community.vectorstores import PGVector

            self._vectorstore = PGVector(
                collection_name=KB_COLLECTION,
                connection=PGVECTOR_CONNECTION,
                embeddings=get_embeddings(),
                use_jsonb=True,
            )
            logger.info(f"Knowledge base vectorstore initialized: collection={KB_COLLECTION}")
        return self._vectorstore

    def discover_domains(self) -> List[str]:
        """Discover available domains by scanning the docs directory."""
        domains = []
        if not os.path.isdir(self.docs_dir):
            logger.warning(f"Knowledge docs directory not found: {self.docs_dir}")
            return domains

        for entry in os.scandir(self.docs_dir):
            if entry.is_dir() and not entry.name.startswith("."):
                domains.append(entry.name)

        return domains

    async def load_all(self) -> Dict[str, int]:
        """
        Load all domains from the docs directory.
        Returns dict of domain_name → chunk_count.
        """
        result = {}
        domain_names = self.discover_domains()

        for domain_name in domain_names:
            try:
                count = await self.load_domain(domain_name)
                result[domain_name] = count
            except Exception as e:
                logger.error(f"Failed to load domain '{domain_name}': {e}")
                result[domain_name] = 0

        self._initialized = True
        logger.info(f"Knowledge base loaded: {result}")
        return result

    async def load_domain(self, domain_name: str, force: bool = False) -> int:
        """
        Load (or reload) a single knowledge domain.

        Args:
            domain_name: Name of the domain subfolder.
            force: Force reload even if files haven't changed.

        Returns:
            Number of chunks loaded.
        """
        docs_path = os.path.join(self.docs_dir, domain_name)
        prompt_path = os.path.join(self.prompts_dir, f"{domain_name}.txt")

        if not os.path.isdir(docs_path):
            logger.warning(f"Domain docs path not found: {docs_path}")
            return 0

        domain = KnowledgeDomain(domain_name, docs_path, prompt_path)
        domain.load_prompt()

        # Scan for document files
        files = self._scan_files(docs_path)
        if not files:
            logger.info(f"No documents found in domain '{domain_name}'")
            self.domains[domain_name] = domain
            return 0

        # Check if files changed (skip reload if not forced and unchanged)
        file_hashes = {f: self._file_hash(f) for f in files}
        existing = self.domains.get(domain_name)
        if not force and existing and existing.file_hashes == file_hashes:
            logger.info(f"Domain '{domain_name}' unchanged, skipping reload.")
            return existing.chunk_count

        # Load and chunk all files
        all_chunks: List[Document] = []
        for filepath in files:
            try:
                chunks = self._load_and_chunk_file(filepath, domain_name)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Failed to process {filepath}: {e}")

        domain.doc_count = len(files)
        domain.chunk_count = len(all_chunks)
        domain.file_hashes = file_hashes
        domain.last_loaded = time.time()

        # Delete old domain docs and insert new ones
        if all_chunks:
            # Note: PGVector doesn't have a great "delete by metadata" in all versions
            # We use add_documents which will create new entries
            store = self._get_vectorstore()
            try:
                store.add_documents(all_chunks)
                logger.info(
                    f"Domain '{domain_name}': loaded {len(files)} files → "
                    f"{len(all_chunks)} chunks"
                )
            except Exception as e:
                logger.error(f"Failed to store chunks for '{domain_name}': {e}")
                return 0

        self.domains[domain_name] = domain
        return len(all_chunks)

    # ----- Search -----

    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        k: int = 5,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant documents.

        Args:
            query: Search query text.
            domain: Optional domain filter.
            k: Max results.
            score_threshold: Minimum similarity threshold.

        Returns:
            List of result dicts with content, metadata, and similarity.
        """
        store = self._get_vectorstore()
        filter_dict = {"domain": domain} if domain else None

        try:
            results = store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter_dict,
                score_threshold=score_threshold,
            )
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return []

        output = []
        for doc, score in results:
            output.append({
                "content": doc.page_content,
                "similarity": score,
                "source": doc.metadata.get("source", ""),
                "domain": doc.metadata.get("domain", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                **doc.metadata,
            })

        return output

    # ----- Hot Reload (C3) -----

    async def reload(self, domain: Optional[str] = None) -> Dict[str, int]:
        """
        Hot-reload knowledge base (C3).

        Args:
            domain: Specific domain to reload, or None for all.

        Returns:
            Dict of domain → chunk_count.
        """
        if domain:
            count = await self.load_domain(domain, force=True)
            return {domain: count}
        else:
            return await self.load_all()

    def get_status(self) -> Dict[str, Any]:
        """Get knowledge base status for admin commands."""
        return {
            "initialized": self._initialized,
            "domains": {
                name: domain.to_dict()
                for name, domain in self.domains.items()
            },
            "total_domains": len(self.domains),
            "total_chunks": sum(d.chunk_count for d in self.domains.values()),
            "total_docs": sum(d.doc_count for d in self.domains.values()),
        }

    def get_domain_prompt(self, domain_name: str) -> str:
        """Get the custom prompt for a domain, if any."""
        domain = self.domains.get(domain_name)
        if domain:
            return domain.prompt_text
        # Try loading from file directly
        prompt_path = os.path.join(self.prompts_dir, f"{domain_name}.txt")
        if os.path.isfile(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""

    # ----- Internal helpers -----

    def _scan_files(self, directory: str) -> List[str]:
        """Recursively scan for supported document files."""
        files = []
        for root, _, filenames in os.walk(directory):
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, fname))
        return files

    def _file_hash(self, filepath: str) -> str:
        """Compute MD5 hash of a file for change detection."""
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_and_chunk_file(
        self, filepath: str, domain: str
    ) -> List[Document]:
        """Load a file, chunk its content, and return LangChain Documents."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return []

        # Get relative path for source reference
        rel_path = os.path.relpath(filepath, self.docs_dir)

        # Chunk the content
        chunks = self.chunker.chunk(content)

        documents = []
        for i, chunk_text in enumerate(chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "domain": domain,
                    "source": rel_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_name": os.path.basename(filepath),
                    "content_type": "knowledge_base",
                },
            )
            documents.append(doc)

        return documents


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_manager: Optional[KnowledgeManager] = None


def get_knowledge_manager() -> KnowledgeManager:
    """Return the singleton KnowledgeManager."""
    global _manager
    if _manager is None:
        _manager = KnowledgeManager()
    return _manager
