"""
Knowledge Manager (D2 + C2 + C3) — Load, chunk, embed, and manage
multi-domain static knowledge base documents.

Features:
- Multi-domain support (pz, server_rules, general, etc.)
- Hybrid chunking: MarkdownSemanticChunker + SentenceChunker fallback (C2)
- Hot-reload via @mention commands (C3)
- Domain-specific prompts
- PGVector storage for similarity search
"""

import os
import re
import time
import logging
import hashlib
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from langchain_core.documents import Document
from rag.trust import trusted_kb_domains

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
KB_VECTOR_FILTER_OVERFETCH_MULTIPLIER: int = int(
    os.getenv("KB_VECTOR_FILTER_OVERFETCH_MULTIPLIER", "5")
)

SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst"}
IGNORED_KNOWLEDGE_FILES = {"list_md.txt"}


def _stable_doc_id(domain: str, source: str, chunk_index: int, text: str) -> str:
    normalized_source = source.replace(os.sep, "/")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{domain}:{normalized_source}:{chunk_index}:{digest}"


def _dedupe_chunks_by_content(chunks: List["Document"]) -> List["Document"]:
    """Collapse byte-identical chunk texts so each unique text is embedded once.

    The markdown KB repeats some sections verbatim across many item pages (e.g. a
    "Trash — can be used as fuel" block appearing ~30×), which previously produced
    hundreds of identical embeddings that crowded out diverse results (audit C3).
    """
    seen: set = set()
    unique: List["Document"] = []
    for chunk in chunks:
        canonical = re.sub(r"^\[[^\]]+\]\s*", "", chunk.page_content)
        canonical = re.sub(r"\s+", " ", canonical).strip().casefold()
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(chunk)
    return unique


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
# Content mode enum
# ---------------------------------------------------------------------------
class ContentMode(Enum):
    """Detected content mode for a markdown section."""
    RECIPE = "recipe"           # ### Recipe: ... with ingredients/tools
    RECORD_ITEM = "record_item" # ### Name + list of - **Key:** Value
    PROSE = "prose"             # Free-form paragraphs / explanatory text


# ---------------------------------------------------------------------------
# MarkdownSemanticChunker — context-aware markdown chunker
# ---------------------------------------------------------------------------
class MarkdownSemanticChunker:
    """
    Split markdown documents into semantically meaningful chunks by
    respecting document structure: headings, record blocks, recipe blocks.

    Modes:
    - RECIPE:      ### Recipe: ... → atomic chunk (1 recipe = 1 chunk)
    - RECORD_ITEM: ### Name + key-value list → atomic chunk (1 item = 1 chunk)
    - PROSE:       General text → SentenceChunker with heading-path context

    Compatible with all KB domains (pz, general, server_rules, etc.)
    """

    # Sections to skip entirely (noise)
    NOISE_HEADINGS = {
        "see also", "gallery", "references", "external links",
        "navigation", "categories", "notes", "trivia",
    }

    # Pattern: "### Recipe: <name>, <products>"
    RE_RECIPE_HEADING = re.compile(
        r"^###\s+Recipe:\s+(.+)", re.IGNORECASE
    )
    # Pattern: "- **Key:** Value"
    RE_KEY_VALUE = re.compile(r"^\s*-\s+\*\*[^*]+\*\*[:\s]")
    # Pattern: any heading "# ...", "## ...", "### ..."
    RE_HEADING = re.compile(r"^(#{1,6})\s+(.+)")

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_chunker = SentenceChunker(chunk_size, chunk_overlap)

    def chunk(
        self, text: str, source: str = ""
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Parse markdown text and return list of (chunk_text, metadata).

        Args:
            text: Raw markdown content (including optional YAML frontmatter).
            source: File path for context prefix.

        Returns:
            List of (chunk_text, chunk_metadata) tuples.
        """
        # 1. Parse and strip YAML frontmatter
        frontmatter, body = self._parse_frontmatter(text)
        body = "\n".join(
            line for line in body.splitlines()
            if not self._is_navigation_line(line)
        )

        # 2. Split into heading-delimited sections
        sections = self._split_into_sections(body)

        # 3. Process each section based on detected content mode
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        for heading_path, heading_text, section_body in sections:
            # Skip noise sections
            if self._is_noise(heading_text):
                continue

            mode = self._detect_mode(heading_text, section_body)
            section_chunks = self._process_section(
                heading_path, heading_text, section_body, mode, frontmatter
            )
            chunks.extend(section_chunks)

        if not chunks:
            headings = [heading for _, heading, _ in sections if heading]
            if headings:
                chunks.append((
                    "\n".join(headings),
                    {
                        "heading_path": " > ".join(headings),
                        "content_mode": "headings",
                        "category": frontmatter.get("category", ""),
                        "type": frontmatter.get("type", ""),
                        "source_url": frontmatter.get("source_url", ""),
                        "scraped_at": str(frontmatter.get("scraped_at", "")),
                        "method": frontmatter.get("method", ""),
                    },
                ))
        return chunks

    @staticmethod
    def _is_navigation_line(line: str) -> bool:
        low = line.strip().lower()
        return line.count("•") >= 5 and low.startswith(
            ("items ", "player ", "game mechanics ", "lore ", "locations ", "vehicle ")
        )

    # ------------------------------------------------------------------ #
    #  Frontmatter
    # ------------------------------------------------------------------ #
    def _parse_frontmatter(self, text: str) -> Tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter and return (metadata_dict, body_text)."""
        text = text.strip()
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    fm = {}
                return fm, parts[2].strip()
        return {}, text

    # ------------------------------------------------------------------ #
    #  Section splitting
    # ------------------------------------------------------------------ #
    def _split_into_sections(
        self, text: str
    ) -> List[Tuple[List[str], str, str]]:
        """
        Split text by headings into (heading_path, heading_text, body).

        heading_path: list of ancestor headings, e.g. ["Clothing — Armor", "Armor"]
        heading_text: the current heading text
        body: the text content under this heading until the next heading
        """
        lines = text.split("\n")
        sections: List[Tuple[List[str], str, str]] = []

        # Track heading hierarchy: level → heading_text
        heading_stack: Dict[int, str] = {}
        current_heading = ""
        current_body_lines: List[str] = []

        def _flush():
            body = "\n".join(current_body_lines).strip()
            if current_heading or body:
                # Build heading path from stack
                path = []
                for lvl in sorted(heading_stack.keys()):
                    if heading_stack[lvl]:
                        path.append(heading_stack[lvl])
                sections.append((path, current_heading, body))

        for line in lines:
            m = self.RE_HEADING.match(line)
            if m:
                _flush()
                level = len(m.group(1))  # number of #
                heading_text = m.group(2).strip()

                # Update stack: set current level, clear deeper levels
                heading_stack[level] = heading_text
                for lvl in list(heading_stack.keys()):
                    if lvl > level:
                        del heading_stack[lvl]

                current_heading = heading_text
                current_body_lines = []
            else:
                current_body_lines.append(line)

        _flush()  # last section
        return sections

    # ------------------------------------------------------------------ #
    #  Content mode detection
    # ------------------------------------------------------------------ #
    def _detect_mode(self, heading: str, body: str) -> ContentMode:
        """Detect content mode from heading and body patterns."""
        # Recipe mode: heading starts with "Recipe:"
        if self.RE_RECIPE_HEADING.match(f"### {heading}"):
            return ContentMode.RECIPE

        # Record item mode: body is mostly key-value list items
        body_lines = [l for l in body.split("\n") if l.strip()]
        if body_lines:
            kv_count = sum(
                1 for l in body_lines if self.RE_KEY_VALUE.match(l)
            )
            ratio = kv_count / len(body_lines)
            if ratio >= 0.5 and kv_count >= 2:
                return ContentMode.RECORD_ITEM

        return ContentMode.PROSE

    # ------------------------------------------------------------------ #
    #  Noise detection
    # ------------------------------------------------------------------ #
    def _is_noise(self, heading: str) -> bool:
        """Check if a heading is noise (should be skipped)."""
        return heading.lower().strip() in self.NOISE_HEADINGS

    # ------------------------------------------------------------------ #
    #  Section processing
    # ------------------------------------------------------------------ #
    def _process_section(
        self,
        heading_path: List[str],
        heading_text: str,
        body: str,
        mode: ContentMode,
        frontmatter: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Process a section and return chunks with metadata."""

        # Build context prefix from heading hierarchy
        context_prefix = self._build_context_prefix(heading_path, frontmatter)

        base_meta = {
            "heading_path": " > ".join(heading_path) if heading_path else "",
            "content_mode": mode.value,
            "category": frontmatter.get("category", ""),
            "type": frontmatter.get("type", ""),
            "source_url": frontmatter.get("source_url", ""),
            "scraped_at": str(frontmatter.get("scraped_at", "")),
            "method": frontmatter.get("method", ""),
        }

        if mode == ContentMode.RECIPE:
            return self._chunk_recipe(
                heading_text, body, context_prefix, base_meta
            )
        elif mode == ContentMode.RECORD_ITEM:
            return self._chunk_record_item(
                heading_text, body, context_prefix, base_meta
            )
        else:
            return self._chunk_prose(
                heading_text, body, context_prefix, base_meta
            )

    def _build_context_prefix(
        self, heading_path: List[str], frontmatter: Dict[str, Any]
    ) -> str:
        """Build a context prefix like [Clothing/Armor] or [Player/Health]."""
        category = frontmatter.get("category", "")
        ftype = frontmatter.get("type", "")
        if category and ftype:
            return f"[{category}/{ftype}]"
        elif category:
            return f"[{category}]"
        return ""

    # ------------------------------------------------------------------ #
    #  Mode-specific chunking
    # ------------------------------------------------------------------ #
    def _chunk_recipe(
        self,
        heading: str,
        body: str,
        context_prefix: str,
        base_meta: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Recipe mode: entire recipe = 1 atomic chunk."""
        # Reconstruct full recipe text
        recipe_text = f"{heading}\n{body}".strip()
        if context_prefix:
            recipe_text = f"{context_prefix} {recipe_text}"

        # Clean markdown bold syntax for cleaner embedding
        recipe_text = self._clean_markdown_bold(recipe_text)

        meta = {**base_meta, "record_name": heading}
        return self._split_atomic(recipe_text, meta)

    def _chunk_record_item(
        self,
        heading: str,
        body: str,
        context_prefix: str,
        base_meta: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Record item mode: heading + key-value list = 1 atomic chunk."""
        item_text = f"{heading}\n{body}".strip()
        if context_prefix:
            item_text = f"{context_prefix} {item_text}"

        # Clean markdown bold syntax
        item_text = self._clean_markdown_bold(item_text)

        meta = {**base_meta, "record_name": heading}
        return self._split_atomic(item_text, meta)

    def _split_atomic(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        if len(text) <= self.chunk_size * 2:
            return [(text, metadata)]
        parts = self.sentence_chunker.chunk(text)
        label_match = re.match(r"^(\[[^\]]+\])", text)
        label = label_match.group(1) if label_match else ""
        record = str(metadata.get("record_name") or "").split(",", 1)[0]
        output = []
        for index, part in enumerate(parts):
            if index:
                part = " ".join(value for value in (label, record, part) if value)
            output.append((
                part,
                {
                    **metadata,
                    "record_part_index": index,
                    "record_total_parts": len(parts),
                },
            ))
        return output

    def _chunk_prose(
        self,
        heading: str,
        body: str,
        context_prefix: str,
        base_meta: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Prose mode: use SentenceChunker with heading context prefix."""
        if not body.strip():
            return []

        # Prepend heading for context
        full_text = body.strip()

        # Use SentenceChunker for sub-chunking
        sub_chunks = self.sentence_chunker.chunk(full_text)
        if not sub_chunks:
            return []

        result = []
        for i, chunk_text in enumerate(sub_chunks):
            # Add context prefix and heading to each sub-chunk
            prefixed = chunk_text
            if heading:
                prefixed = f"{heading}: {prefixed}"
            if context_prefix:
                prefixed = f"{context_prefix} {prefixed}"

            meta = {
                **base_meta,
                "prose_chunk_index": i,
                "prose_total_chunks": len(sub_chunks),
            }
            result.append((prefixed, meta))

        return result

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_markdown_bold(text: str) -> str:
        """Remove markdown bold markers **text** → text for cleaner embedding."""
        return re.sub(r"\*\*([^*]+)\*\*", r"\1", text)


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
        self.sentence_chunker = SentenceChunker(chunk_size, chunk_overlap)
        self.md_chunker = MarkdownSemanticChunker(chunk_size, chunk_overlap)
        self.domains: Dict[str, KnowledgeDomain] = {}
        self._vectorstore = None
        self._initialized = False
        self.kb_version = os.getenv("KB_VERSION", "v1")

    # ----- Initialization -----

    def _get_vectorstore(self):
        """Get or create the knowledge base PGVector store (separate collection)."""
        if self._vectorstore is None:
            from rag.db import get_embeddings, PGVECTOR_CONNECTION
            from langchain_community.vectorstores import PGVector

            self._vectorstore = PGVector(
                collection_name=KB_COLLECTION,
                connection_string=PGVECTOR_CONNECTION,
                embedding_function=get_embeddings(),
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

    async def initialize_catalog(self) -> Dict[str, int]:
        """Register local KB domains without embedding any document.

        A query service must never re-embed an entire corpus during startup.
        Apart from causing a cold-start stampede, a temporary embedding outage
        used to leave ``self.domains`` empty and made the router skip an already
        indexed KB altogether.  The explicit indexer owns vector writes;
        startup only needs domain names and prompts for safe routing.
        """
        result: Dict[str, int] = {}
        for domain_name in self.discover_domains():
            docs_path = os.path.join(self.docs_dir, domain_name)
            prompt_path = os.path.join(self.prompts_dir, f"{domain_name}.txt")
            domain = self.domains.get(domain_name) or KnowledgeDomain(
                domain_name, docs_path, prompt_path
            )
            domain.load_prompt()
            domain.doc_count = len(self._scan_files(docs_path))
            self.domains[domain_name] = domain
            result[domain_name] = domain.doc_count

        self._initialized = True
        logger.info("Knowledge catalog initialized without indexing: %s", result)
        return result

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

        existing = self.domains.get(domain_name)
        domain = KnowledgeDomain(domain_name, docs_path, prompt_path)
        domain.load_prompt()

        # Scan for document files
        files = self._scan_files(docs_path)
        if not files:
            logger.info(f"No documents found in domain '{domain_name}'")
            self.domains[domain_name] = domain
            self._refresh_kb_version()
            return 0

        # Check if files changed (skip reload if not forced and unchanged)
        file_hashes = {f: self._file_hash(f) for f in files}
        if not force and existing and existing.file_hashes == file_hashes:
            logger.info(f"Domain '{domain_name}' unchanged, skipping reload.")
            return existing.chunk_count

        # Keep routing available even if a subsequent embedding/upsert fails.
        # The existing collection may still be valid, and a failed index attempt
        # must surface as an indexer error rather than as a fake "no KB domain".
        self.domains[domain_name] = domain

        # Load and chunk all files
        all_chunks: List[Document] = []
        for filepath in files:
            try:
                chunks = self._load_and_chunk_file(filepath, domain_name)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Failed to process {filepath}: {e}")

        # Collapse byte-identical chunks before embedding (audit C3).
        all_chunks = _dedupe_chunks_by_content(all_chunks)

        domain.doc_count = len(files)
        domain.chunk_count = len(all_chunks)
        domain.file_hashes = file_hashes
        domain.last_loaded = time.time()

        # Upsert new chunks by stable doc_id, THEN prune chunks that no longer
        # exist. Insert-before-delete means the domain is never emptied mid-reload
        # on a partial failure, and add_documents(ids=…) makes re-inserts
        # idempotent so repeated @reload cannot multiply rows (audit C3/M9).
        if all_chunks:
            store = self._get_vectorstore()
            try:
                from rag.db import get_pool, ensure_embedding_collection_signature

                # Indexing is the single writer for a KB collection.  Record the
                # actual embedding input mode here so a raw-text v2 index can
                # coexist with the legacy tokenized v1 collection.
                await ensure_embedding_collection_signature(KB_COLLECTION)

                ids = [c.metadata["doc_id"] for c in all_chunks]
                store.add_documents(all_chunks, ids=ids)

                pool = await get_pool()
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        DELETE FROM langchain_pg_embedding e
                        USING langchain_pg_collection c
                        WHERE e.collection_id = c.uuid
                          AND c.name = $1
                          AND e.cmetadata->>'domain' = $2
                          AND NOT (e.cmetadata->>'doc_id' = ANY($3::text[]));
                        """,
                        KB_COLLECTION,
                        domain_name,
                        ids,
                    )
                logger.info(
                    f"Domain '{domain_name}': loaded {len(files)} files → "
                    f"{len(all_chunks)} unique chunks (upserted by doc_id)"
                )
            except Exception as e:
                logger.error(f"Failed to store chunks for '{domain_name}': {e}")
                return 0

        self.domains[domain_name] = domain
        self._refresh_kb_version()
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
        if domain and domain not in trusted_kb_domains():
            return []
        search_k = k
        if domain:
            search_k = max(k, k * KB_VECTOR_FILTER_OVERFETCH_MULTIPLIER)

        try:
            results = store.similarity_search_with_relevance_scores(
                query=query,
                k=search_k,
                score_threshold=score_threshold,
            )
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return []

        output = []
        for doc, score in results:
            doc_domain = doc.metadata.get("domain", "")
            if domain and doc_domain != domain:
                continue
            if doc_domain not in trusted_kb_domains():
                continue
            output.append({
                "content": doc.page_content,
                "similarity": score,
                "source": doc.metadata.get("source", ""),
                "domain": doc.metadata.get("domain", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                **doc.metadata,
            })
            if len(output) >= k:
                break

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
        old_kb_version = self.kb_version

        # Lazy-import to prevent circular imports
        try:
            from utils.cache import get_query_cache
            cache = get_query_cache()
        except Exception:
            cache = None

        if domain:
            count = await self.load_domain(domain, force=True)
            if cache:
                await cache.invalidate_domain(domain, kb_version=old_kb_version)
            await self._refresh_kb_bm25()
            return {domain: count}
        else:
            res = await self.load_all()
            if cache:
                await cache.invalidate_domain("all", kb_version=old_kb_version)
            await self._refresh_kb_bm25()
            return res

    async def _refresh_kb_bm25(self) -> None:
        """Keep the KB BM25 index in sync with the vector store after a reload,
        so hybrid search never runs the lexical and semantic arms over divergent
        corpora (audit H4). Non-fatal — a BM25 refresh failure must not break the
        reload itself."""
        try:
            from rag.bm25_search import get_kb_bm25

            await get_kb_bm25().refresh_from_kb()
        except Exception as e:
            logger.warning(f"KB BM25 refresh after reload failed (non-fatal): {e}")

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
                if fname.lower() in IGNORED_KNOWLEDGE_FILES:
                    continue
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

    def _compute_kb_version(
        self,
        domain_hashes: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> str:
        if domain_hashes is None:
            domain_hashes = {
                name: domain.file_hashes
                for name, domain in self.domains.items()
                if domain.file_hashes
            }
        if not domain_hashes:
            return os.getenv("KB_VERSION", "v1")

        hasher = hashlib.sha256()
        for domain_name in sorted(domain_hashes):
            hasher.update(domain_name.encode("utf-8"))
            for path, digest in sorted(domain_hashes[domain_name].items()):
                normalized_path = self._normalize_hash_path(path)
                hasher.update(normalized_path.encode("utf-8"))
                hasher.update(digest.encode("utf-8"))
        return hasher.hexdigest()[:12]

    def _refresh_kb_version(self) -> str:
        self.kb_version = self._compute_kb_version()
        return self.kb_version

    def _normalize_hash_path(self, filepath: str) -> str:
        try:
            if os.path.isabs(filepath):
                filepath = os.path.relpath(filepath, self.docs_dir)
        except ValueError:
            pass
        return filepath.replace(os.sep, "/")

    def _load_and_chunk_file(
        self, filepath: str, domain: str
    ) -> List[Document]:
        """Load a file, chunk its content, and return LangChain Documents.

        Routing:
        - .md files → MarkdownSemanticChunker (heading-aware, mode detection)
        - .txt/.rst  → SentenceChunker (fallback)
        """
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return []

        # Get relative path for source reference
        rel_path = os.path.relpath(filepath, self.docs_dir)
        ext = os.path.splitext(filepath)[1].lower()

        documents = []

        if ext == ".md":
            # ---- MarkdownSemanticChunker ----
            try:
                md_chunks = self.md_chunker.chunk(content, source=rel_path)
                for i, (chunk_text, chunk_meta) in enumerate(md_chunks):
                    doc = Document(
                        page_content=chunk_text,
                        metadata={
                            "domain": domain,
                            "source": rel_path,
                            "doc_id": _stable_doc_id(domain, rel_path, i, chunk_text),
                            "trusted": domain in trusted_kb_domains(),
                            "source_kind": "knowledge_base",
                            "chunk_index": i,
                            "total_chunks": len(md_chunks),
                            "file_name": os.path.basename(filepath),
                            "content_type": "knowledge_base",
                            **chunk_meta,  # heading_path, content_mode, etc.
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                logger.warning(
                    f"MarkdownSemanticChunker failed for {filepath}, "
                    f"falling back to SentenceChunker: {e}"
                )
                # Fallback to SentenceChunker
                documents = self._fallback_chunk(content, rel_path, domain)
        else:
            # ---- SentenceChunker (fallback for .txt, .rst) ----
            documents = self._fallback_chunk(content, rel_path, domain)

        return documents

    def _fallback_chunk(
        self, content: str, rel_path: str, domain: str
    ) -> List[Document]:
        """Fallback chunking using SentenceChunker."""
        chunks = self.sentence_chunker.chunk(content)
        documents = []
        for i, chunk_text in enumerate(chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "domain": domain,
                    "source": rel_path,
                    "doc_id": _stable_doc_id(domain, rel_path, i, chunk_text),
                    "trusted": domain in trusted_kb_domains(),
                    "source_kind": "knowledge_base",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_name": os.path.basename(rel_path),
                    "content_type": "knowledge_base",
                    "content_mode": "sentence_fallback",
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
