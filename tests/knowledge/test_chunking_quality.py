from langchain_core.documents import Document

from knowledge.manager import (
    KnowledgeManager,
    MarkdownSemanticChunker,
    _dedupe_chunks_by_content,
)


def test_dedupe_ignores_category_context_prefix():
    chunks = [
        Document(page_content="[Lore/Easter eggs] Same fact", metadata={"source": "lore.md"}),
        Document(page_content="[Other/Easter eggs] Same fact", metadata={"source": "other.md"}),
    ]

    assert len(_dedupe_chunks_by_content(chunks)) == 1


def test_markdown_chunker_removes_navigation_and_keeps_provenance():
    text = """---
title: Fire
category: Environment
type: Fire
source_url: https://example.test/fire
scraped_at: 2026-02-24
method: archived_wiki_markdown
---
# Fire
Game mechanics Zombie • Health • Crafting • Building • Fire • AI • Noise

Fire spreads to survivors and structures.
"""

    chunks = MarkdownSemanticChunker().chunk(text)

    assert len(chunks) == 1
    content, metadata = chunks[0]
    assert "Game mechanics" not in content
    assert "Fire spreads" in content
    assert metadata["source_url"] == "https://example.test/fire"
    assert metadata["scraped_at"] == "2026-02-24"
    assert metadata["method"] == "archived_wiki_markdown"


def test_heading_only_markdown_still_produces_searchable_chunk(tmp_path):
    docs = tmp_path / "docs"
    source = docs / "general" / "members.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Members\n\n### Peachiin\n\n### Simon\n", encoding="utf-8")
    manager = KnowledgeManager(docs_dir=str(docs), chunk_size=100, chunk_overlap=10)

    chunks = manager._load_and_chunk_file(str(source), "general")

    assert len(chunks) == 1
    assert "Peachiin" in chunks[0].page_content
    assert "Simon" in chunks[0].page_content


def test_scanner_ignores_generated_file_manifest(tmp_path):
    docs = tmp_path / "docs" / "pz"
    docs.mkdir(parents=True)
    (docs / "list_md.txt").write_text("Weapons/Axes.md", encoding="utf-8")
    guide = docs / "guide.txt"
    guide.write_text("Real guide content", encoding="utf-8")
    manager = KnowledgeManager(docs_dir=str(tmp_path / "docs"))

    assert manager._scan_files(str(docs)) == [str(guide)]


def test_oversized_recipe_is_split():
    products = ", ".join(f"Item {index} ×1" for index in range(80))
    text = f"# Packing\n\n### Recipe: Pack Items, One of:, {products}\n- **Tool:** Box"

    chunks = MarkdownSemanticChunker(chunk_size=200, chunk_overlap=20).chunk(text)

    assert len(chunks) > 1
    assert max(len(content) for content, _ in chunks) < 500
