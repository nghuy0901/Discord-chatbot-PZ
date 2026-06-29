"""Chunk-level citation assembly and rendering."""

from rag.citations import (
    build_citation_context,
    referenced_labels,
    render_sources_footer,
)


def kb_chunk():
    return {
        "content": "Axes degrade with repeated use and eventually break.",
        "source": "pz/Weapons/Axes.md",
        "heading_path": "Weapons > Axes",
        "doc_id": "pz:Weapons/Axes.md:0:deadbeef",
        "domain": "pz",
        "content_type": "knowledge_base",
        "source_kind": "knowledge_base",
        "trusted": True,
        "similarity": 0.71,
    }


def chat_chunk():
    return {
        "content": "I always carry a spare axe.",
        "message_id": "12345",
        "author_name": "bob",
        "source_kind": "approved_chat",
        "trusted": True,
        "similarity": 0.52,
    }


def test_build_context_numbers_kb_first():
    ctx, prov = build_citation_context([kb_chunk()], [chat_chunk()])
    assert "[1]" in ctx and "[2]" in ctx
    assert "Axes.md" in ctx
    assert len(prov) == 2
    assert prov[0].label == "1"
    assert prov[0].source_id == "pz:Weapons/Axes.md:0:deadbeef"
    assert "Axes.md" in prov[0].locator
    assert prov[1].label == "2"


def test_excerpt_is_carried_into_provenance():
    _, prov = build_citation_context([kb_chunk()], [])
    assert "degrade" in prov[0].excerpt


def test_referenced_labels_in_order_no_dupes():
    assert referenced_labels("a [2] b [1] c [2]") == ["2", "1"]


def test_footer_lists_only_cited_sources():
    _, prov = build_citation_context([kb_chunk()], [chat_chunk()])
    footer = render_sources_footer("Axes break over time [1].", prov, language="vi")
    assert "[1]" in footer
    assert "Axes.md" in footer
    assert "[2]" not in footer


def test_footer_falls_back_to_top_sources():
    _, prov = build_citation_context([kb_chunk()], [chat_chunk()])
    footer = render_sources_footer("No markers in this answer.", prov)
    assert "[1]" in footer  # fallback shows top sources


def test_no_sources_no_footer():
    assert render_sources_footer("anything [1]", []) == ""
