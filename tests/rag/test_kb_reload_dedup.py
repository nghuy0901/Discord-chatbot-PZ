from knowledge.manager import KnowledgeManager


def test_kb_document_ids_are_stable(tmp_path):
    docs = tmp_path / "docs" / "pz"
    prompts = tmp_path / "prompts"
    docs.mkdir(parents=True)
    prompts.mkdir()
    (docs / "a.md").write_text("# Axe\nAxe damage info.", encoding="utf-8")

    manager = KnowledgeManager(docs_dir=str(tmp_path / "docs"), prompts_dir=str(prompts))
    chunks = manager._load_and_chunk_file(str(docs / "a.md"), "pz")

    ids = [doc.metadata["doc_id"] for doc in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0].startswith("pz:")
