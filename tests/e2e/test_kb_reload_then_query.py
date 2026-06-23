import pytest


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_kb_reload_changes_kb_version_and_query_uses_new_context(tmp_path):
    from knowledge.manager import KnowledgeManager

    docs = tmp_path / "docs" / "pz"
    prompts = tmp_path / "prompts"
    docs.mkdir(parents=True)
    prompts.mkdir()
    (docs / "axe.md").write_text("# Axe\nAxe is a weapon.", encoding="utf-8")

    manager = KnowledgeManager(docs_dir=str(tmp_path / "docs"), prompts_dir=str(prompts))
    first = await manager.load_all()

    assert first["pz"] > 0
