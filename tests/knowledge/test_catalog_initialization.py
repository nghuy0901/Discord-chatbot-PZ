import pytest

from knowledge.manager import KnowledgeManager


@pytest.mark.asyncio
async def test_catalog_initialization_registers_domains_without_vectorstore(tmp_path):
    docs = tmp_path / "docs" / "pz"
    prompts = tmp_path / "prompts"
    docs.mkdir(parents=True)
    prompts.mkdir()
    (docs / "axe.md").write_text("# Axe\nAxe is a tool.", encoding="utf-8")
    (prompts / "pz.txt").write_text("Use trusted sources.", encoding="utf-8")

    manager = KnowledgeManager(
        docs_dir=str(tmp_path / "docs"), prompts_dir=str(prompts)
    )
    result = await manager.initialize_catalog()

    assert result == {"pz": 1}
    assert manager.get_status()["initialized"] is True
    assert manager.domains["pz"].chunk_count == 0
    assert manager.get_domain_prompt("pz") == "Use trusted sources."
    assert manager._vectorstore is None
