from knowledge.manager import KnowledgeManager
from utils.cache import CacheVersion, QueryCache


def test_cache_key_changes_when_prompt_version_changes():
    cache = QueryCache()
    v1 = CacheVersion(
        kb_version="1",
        prompt_version="a",
        llm_model="m",
        embedding_model="e",
        retrieval_config_version="r",
    )
    v2 = CacheVersion(
        kb_version="1",
        prompt_version="b",
        llm_model="m",
        embedding_model="e",
        retrieval_config_version="r",
    )
    assert cache.make_key_for_test("hello", "pz", v1) != cache.make_key_for_test("hello", "pz", v2)


def test_cache_key_changes_when_kb_version_changes():
    cache = QueryCache()
    v1 = CacheVersion(
        kb_version="1",
        prompt_version="p",
        llm_model="m",
        embedding_model="e",
        retrieval_config_version="r",
    )
    v2 = CacheVersion(
        kb_version="2",
        prompt_version="p",
        llm_model="m",
        embedding_model="e",
        retrieval_config_version="r",
    )
    assert cache.make_key_for_test("hello", "pz", v1) != cache.make_key_for_test("hello", "pz", v2)


def test_kb_version_changes_when_file_hashes_change(tmp_path):
    manager = KnowledgeManager(docs_dir=str(tmp_path / "docs"))

    v1 = manager._compute_kb_version({"pz": {"a.md": "hash-a"}})
    v2 = manager._compute_kb_version({"pz": {"a.md": "hash-b"}})

    assert v1 != v2
