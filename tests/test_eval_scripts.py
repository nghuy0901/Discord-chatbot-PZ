import ast
from pathlib import Path


def test_ragas_runner_imports_official_ragas_package():
    text = Path("evaluation/ragas_runner.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert any(name == "ragas" or name.startswith("ragas.") for name in imports)


def test_run_ragas_eval_uses_ragas_runner():
    text = Path("scripts/run_ragas_eval.py").read_text(encoding="utf-8")
    assert "from evaluation.ragas_runner import run_ragas_evaluation" in text


def test_reindex_verification_qualifies_embedding_metadata():
    text = Path("scripts/reindex_vectors.py").read_text(encoding="utf-8")

    assert "COUNT(DISTINCT e.cmetadata->>'source')" in text
    assert "COUNT(DISTINCT e.cmetadata->>'domain')" in text


def test_release_eval_builds_complete_request_context():
    text = Path("scripts/run_release_eval.py").read_text(encoding="utf-8")

    assert 'channel_id="release_eval"' in text
    assert 'user_id="release_eval"' in text


def test_release_eval_initializes_production_retrieval_indices():
    text = Path("scripts/run_release_eval.py").read_text(encoding="utf-8")

    assert "await init_db()" in text
    assert "await get_knowledge_manager().initialize_catalog()" in text
    assert "await init_bm25_indices()" in text


def test_release_eval_scores_post_finalization_decision():
    text = Path("scripts/run_release_eval.py").read_text(encoding="utf-8")

    assert "await finalize_rag_answer" in text
    assert "actual_behavior=actual_decision.value" in text
