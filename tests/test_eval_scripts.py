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
