from pathlib import Path


def test_migrations_are_numbered_and_non_empty():
    paths = sorted(Path("migrations").glob("*.sql"))
    assert [path.name for path in paths] == [
        "001_init_metrics.sql",
        "002_add_rag_quality_metrics.sql",
        "003_add_eval_runs.sql",
    ]
    assert all(path.read_text(encoding="utf-8").strip() for path in paths)


def test_migration_runner_creates_schema_migrations_table():
    text = Path("scripts/migrate_db.py").read_text(encoding="utf-8")
    assert "schema_migrations" in text
    assert "asyncpg" in text
