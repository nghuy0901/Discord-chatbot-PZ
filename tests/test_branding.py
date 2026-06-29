from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKED_EXTENSIONS = {".py", ".md", ".yml", ".yaml", ".txt"}
IGNORED_PARTS = {".git", ".worktrees", "__pycache__", ".pytest_cache", "pgdata", "tmp"}
IGNORED_PREFIXES = {
    ("docs", "audit"),
    ("docs", "superpowers", "plans"),
}
LEGACY_ALIAS_FILE = ROOT / "src" / "config" / "legacy_env.py"
LEGACY_UPPER = "CL" "CT"
LEGACY_LOWER = LEGACY_UPPER.lower()


def is_ignored(path: Path) -> bool:
    relative_parts = path.relative_to(ROOT).parts
    if any(part in IGNORED_PARTS for part in relative_parts):
        return True
    return any(relative_parts[: len(prefix)] == prefix for prefix in IGNORED_PREFIXES)


def iter_checked_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if is_ignored(path):
            continue
        if path == LEGACY_ALIAS_FILE:
            continue
        if path.suffix.lower() in CHECKED_EXTENSIONS:
            yield path


def test_user_facing_brand_is_nomnom_not_legacy():
    offenders = []
    for path in iter_checked_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if LEGACY_UPPER in text or LEGACY_LOWER in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_legacy_env_aliases_are_isolated_and_deprecated():
    if not LEGACY_ALIAS_FILE.exists():
        return
    text = LEGACY_ALIAS_FILE.read_text(encoding="utf-8")
    assert LEGACY_UPPER in text or LEGACY_LOWER in text
    assert "deprecated" in text.lower()
    assert "remove after one release" in text.lower()
