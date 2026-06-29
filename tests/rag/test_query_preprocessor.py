"""Cross-lingual augmentation & bilingual domain routing (audit H3/M4/L5)."""

from rag.query_preprocessor import QueryPreprocessor


def test_vietnamese_query_augmented_with_english_terms():
    p = QueryPreprocessor()
    out, meta = p.preprocess("rìu nào chặt cây nhanh nhất?")
    assert meta["language"] == "vi"
    low = out.lower()
    # English equivalents appended so BM25 can hit the English KB
    assert "axe" in low
    assert "tree" in low or "wood" in low
    # original Vietnamese preserved
    assert "rìu" in low


def test_vietnamese_pz_domain_detected():
    p = QueryPreprocessor()
    _, meta = p.preprocess("công thức làm băng gạc vô trùng")
    assert meta["domain"] == "pz"


def test_server_rules_domain_detected_for_vietnamese():
    p = QueryPreprocessor()
    _, meta = p.preprocess("nội quy server về xây nhà là gì")
    assert meta["domain"] in ("server_rules", "pz")


def test_english_query_not_augmented():
    p = QueryPreprocessor()
    out, meta = p.preprocess("what is the strongest axe")
    assert meta["language"] == "en"
    # No Vietnamese glossary expansion duplicated onto an English query
    assert out.lower().count("axe") == 1


def test_ascii_vietnamese_detected():
    p = QueryPreprocessor()
    # typed without diacritics — should still be detected as Vietnamese (L5)
    assert p._detect_language("minh khong biet lam sao") == "vi"
