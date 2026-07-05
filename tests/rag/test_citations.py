from rag.citations import extract_citation_ids, validate_citations


def test_extracts_kb_and_chat_citation_ids_from_answer():
    answer = "Use a generator for power [KB-1]. A player also mentioned fuel [2]."

    assert extract_citation_ids(answer) == ["KB-1", "2"]


def test_validate_citations_rejects_missing_ids():
    answer = "Generator details are here [KB-1], but this citation is missing [KB-9]."
    evidence_ids = {"KB-1", "1"}

    result = validate_citations(answer, evidence_ids)

    assert result.citation_ids == ["KB-1", "KB-9"]
    assert result.valid_ids == ["KB-1"]
    assert result.missing_ids == ["KB-9"]
    assert result.is_valid is False


def test_validate_citations_requires_citation_when_requested():
    result = validate_citations("Generator details are here.", {"KB-1"}, require_citation=True)

    assert result.is_valid is False
    assert result.error == "missing_required_citation"
