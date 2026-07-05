import re
from dataclasses import asdict, dataclass
from typing import Iterable, List, Set


_CITATION_PATTERN = re.compile(r"\[(KB-\d+|\d+)\]")


@dataclass(frozen=True)
class CitationValidationResult:
    citation_ids: List[str]
    valid_ids: List[str]
    missing_ids: List[str]
    is_valid: bool
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def extract_citation_ids(answer: str) -> List[str]:
    """Return citation ids in first-seen order from an answer."""
    seen: Set[str] = set()
    citations: List[str] = []
    for match in _CITATION_PATTERN.finditer(answer or ""):
        citation_id = match.group(1)
        if citation_id not in seen:
            seen.add(citation_id)
            citations.append(citation_id)
    return citations


def validate_citations(
    answer: str,
    evidence_ids: Iterable[str],
    *,
    require_citation: bool = False,
) -> CitationValidationResult:
    citation_ids = extract_citation_ids(answer)
    valid_evidence_ids = {str(evidence_id) for evidence_id in evidence_ids}
    valid_ids = [
        citation_id for citation_id in citation_ids if citation_id in valid_evidence_ids
    ]
    missing_ids = [
        citation_id for citation_id in citation_ids if citation_id not in valid_evidence_ids
    ]

    if require_citation and not citation_ids:
        return CitationValidationResult(
            citation_ids=[],
            valid_ids=[],
            missing_ids=[],
            is_valid=False,
            error="missing_required_citation",
        )

    return CitationValidationResult(
        citation_ids=citation_ids,
        valid_ids=valid_ids,
        missing_ids=missing_ids,
        is_valid=not missing_ids,
    )
