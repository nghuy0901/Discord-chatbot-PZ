from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from rag.metrics import RAGMetric


class RAGDecision(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class ProvenanceItem:
    source_id: str
    source_kind: str
    domain: str
    trusted: bool
    rank: int
    similarity: float = 0.0
    # Citation surface — lets a citation marker [n] in the answer resolve back
    # to a specific chunk/passage rather than just a file.
    label: str = ""          # the marker shown to the model/user, e.g. "1"
    locator: str = ""        # human-readable pointer, e.g. "pz/Axes.md › Durability"
    source: str = ""         # file path or message id
    heading_path: str = ""   # section breadcrumb for KB chunks
    excerpt: str = ""        # short quote from the cited chunk
    url: str = ""            # message link / external url when available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "domain": self.domain,
            "trusted": self.trusted,
            "rank": self.rank,
            "similarity": self.similarity,
            "label": self.label,
            "locator": self.locator,
            "source": self.source,
            "heading_path": self.heading_path,
            "excerpt": self.excerpt,
            "url": self.url,
        }


@dataclass
class RAGBuildResult:
    context: str
    domain_prompt: str
    metric: RAGMetric
    decision: RAGDecision
    decision_reason: str
    evidence_score: float
    provenance: List[ProvenanceItem] = field(default_factory=list)
    retrieved_results: List[Dict[str, Any]] = field(default_factory=list)
    primary_domain: Optional[str] = None
