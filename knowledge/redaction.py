import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RedactionResult:
    text: str
    redaction_count: int


PATTERNS = (
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.-]?\d){8,10}(?!\d)"), "[REDACTED_PHONE]"),
    (re.compile(r"\b(?:sk|ghp|glpat|xoxb)-[A-Za-z0-9_-]{12,}\b"), "[REDACTED_SECRET]"),
    (re.compile(r"<@!?\d{15,20}>"), "[REDACTED_USER]"),
)


def redact_sensitive_text(text: str) -> RedactionResult:
    output = text
    count = 0
    for pattern, replacement in PATTERNS:
        output, hits = pattern.subn(replacement, output)
        count += hits
    return RedactionResult(text=output, redaction_count=count)
