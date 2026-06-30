import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RedactionResult:
    text: str
    redaction_count: int


PATTERNS = (
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (
        re.compile(
            r"(?<![\w\d])(?:"
            r"(?:\+?84|0)(?:[\s.-]?\d){8,10}"
            r"|\+(?!84)\d{1,3}[\s.-]?(?:\(\d{1,4}\)[\s.-]?)?"
            r"\d(?:[\s.-]?\d){6,12}"
            r")(?!\d)"
        ),
        "[REDACTED_PHONE]",
    ),
    (
        re.compile(
            r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b",
            re.IGNORECASE,
        ),
        "[REDACTED_SECRET]",
    ),
    (
        re.compile(
            r"\b(?:"
            r"sk-[A-Za-z0-9_-]{12,}"
            r"|sk_live_[A-Za-z0-9]{16,}"
            r"|gh[pousr][_-][A-Za-z0-9]{20,}"
            r"|github_pat_[A-Za-z0-9_]{20,}"
            r"|glpat-[A-Za-z0-9_-]{12,}"
            r"|xox[baprs]-[A-Za-z0-9_-]{12,}"
            r"|AKIA[A-Z0-9]{16}"
            r"|AIza[A-Za-z0-9_-]{20,}"
            r"|eyJ[A-Za-z0-9_-]{10,}\."
            r"[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
            r")\b"
        ),
        "[REDACTED_SECRET]",
    ),
    (re.compile(r"<@!?\d{15,20}>"), "[REDACTED_USER]"),
    (re.compile(r"<@&\d{15,20}>"), "[REDACTED_ROLE]"),
    (re.compile(r"<#\d{15,20}>"), "[REDACTED_CHANNEL]"),
)


def redact_sensitive_text(text: str) -> RedactionResult:
    output = text
    count = 0
    for pattern, replacement in PATTERNS:
        output, hits = pattern.subn(replacement, output)
        count += hits
    return RedactionResult(text=output, redaction_count=count)
