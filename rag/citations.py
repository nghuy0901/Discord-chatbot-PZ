"""
Citations — chunk-level citation surface for RAG answers.

This module unifies knowledge-base and chat-history results into a single,
consistently numbered citation list so that:

1. The context block injected into the prompt carries stable markers [1], [2], …
2. Each marker resolves back to a *specific chunk/passage* (file + heading +
   excerpt, or a message link) via the provenance list.
3. The markers the model actually uses in its answer can be rendered into a
   user-facing "Sources" footer.

The numbering is shared across KB and chat sources (KB first, as it is the more
authoritative corpus), so a single [n] space is unambiguous end-to-end.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from rag.result import ProvenanceItem

_KB_KINDS = {"knowledge_base"}
_MARKER_RE = re.compile(r"\[(\d{1,3})\]")

# Default budgets (characters) for the injected context block.
DEFAULT_CONTEXT_MAX_CHARS = 4000
DEFAULT_EXCERPT_CHARS = 220
DEFAULT_FOOTER_EXCERPT_CHARS = 140


def _is_kb(result: Dict[str, Any]) -> bool:
    return (
        result.get("content_type") in _KB_KINDS
        or result.get("source_kind") in _KB_KINDS
        or result.get("source_type") in _KB_KINDS
    )


def _one_line(text: str, limit: int) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(collapsed) > limit:
        collapsed = collapsed[: limit - 1].rstrip() + "…"
    return collapsed


def _source_id(result: Dict[str, Any]) -> str:
    if result.get("doc_id"):
        return str(result["doc_id"])
    if result.get("message_id"):
        return str(result["message_id"])
    source = result.get("source", "unknown")
    chunk_index = result.get("chunk_index", 0)
    return f"{source}#{chunk_index}"


def _similarity(result: Dict[str, Any]) -> float:
    from rag.scoring import relevance_score

    return relevance_score(result)


def _kb_locator(result: Dict[str, Any]) -> str:
    source = result.get("source") or result.get("file_name") or "knowledge_base"
    heading = result.get("heading_path") or result.get("record_name") or ""
    return f"{source} › {heading}" if heading else str(source)


def _chat_locator(result: Dict[str, Any]) -> str:
    author = result.get("author_name") or result.get("author_id") or "unknown"
    ts = result.get("timestamp", "")
    if hasattr(ts, "isoformat"):
        ts = ts.isoformat()
    ts = _one_line(ts, 19)
    return f"@{author}" + (f" • {ts}" if ts else "")


def _discord_url(result: Dict[str, Any]) -> str:
    guild = result.get("guild_id")
    channel = result.get("channel_id")
    message = result.get("message_id")
    if guild and channel and message:
        return f"https://discord.com/channels/{guild}/{channel}/{message}"
    return ""


def _locator(result: Dict[str, Any]) -> str:
    return _kb_locator(result) if _is_kb(result) else _chat_locator(result)


def order_sources(
    kb_results: List[Dict[str, Any]],
    chat_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """KB sources first (more authoritative), then chat sources."""
    return list(kb_results) + list(chat_results)


def build_citation_context(
    kb_results: List[Dict[str, Any]],
    chat_results: List[Dict[str, Any]],
    *,
    sanitize: Optional[Callable[[str], str]] = None,
    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
) -> Tuple[str, List[ProvenanceItem]]:
    """Build the numbered context block and the matching provenance list.

    Returns ``(context_str, provenance)`` where every ``ProvenanceItem`` carries
    a ``label`` equal to the ``[n]`` marker used in the context. Provenance is
    produced for every ordered source so metrics stay complete; the context
    block itself is truncated to ``max_chars``.
    """
    ordered = order_sources(kb_results, chat_results)
    if not ordered:
        return "", []

    clean = sanitize or (lambda text: text)

    provenance: List[ProvenanceItem] = []
    body_lines: List[str] = []
    used_chars = 0
    truncated = 0

    for rank, result in enumerate(ordered, start=1):
        marker = str(rank)
        is_kb = _is_kb(result)
        content = clean(str(result.get("content", "")))
        excerpt = _one_line(content, excerpt_chars)
        locator = _locator(result)

        provenance.append(
            ProvenanceItem(
                source_id=_source_id(result),
                source_kind=str(
                    result.get("source_kind") or result.get("content_type") or ""
                ),
                domain=str(result.get("domain") or ""),
                trusted=bool(result.get("trusted")),
                rank=rank,
                similarity=_similarity(result),
                label=marker,
                locator=locator,
                source=str(result.get("source") or result.get("message_id") or ""),
                heading_path=str(result.get("heading_path") or ""),
                excerpt=_one_line(content, DEFAULT_FOOTER_EXCERPT_CHARS),
                url=_discord_url(result),
            )
        )

        tag = "KB" if is_kb else "chat"
        line = f"[{marker}] ({tag}) {locator}\n    {excerpt}"
        if used_chars + len(line) > max_chars and body_lines:
            truncated = len(ordered) - rank + 1
            break
        body_lines.append(line)
        used_chars += len(line) + 1

    header = (
        "[Sources — cite these with their [n] markers]\n"
        "Only state facts supported by the sources below and put the matching "
        "[n] marker right after each such fact. Do not invent markers."
    )
    if truncated:
        body_lines.append(f"... ({truncated} more sources omitted)")

    context_str = header + "\n" + "\n".join(body_lines)
    return context_str, provenance


def referenced_labels(answer_text: str) -> List[str]:
    """Return the [n] markers used in the answer, in first-appearance order."""
    seen: List[str] = []
    for match in _MARKER_RE.finditer(answer_text or ""):
        label = match.group(1)
        if label not in seen:
            seen.append(label)
    return seen


def validate_factual_line_citations(
    answer_text: str,
    provenance: List[ProvenanceItem],
) -> Tuple[bool, List[str]]:
    """Require every substantive answer line to cite an existing source.

    This deterministic guard catches a common judge failure: the model cites a
    few bullets correctly, then appends an uncited recommendation or plausible
    conclusion. Structural headings ending in ``:`` are allowed; factual lines
    are not allowed to borrow a marker from another paragraph.
    """
    valid_labels = {item.label for item in provenance if item.label}
    if not valid_labels:
        return False, ["no_valid_provenance"]

    missing: List[str] = []
    for raw_line in (answer_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not normalized or normalized.startswith("#"):
            continue
        if normalized.endswith(":"):
            continue
        if re.fullmatch(r"[-|:\s]+", normalized):
            continue

        labels = set(_MARKER_RE.findall(normalized))
        if not labels or not labels <= valid_labels:
            missing.append(_one_line(normalized, 160))

    return not missing, missing


def render_sources_footer(
    answer_text: str,
    provenance: List[ProvenanceItem],
    *,
    language: str = "vi",
    max_items: int = 8,
    fallback_top: int = 3,
) -> str:
    """Render a user-facing "Sources" footer from the cited markers.

    Shows the sources the answer actually cited. If the answer cited nothing but
    provenance exists, falls back to listing the top ``fallback_top`` sources so
    the user still gets references.
    """
    if not provenance:
        return ""

    by_label = {item.label: item for item in provenance if item.label}
    labels = [lbl for lbl in referenced_labels(answer_text) if lbl in by_label]

    if labels:
        items = [by_label[lbl] for lbl in labels[:max_items]]
    else:
        items = sorted(provenance, key=lambda p: p.rank)[:fallback_top]

    if not items:
        return ""

    title = "📚 **Nguồn:**" if language == "vi" else "📚 **Sources:**"
    lines = [title]
    for item in items:
        pointer = item.url or item.locator or item.source or item.source_id
        line = f"[{item.label}] {pointer}"
        if item.excerpt:
            line += f' — "{item.excerpt}"'
        lines.append(line)
    return "\n".join(lines)


def render_sources_footer_from_dicts(
    answer_text: str,
    provenance: List[Dict[str, Any]],
    *,
    language: str = "vi",
    max_items: int = 8,
    fallback_top: int = 3,
) -> str:
    """Same as :func:`render_sources_footer` but for serialized provenance dicts.

    Used on the cache-hit path, where provenance comes back from Redis as the
    output of :meth:`ProvenanceItem.to_dict`.
    """
    if not provenance:
        return ""

    by_label = {
        str(item.get("label")): item for item in provenance if item.get("label")
    }
    labels = [lbl for lbl in referenced_labels(answer_text) if lbl in by_label]

    if labels:
        items = [by_label[lbl] for lbl in labels[:max_items]]
    else:
        items = sorted(provenance, key=lambda p: p.get("rank", 0))[:fallback_top]

    if not items:
        return ""

    title = "📚 **Nguồn:**" if language == "vi" else "📚 **Sources:**"
    lines = [title]
    for item in items:
        label = item.get("label") or str(item.get("rank", ""))
        pointer = (
            item.get("url")
            or item.get("locator")
            or item.get("source")
            or item.get("source_id")
            or ""
        )
        line = f"[{label}] {pointer}"
        excerpt = item.get("excerpt")
        if excerpt:
            line += f' — "{excerpt}"'
        lines.append(line)
    return "\n".join(lines)
