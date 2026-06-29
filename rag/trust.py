import os
from typing import Any, Dict, Iterable, List, Optional


def trusted_kb_domains() -> set[str]:
    raw = os.getenv("TRUSTED_KB_DOMAINS", "pz,server_rules")
    return {item.strip() for item in raw.split(",") if item.strip()}


def is_trusted_result(result: Dict[str, Any]) -> bool:
    domain = str(result.get("domain") or "")
    content_type = result.get("content_type") or result.get("source_type")
    if content_type == "knowledge_base":
        # Fail closed: a KB chunk missing the ``trusted`` flag is treated as
        # untrusted rather than trusted (audit M3). The ingest path always sets
        # this flag, so well-formed chunks are unaffected.
        return domain in trusted_kb_domains() and bool(result.get("trusted", False))

    return (
        result.get("approval_status") == "approved"
        and result.get("trusted") is True
        and result.get("source_kind") in {"approved_chat", "chat"}
    )


def filter_trusted_results(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [result for result in results if is_trusted_result(result)]


def trusted_chat_filter(channel_id: Optional[str] = None) -> Dict[str, Any]:
    filter_dict: Dict[str, Any] = {
        "approval_status": "approved",
        "trusted": True,
    }
    if channel_id:
        filter_dict["channel_id"] = channel_id
    return filter_dict


def trusted_domains_from(domains: Optional[Iterable[str]]) -> List[str]:
    allowed = trusted_kb_domains()
    if domains is None:
        return sorted(allowed)
    return [domain for domain in domains if domain in allowed]
