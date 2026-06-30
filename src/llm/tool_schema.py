"""
Convert plain Python callables into OpenAI-style JSON tool schemas.

The PZ tools are declared as bare functions with Google-style docstrings (an
Ollama-native convention). OpenAI / OpenAI-compatible / vLLM providers, and
Ollama's own /api/chat endpoint, require JSON tool *schemas* — passing the raw
function objects makes the request body unserializable ("Object of type
function is not JSON serializable").

This module introspects a function's signature + docstring and emits the schema
those providers expect. It is provider-neutral and has no project dependencies.
"""

import inspect
import re
from typing import Any, Callable, List

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> str:
    if annotation in _PY_TO_JSON:
        return _PY_TO_JSON[annotation]
    # typing.Optional[X] / List[X] etc. — best-effort by name.
    name = str(annotation).lower()
    if "list" in name:
        return "array"
    if "int" in name:
        return "integer"
    if "float" in name:
        return "number"
    if "bool" in name:
        return "boolean"
    if "dict" in name:
        return "object"
    return "string"


def _summary(doc: str, fallback: str) -> str:
    if not doc:
        return fallback
    # Everything before the "Args:" section, collapsed to a short blurb.
    head = re.split(r"\n\s*Args:", doc, maxsplit=1)[0]
    return " ".join(head.split()).strip()[:1024] or fallback


def _arg_description(doc: str, name: str) -> str:
    if not doc:
        return ""
    match = re.search(rf"^\s*{re.escape(name)}\s*:\s*(.+)$", doc, re.MULTILINE)
    if not match:
        return ""
    return " ".join(match.group(1).split())[:300]


def to_openai_tool_schema(fn: Callable) -> dict:
    """Build a single OpenAI ``tools=[...]`` entry from a callable."""
    doc = inspect.getdoc(fn) or ""
    signature = inspect.signature(fn)

    properties: dict[str, Any] = {}
    required: List[str] = []
    for name, param in signature.parameters.items():
        if name in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = (
            param.annotation
            if param.annotation is not inspect.Parameter.empty
            else str
        )
        prop: dict[str, Any] = {"type": _json_type(annotation)}
        if prop["type"] == "array":
            prop["items"] = {"type": "string"}
        desc = _arg_description(doc, name)
        if desc:
            prop["description"] = desc
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": _summary(doc, fn.__name__),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def functions_to_openai_tools(tools: List[Any]) -> List[dict]:
    """Convert a list of tools to OpenAI schemas.

    Items that are already dict schemas are passed through unchanged, so callers
    may mix callables and pre-built schemas.
    """
    schemas: List[dict] = []
    for tool in tools or []:
        if callable(tool):
            schemas.append(to_openai_tool_schema(tool))
        elif isinstance(tool, dict):
            schemas.append(tool)
    return schemas
