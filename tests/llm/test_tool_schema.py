"""Tool-schema conversion (fix for 'function is not JSON serializable')."""

import json

from src.llm.tool_schema import to_openai_tool_schema, functions_to_openai_tools


def sample(query: str, limit: int = 10, names: list = None):
    """Search for things.

    Args:
        query: the search text
        limit: max results
    """
    return "x"


def test_schema_shape_types_and_required():
    schema = to_openai_tool_schema(sample)
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "sample"
    assert "Search for things." in fn["description"]

    props = fn["parameters"]["properties"]
    assert props["query"]["type"] == "string"
    assert props["limit"]["type"] == "integer"
    assert props["names"]["type"] == "array"
    assert props["names"]["items"] == {"type": "string"}
    assert "search text" in props["query"]["description"]
    assert fn["parameters"]["required"] == ["query"]


def test_dict_schemas_pass_through():
    existing = {"type": "function", "function": {"name": "x"}}
    assert functions_to_openai_tools([existing]) == [existing]


def test_real_pz_tools_are_json_serializable():
    # This is the exact failure the bug produced: json.dumps(raw functions)
    # raised "Object of type function is not JSON serializable".
    from knowledge.structured.pz_tools import PZ_TOOLS

    schemas = functions_to_openai_tools(PZ_TOOLS)
    assert len(schemas) == len(PZ_TOOLS)
    # Must be serializable now (no function objects left).
    json.dumps(schemas)
    names = {s["function"]["name"] for s in schemas}
    assert {"search_items", "compare_items", "get_recipe"} <= names
