"""Tests for the Python-callable -> JSON tool-schema converter.

The bug this guards against: PZ tools were exported as raw function objects and
passed straight to the provider, producing
"Object of type function is not JSON serializable".
"""

import json

import pytest

from knowledge.structured.pz_tools import (
    PZ_TOOLS,
    PZ_TOOL_FUNCTIONS,
    search_items,
    compare_items,
)
from src.llm.tool_schema import to_openai_tool_schema, functions_to_openai_tools


def test_raw_functions_are_not_json_serializable():
    # Sanity check: this is exactly the crash we are fixing.
    with pytest.raises(TypeError):
        json.dumps(PZ_TOOLS)


def test_all_pz_tools_convert_to_json_serializable_schemas():
    schemas = functions_to_openai_tools(PZ_TOOLS)
    # Must serialize cleanly — the whole point of the converter.
    blob = json.dumps(schemas)
    assert blob
    assert len(schemas) == len(PZ_TOOLS)
    names = {s["function"]["name"] for s in schemas}
    assert names == set(PZ_TOOL_FUNCTIONS.keys())
    for s in schemas:
        assert s["type"] == "function"
        params = s["function"]["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        assert isinstance(params["required"], list)


def test_optional_params_are_not_required():
    schema = to_openai_tool_schema(search_items)["function"]
    # Every search_items param has a default, so nothing is required.
    assert schema["parameters"]["required"] == []
    props = schema["parameters"]["properties"]
    assert props["limit"]["type"] == "integer"
    assert props["sort_by"]["type"] == "string"
    assert props["min_value"]["type"] == "number"
    # Arg descriptions are pulled from the Google-style docstring.
    assert "description" in props["sort_by"]
    # Summary is the docstring head, not empty.
    assert schema["description"]


def test_required_params_and_array_type():
    schema = to_openai_tool_schema(compare_items)["function"]
    # `names` has no default -> required; it is a list -> array with items.
    assert "names" in schema["parameters"]["required"]
    names_prop = schema["parameters"]["properties"]["names"]
    assert names_prop["type"] == "array"
    assert names_prop["items"] == {"type": "string"}


def test_passthrough_of_prebuilt_dict_schemas():
    prebuilt = {"type": "function", "function": {"name": "x", "parameters": {}}}
    out = functions_to_openai_tools([search_items, prebuilt])
    assert out[1] is prebuilt
