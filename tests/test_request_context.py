from src.observability.prompts import prompt_version
from src.observability.request_context import RequestContext


def test_request_context_generates_ids():
    ctx = RequestContext.new(channel_id="c1", user_id="u1", source="discord")
    assert ctx.query_id
    assert ctx.request_id
    assert ctx.channel_id == "c1"
    assert ctx.user_id == "u1"


def test_prompt_version_is_stable_for_same_text():
    assert prompt_version("abc") == prompt_version("abc")
    assert prompt_version("abc") != prompt_version("abcd")
