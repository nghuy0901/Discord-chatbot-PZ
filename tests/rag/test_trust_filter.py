from rag.trust import filter_trusted_results, is_trusted_result, trusted_chat_filter


def test_allows_only_release_scope_knowledge_domains():
    assert is_trusted_result(
        {"content_type": "knowledge_base", "domain": "pz", "trusted": True}
    )
    assert is_trusted_result(
        {"content_type": "knowledge_base", "domain": "server_rules", "trusted": True}
    )
    assert not is_trusted_result(
        {"content_type": "knowledge_base", "domain": "general", "trusted": True}
    )


def test_allows_only_approved_chat_sources():
    assert is_trusted_result(
        {
            "source_kind": "approved_chat",
            "approval_status": "approved",
            "trusted": True,
        }
    )
    assert not is_trusted_result(
        {
            "source_kind": "chat",
            "approval_status": "unapproved",
            "trusted": False,
        }
    )


def test_filter_trusted_results_drops_unapproved_and_out_of_scope_sources():
    results = [
        {"content": "rule", "content_type": "knowledge_base", "domain": "server_rules", "trusted": True},
        {"content": "profile", "content_type": "knowledge_base", "domain": "general", "trusted": True},
        {"content": "chat", "source_kind": "approved_chat", "approval_status": "approved", "trusted": True},
        {"content": "rumor", "source_kind": "chat", "approval_status": "unapproved", "trusted": False},
    ]

    filtered = filter_trusted_results(results)

    assert [item["content"] for item in filtered] == ["rule", "chat"]


def test_trusted_chat_filter_preserves_channel_constraint():
    assert trusted_chat_filter("c1") == {
        "channel_id": "c1",
        "approval_status": "approved",
        "trusted": True,
    }
