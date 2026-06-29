from knowledge.redaction import redact_sensitive_text


def test_redacts_email_phone_token_and_discord_id():
    text = (
        "Mail a@b.com phone +84 912 345 678 "
        "token sk-abcdefghijklmnop user <@123456789012345678>"
    )
    result = redact_sensitive_text(text)
    assert "a@b.com" not in result.text
    assert "912 345 678" not in result.text
    assert "sk-abcdefghijklmnop" not in result.text
    assert "123456789012345678" not in result.text
    assert result.redaction_count == 4


def test_safe_game_content_is_unchanged():
    result = redact_sensitive_text("Axe has high tree damage.")
    assert result.text == "Axe has high tree damage."
    assert result.redaction_count == 0
