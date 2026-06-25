import pytest

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


@pytest.mark.parametrize(
    "phone",
    [
        "0912 345 678",
        "+84 912-345-678",
        "+1 (415) 555-2671",
        "+44 20 7946 0958",
    ],
)
def test_redacts_vietnamese_and_international_phone_numbers(phone):
    result = redact_sensitive_text(f"Call {phone} for help")

    assert phone not in result.text
    assert result.redaction_count == 1


@pytest.mark.parametrize(
    "token",
    [
        "sk-abcdefghijklmnop",
        "ghp_abcdefghijklmnopqrstuvwxyz123456",
        "ghp-abcdefghijklmnopqrst",
        "glpat-abcdefghijklmnopqrst",
        "xoxb-123456789012-abcdefghijklmnop",
        "AKIAIOSFODNN7EXAMPLE",
        "sk_live_abcdefghijklmnopqrstuvwxyz",
        "github_pat_11ABCDEFG_abcdefghijklmnopqrstuvwxyz",
        "AIzaSyA1234567890abcdefghijklmnop",
        (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6Ik5vbU5vbSJ9."
            "abcdefghijklmnopqrstuvwxyz123456"
        ),
        "Bearer abcdefghijklmnopqrstuvwxyz012345",
    ],
)
def test_redacts_common_api_token_prefixes(token):
    result = redact_sensitive_text(f"Secret: {token}")

    assert token not in result.text
    assert result.redaction_count == 1


@pytest.mark.parametrize(
    "mention",
    [
        "<@123456789012345678>",
        "<@!123456789012345678>",
        "<@&123456789012345678>",
        "<#123456789012345678>",
    ],
)
def test_redacts_discord_mentions(mention):
    result = redact_sensitive_text(f"Ping {mention}")

    assert "123456789012345678" not in result.text
    assert result.redaction_count == 1


def test_safe_game_content_is_unchanged():
    text = "Axe has high tree damage. Use channel 12 and item 415."

    result = redact_sensitive_text(text)

    assert result.text == text
    assert result.redaction_count == 0
