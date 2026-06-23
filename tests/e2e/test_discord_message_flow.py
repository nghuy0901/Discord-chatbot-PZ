import pytest


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_discord_message_flow_builds_prompt_and_records_metric(
    mock_discord_message,
    monkeypatch,
):
    from src.aclient import discordClient

    await discordClient._generate_and_send(mock_discord_message, "NomNom, which axe is strong?")

    channel_id = str(mock_discord_message.channel.id)
    assert discordClient.context_manager.get_last_query_id(channel_id)
