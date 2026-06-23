import pytest


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_query_returns_answer_query_id_and_metrics(async_client):
    response = await async_client.post(
        "/api/query",
        headers={"X-API-Key": "test-api-key-with-at-least-32chars"},
        json={
            "query": "What is an axe?",
            "domain": "pz",
            "user_id": "u1",
            "channel_id": "c1",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query_id"]
    assert "response" in data
    assert "metrics" in data
