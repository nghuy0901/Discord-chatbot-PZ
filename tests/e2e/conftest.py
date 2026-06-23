from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient


TEST_API_KEY = "test-api-key-with-at-least-32chars"


class _DummyAcquire:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, *args, **kwargs):
        return None


class _DummyPool:
    def acquire(self):
        return _DummyAcquire()


class _DummyVectorStore:
    def __init__(self):
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)


@pytest.fixture(autouse=True)
def e2e_external_service_mocks(monkeypatch):
    monkeypatch.setenv("API_KEY", TEST_API_KEY)

    async def fake_get_pool():
        return _DummyPool()

    import rag.db

    monkeypatch.setattr(rag.db, "get_pool", fake_get_pool)

    from knowledge.manager import KnowledgeManager

    monkeypatch.setattr(
        KnowledgeManager,
        "_get_vectorstore",
        lambda self: _DummyVectorStore(),
    )


@pytest.fixture
async def async_client(monkeypatch):
    import api.main as api_main

    class DummyMetricsManager:
        async def record(self, metric):
            return None

        async def update_db(self, metric):
            return None

        async def get_db_summary(self):
            return {"total_queries": 1, "errors": 0}

    async def fake_chat_completion(**kwargs):
        return "Axe is a weapon."

    monkeypatch.setattr(api_main, "ENABLE_RAG", False)
    monkeypatch.setattr(api_main, "ENABLE_TOOL_CALLING", False)
    monkeypatch.setattr(api_main, "get_metrics_manager", lambda: DummyMetricsManager())
    monkeypatch.setattr(api_main, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(
        api_main,
        "get_last_token_usage",
        lambda: {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    )
    monkeypatch.setattr(
        api_main,
        "api_rate_limiter",
        api_main.InMemoryRateLimiter(limit=100, window_seconds=60),
    )

    transport = ASGITransport(app=api_main.app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


class _DummyTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummyBotMessage:
    def __init__(self, content):
        self.id = 9001
        self.content = content

    async def add_reaction(self, emoji):
        return None

    async def edit(self, content):
        self.content = content


class _DummyChannel:
    id = "e2e_channel"
    name = "pz"

    def __init__(self):
        self.sent = []

    def typing(self):
        return _DummyTyping()

    async def send(self, content):
        message = _DummyBotMessage(content)
        self.sent.append(message)
        return message


class _DummyMessage:
    id = "e2e_message"
    guild = None

    def __init__(self):
        self.channel = _DummyChannel()
        self.author = SimpleNamespace(id="e2e_user", display_name="E2E User")

    async def reply(self, content, mention_author=False):
        return await self.channel.send(content)


@pytest.fixture
def mock_discord_message(monkeypatch):
    import src.aclient as aclient
    from utils.context_manager import ContextManager

    manager = ContextManager()

    async def fake_build_prompt(
        channel_id,
        user_message,
        user_name,
        enable_rag=True,
        channel_name=None,
        user_id=None,
        request_context=None,
    ):
        manager.set_last_query_id(channel_id, "e2e-query-id")
        return [{"role": "user", "content": user_message}], 0.1, "conversation"

    async def fake_chat_completion(**kwargs):
        return "Axe is a strong melee weapon."

    manager.build_prompt = fake_build_prompt
    monkeypatch.setattr(aclient.discordClient, "context_manager", manager)
    monkeypatch.setattr(aclient.discordClient._connection, "user", SimpleNamespace(id=999, display_name="NomNom"))
    monkeypatch.setattr(aclient, "ENABLE_RAG", False)
    monkeypatch.setattr(aclient, "ENABLE_STREAMING", False)
    monkeypatch.setattr(aclient, "ENABLE_FEEDBACK", False)
    monkeypatch.setattr(aclient, "ENABLE_TOOL_CALLING", False)
    monkeypatch.setattr(aclient, "chat_completion", fake_chat_completion)

    return _DummyMessage()
