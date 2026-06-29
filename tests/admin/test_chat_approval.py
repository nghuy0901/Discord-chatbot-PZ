import pytest

from admin.chat_approval import ChatApprovalService


class FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self):
        self.calls = []

    def transaction(self):
        return FakeTransaction()

    async def execute(self, sql, *args):
        self.calls.append((sql, args))


class FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self):
        self.connection = FakeConnection()

    def acquire(self):
        return FakeAcquire(self.connection)


class FakeBM25:
    def __init__(self):
        self.refresh_count = 0

    async def refresh_from_db(self):
        self.refresh_count += 1


@pytest.mark.asyncio
async def test_approve_redacts_then_marks_pgvector_metadata():
    pool = FakePool()
    bm25 = FakeBM25()
    service = ChatApprovalService(pool=pool, chat_bm25=bm25)
    result = await service.approve(
        message_id="m1",
        channel_id="c1",
        content="Contact a@b.com about generator rules.",
        approved_by="admin1",
        notes="verified",
    )
    assert result.redacted_content == "Contact [REDACTED_EMAIL] about generator rules."
    assert result.approval_status == "approved"
    update_sql = pool.connection.calls[-1][0]
    assert "'approval_status', 'approved'" in update_sql
    assert "'trusted', true" in update_sql
    assert bm25.refresh_count == 1
