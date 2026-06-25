import json

import pytest

from admin.chat_approval import BM25RefreshError, ChatApprovalService
from knowledge.redaction import redact_sensitive_text
from rag.bm25_search import BM25Index
from rag.db import COLLECTION_NAME, _msg_to_document
from rag.embedder import embed_documents
from scripts.approve_chat_source import run


class FakeTransaction:
    def __init__(self):
        self.entered = False
        self.exited = False
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        self.committed = exc_type is None
        self.rolled_back = exc_type is not None
        return False


class FakeConnection:
    def __init__(self, command_tags=None):
        self.calls = []
        self.transactions = []
        self.command_tags = list(command_tags or [])

    def transaction(self):
        transaction = FakeTransaction()
        self.transactions.append(transaction)
        return transaction

    async def execute(self, sql, *args):
        self.calls.append((sql, args))
        if self.command_tags:
            return self.command_tags.pop(0)
        return "UPDATE 1"


class FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, command_tags=None):
        self.connection = FakeConnection(command_tags=command_tags)

    def acquire(self):
        return FakeAcquire(self.connection)


class FakeRefresher:
    def __init__(self, count=1, error=None):
        self.count = count
        self.error = error
        self.calls = []

    async def __call__(self):
        self.calls.append(True)
        if self.error is not None:
            raise self.error
        return self.count


class FakeEmbedder:
    def __init__(self, vector=None):
        self.vector = vector or [0.25, 0.75]
        self.batches = []

    async def __call__(self, texts):
        self.batches.append(texts)
        return [self.vector]


class FakeInvalidator:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1


@pytest.mark.asyncio
async def test_approve_upserts_redacted_chat_and_refreshes_bm25():
    pool = FakePool(command_tags=["UPDATE 1", "INSERT 0 1"])
    embedder = FakeEmbedder()
    refresher = FakeRefresher(count=1)
    service = ChatApprovalService(
        pool=pool,
        embed_documents=embedder,
        bm25_refresher=refresher,
    )

    result = await service.approve(
        message_id="m1",
        channel_id="c1",
        content="Contact a@b.com about generator rules.",
        approved_by="admin1",
        notes="verified",
    )

    assert result.redacted_content == (
        "Contact [REDACTED_EMAIL] about generator rules."
    )
    assert result.approval_status == "approved"
    assert len(pool.connection.calls) == 2
    update_sql, update_args = pool.connection.calls[0]
    upsert_sql, upsert_args = pool.connection.calls[1]
    assert embedder.batches == [[result.redacted_content]]
    assert "SET document = $2" in update_sql
    assert "embedding = $3::vector" in update_sql
    assert "FROM langchain_pg_collection" in update_sql
    assert "c.name = $4" in update_sql
    assert "source_kind" in update_sql
    assert "'discord_chat', 'approved_chat'" in update_sql
    assert "'approval_status', 'approved'" in update_sql
    assert "'source_kind', 'approved_chat'" in update_sql
    assert "'trusted', true" in update_sql
    assert update_args == (
        "m1",
        result.redacted_content,
        "[0.25,0.75]",
        COLLECTION_NAME,
    )
    assert "INSERT INTO approved_chat_sources" in upsert_sql
    assert "ON CONFLICT (message_id) DO UPDATE" in upsert_sql
    assert upsert_args[:3] == ("m1", "c1", "admin1")
    assert all(tx.entered and tx.exited for tx in pool.connection.transactions)
    assert all(tx.committed for tx in pool.connection.transactions)
    assert len(refresher.calls) == 1


@pytest.mark.asyncio
async def test_revoke_marks_source_and_vector_untrusted_then_refreshes_bm25():
    pool = FakePool(command_tags=["UPDATE 1", "UPDATE 1"])
    refresher = FakeRefresher(count=0)
    service = ChatApprovalService(pool=pool, bm25_refresher=refresher)

    result = await service.revoke(message_id="m1", approved_by="admin2")

    assert result.message_id == "m1"
    assert result.approval_status == "revoked"
    assert result.redacted_content == ""
    assert len(pool.connection.calls) == 2
    vector_sql, vector_args = pool.connection.calls[0]
    source_sql, source_args = pool.connection.calls[1]
    assert "FROM langchain_pg_collection" in vector_sql
    assert "c.name = $2" in vector_sql
    assert "source_kind" in vector_sql
    assert "'discord_chat', 'approved_chat'" in vector_sql
    assert "'approval_status', 'revoked'" in vector_sql
    assert "'trusted', false" in vector_sql
    assert vector_args == ("m1", COLLECTION_NAME)
    assert "UPDATE approved_chat_sources" in source_sql
    assert "approval_status = 'revoked'" in source_sql
    assert source_args == ("m1", "admin2")
    assert len(refresher.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("command_tag", ["UPDATE 0", "UPDATE 2"])
async def test_approve_requires_exactly_one_scoped_embedding_row(command_tag):
    pool = FakePool(command_tags=[command_tag])
    refresher = FakeRefresher()
    service = ChatApprovalService(
        pool=pool,
        embed_documents=FakeEmbedder(),
        bm25_refresher=refresher,
    )

    with pytest.raises(RuntimeError, match="exactly one chat embedding"):
        await service.approve(
            message_id="missing",
            channel_id="c1",
            content="safe content",
            approved_by="admin1",
        )

    assert len(pool.connection.calls) == 1
    transaction = pool.connection.transactions[0]
    assert transaction.rolled_back is True
    assert transaction.committed is False
    assert refresher.calls == []


@pytest.mark.asyncio
async def test_revoke_requires_exactly_one_scoped_embedding_row():
    pool = FakePool(command_tags=["UPDATE 0"])
    refresher = FakeRefresher()
    service = ChatApprovalService(pool=pool, bm25_refresher=refresher)

    with pytest.raises(RuntimeError, match="exactly one chat embedding"):
        await service.revoke(message_id="missing", approved_by="admin1")

    assert len(pool.connection.calls) == 1
    assert pool.connection.transactions[0].rolled_back is True
    assert refresher.calls == []


@pytest.mark.asyncio
async def test_approve_reports_refresh_failure_after_database_commit():
    pool = FakePool(command_tags=["UPDATE 1", "INSERT 0 1"])
    refresher = FakeRefresher(error=ConnectionError("database unavailable"))
    service = ChatApprovalService(
        pool=pool,
        embed_documents=FakeEmbedder(),
        bm25_refresher=refresher,
    )

    with pytest.raises(BM25RefreshError, match="BM25 refresh failed"):
        await service.approve(
            message_id="m1",
            channel_id="c1",
            content="safe content",
            approved_by="admin1",
        )

    transaction = pool.connection.transactions[0]
    assert transaction.committed is True
    assert transaction.rolled_back is False
    assert len(pool.connection.calls) == 2


@pytest.mark.asyncio
async def test_approve_rejects_zero_document_bm25_refresh():
    pool = FakePool(command_tags=["UPDATE 1", "INSERT 0 1"])
    service = ChatApprovalService(
        pool=pool,
        embed_documents=FakeEmbedder(),
        bm25_refresher=FakeRefresher(count=0),
    )

    with pytest.raises(BM25RefreshError, match="no chat documents"):
        await service.approve(
            message_id="m1",
            channel_id="c1",
            content="safe content",
            approved_by="admin1",
        )


@pytest.mark.asyncio
async def test_default_embedding_uses_langchain_document_api(monkeypatch):
    class FakeLangChainEmbeddings:
        def __init__(self):
            self.document_batches = []
            self.query_texts = []

        async def aembed_documents(self, texts):
            self.document_batches.append(texts)
            return [[0.1, 0.9]]

        async def aembed_query(self, text):
            self.query_texts.append(text)
            return [9.0, 9.0]

    embeddings = FakeLangChainEmbeddings()
    monkeypatch.setattr("rag.embedder.get_embeddings", lambda: embeddings)

    vector = await ChatApprovalService()._generate_embedding("redacted text")

    assert vector == [0.1, 0.9]
    assert embeddings.document_batches == [["redacted text"]]
    assert embeddings.query_texts == []


@pytest.mark.asyncio
async def test_embed_documents_never_uses_query_embedding(monkeypatch):
    class FakeLangChainEmbeddings:
        def __init__(self):
            self.document_batches = []
            self.query_texts = []

        async def aembed_documents(self, texts):
            self.document_batches.append(texts)
            return [[0.2, 0.8]]

        async def aembed_query(self, text):
            self.query_texts.append(text)
            return [8.0, 8.0]

    embeddings = FakeLangChainEmbeddings()
    monkeypatch.setattr("rag.embedder.get_embeddings", lambda: embeddings)

    vectors = await embed_documents(["redacted document"])

    assert vectors == [[0.2, 0.8]]
    assert embeddings.document_batches == [["redacted document"]]
    assert embeddings.query_texts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["approve", "revoke"])
async def test_refresh_failure_invalidates_chat_bm25(operation):
    pool = FakePool(command_tags=["UPDATE 1", "UPDATE 1"])
    invalidator = FakeInvalidator()
    service = ChatApprovalService(
        pool=pool,
        embed_documents=FakeEmbedder(),
        bm25_refresher=FakeRefresher(
            error=ConnectionError("database unavailable")
        ),
        bm25_invalidator=invalidator,
    )

    with pytest.raises(BM25RefreshError, match="BM25 refresh failed"):
        if operation == "approve":
            await service.approve(
                message_id="m1",
                channel_id="c1",
                content="safe content",
                approved_by="admin1",
            )
        else:
            await service.revoke(message_id="m1", approved_by="admin1")

    assert invalidator.calls == 1


def test_bm25_clear_removes_searchable_documents():
    index = BM25Index()
    index.refresh_from_documents(
        [
            {"content": "generator safety rules", "message_id": "m1"},
            {"content": "farming crop calendar", "message_id": "m2"},
            {"content": "medical bandage guide", "message_id": "m3"},
        ]
    )
    assert index.is_ready is True
    assert index.search("generator", min_score=0) != []

    index.clear()

    assert index.is_ready is False
    assert index.doc_count == 0
    assert index.search("generator", min_score=0) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("approval_tag", ["UPDATE 0", "UPDATE 2"])
async def test_revoke_requires_exactly_one_approval_source_row(approval_tag):
    pool = FakePool(command_tags=["UPDATE 1", approval_tag])
    refresher = FakeRefresher()
    service = ChatApprovalService(pool=pool, bm25_refresher=refresher)

    with pytest.raises(RuntimeError, match="exactly one approved chat source"):
        await service.revoke(message_id="m1", approved_by="admin1")

    transaction = pool.connection.transactions[0]
    assert transaction.rolled_back is True
    assert transaction.committed is False
    assert refresher.calls == []


@pytest.mark.asyncio
async def test_bm25_strict_refresh_reraises_database_error(monkeypatch):
    class FailingPool:
        def acquire(self):
            raise ConnectionError("database unavailable")

    async def fake_get_pool():
        return FailingPool()

    monkeypatch.setattr("rag.db.get_pool", fake_get_pool)

    with pytest.raises(ConnectionError, match="database unavailable"):
        await BM25Index().refresh_from_db(
            collection_name=COLLECTION_NAME,
            raise_on_error=True,
        )


def test_new_chat_document_forces_unapproved_trust_defaults():
    document = _msg_to_document(
        message_id="m1",
        channel_id="c1",
        author_id="u1",
        content="safe",
        timestamp="2026-06-25T00:00:00Z",
        metadata={
            "source_kind": "approved_chat",
            "approval_status": "approved",
            "trusted": True,
            "custom": "kept",
        },
    )

    assert document.metadata["source_kind"] == "discord_chat"
    assert document.metadata["approval_status"] == "unapproved"
    assert document.metadata["trusted"] is False
    assert document.metadata["custom"] == "kept"


class FakeApprovalService:
    def __init__(self):
        self.calls = []

    async def approve(self, **kwargs):
        self.calls.append(("approve", kwargs))
        return type(
            "Result",
            (),
            {
                "message_id": kwargs["message_id"],
                "approval_status": "approved",
                "redacted_content": "Xin chào [REDACTED_EMAIL]",
                "redaction_count": 1,
            },
        )()

    async def revoke(self, **kwargs):
        self.calls.append(("revoke", kwargs))
        return type(
            "Result",
            (),
            {
                "message_id": kwargs["message_id"],
                "approval_status": "revoked",
                "redacted_content": "",
                "redaction_count": 0,
            },
        )()


@pytest.mark.asyncio
async def test_cli_approve_reads_utf8_and_prints_only_redacted_json(tmp_path, capsys):
    content_file = tmp_path / "message.txt"
    original = "Xin chào bí mật@example.com"
    content_file.write_text(original, encoding="utf-8")
    service = FakeApprovalService()

    exit_code = await run(
        [
            "approve",
            "--message-id",
            "m1",
            "--channel-id",
            "c1",
            "--approved-by",
            "admin1",
            "--content-file",
            str(content_file),
        ],
        service=service,
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 0
    assert service.calls[0][1]["content"] == original
    assert payload["redacted_content"] == "Xin chào [REDACTED_EMAIL]"
    assert original not in output
    assert "bí mật@example.com" not in output


@pytest.mark.asyncio
async def test_cli_revoke_prints_json_without_content(capsys):
    service = FakeApprovalService()

    exit_code = await run(
        [
            "revoke",
            "--message-id",
            "m1",
            "--approved-by",
            "admin1",
        ],
        service=service,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {
        "message_id": "m1",
        "approval_status": "revoked",
        "redaction_count": 0,
    }


@pytest.mark.asyncio
async def test_cli_does_not_leak_original_content_when_approval_fails(
    tmp_path, capsys
):
    content_file = tmp_path / "message.txt"
    original = "private người.dùng@example.com"
    content_file.write_text(original, encoding="utf-8")

    class FailingService:
        async def approve(self, **kwargs):
            raise RuntimeError(f"database rejected {kwargs['content']}")

    exit_code = await run(
        [
            "approve",
            "--message-id",
            "m1",
            "--channel-id",
            "c1",
            "--approved-by",
            "admin1",
            "--content-file",
            str(content_file),
        ],
        service=FailingService(),
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert json.loads(output) == {"error": "chat approval failed"}
    assert original not in output
    assert "người.dùng@example.com" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret",
    [
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
async def test_cli_never_prints_supported_secrets(tmp_path, capsys, secret):
    content_file = tmp_path / "message.txt"
    content_file.write_text(f"credential {secret}", encoding="utf-8")

    class RedactingService:
        async def approve(self, **kwargs):
            redacted = redact_sensitive_text(kwargs["content"])
            return type(
                "Result",
                (),
                {
                    "message_id": kwargs["message_id"],
                    "approval_status": "approved",
                    "redacted_content": redacted.text,
                    "redaction_count": redacted.redaction_count,
                },
            )()

    exit_code = await run(
        [
            "approve",
            "--message-id",
            "m1",
            "--channel-id",
            "c1",
            "--approved-by",
            "admin1",
            "--content-file",
            str(content_file),
        ],
        service=RedactingService(),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert secret not in output
    assert "[REDACTED_SECRET]" in json.loads(output)["redacted_content"]
