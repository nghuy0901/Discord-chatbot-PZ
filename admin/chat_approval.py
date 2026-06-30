import hashlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Sequence

from knowledge.redaction import redact_sensitive_text


class BM25RefreshError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApprovalResult:
    message_id: str
    approval_status: str
    redacted_content: str
    redaction_count: int


class ChatApprovalService:
    def __init__(
        self,
        pool: Optional[Any] = None,
        embed_documents: Optional[
            Callable[
                [Sequence[str]],
                Awaitable[Sequence[Sequence[float]]],
            ]
        ] = None,
        bm25_refresher: Optional[Callable[[], Awaitable[int]]] = None,
        bm25_invalidator: Optional[Callable[[], None]] = None,
    ):
        self._pool = pool
        self._embed_documents = embed_documents
        self._bm25_refresher = bm25_refresher
        self._bm25_invalidator = bm25_invalidator

    async def _get_pool(self):
        if self._pool is not None:
            return self._pool
        from rag.db import get_pool

        return await get_pool()

    async def _generate_embedding(self, text: str) -> Sequence[float]:
        if self._embed_documents is not None:
            vectors = await self._embed_documents([text])
        else:
            from rag.embedder import embed_documents

            vectors = await embed_documents([text])
        if len(vectors) != 1:
            raise RuntimeError(
                "Document embedding provider must return exactly one vector"
            )
        return vectors[0]

    def _invalidate_bm25(self) -> None:
        if self._bm25_invalidator is not None:
            self._bm25_invalidator()
            return
        from rag.bm25_search import get_chat_bm25

        get_chat_bm25().clear()

    async def _refresh_bm25(self, *, require_documents: bool) -> int:
        try:
            if self._bm25_refresher is not None:
                count = await self._bm25_refresher()
            else:
                from rag.bm25_search import get_chat_bm25
                from rag.db import COLLECTION_NAME

                count = await get_chat_bm25().refresh_from_db(
                    collection_name=COLLECTION_NAME,
                    raise_on_error=True,
                )
        except Exception as exc:
            self._invalidate_bm25()
            raise BM25RefreshError("BM25 refresh failed") from exc

        if require_documents and count < 1:
            self._invalidate_bm25()
            raise BM25RefreshError("BM25 refresh returned no chat documents")
        return count

    @staticmethod
    def _vector_literal(vector: Sequence[float]) -> str:
        if not vector:
            raise RuntimeError("Embedding provider returned an empty vector")
        return "[" + ",".join(format(float(value), ".17g") for value in vector) + "]"

    @staticmethod
    def _require_exactly_one(command_tag: str, entity: str) -> None:
        parts = command_tag.split()
        if len(parts) != 2 or parts[0] != "UPDATE":
            raise RuntimeError(f"Unexpected {entity} update command tag")
        try:
            affected_rows = int(parts[1])
        except ValueError as exc:
            raise RuntimeError(f"Unexpected {entity} update command tag") from exc
        if affected_rows != 1:
            raise RuntimeError(f"Approval operation requires exactly one {entity} row")

    async def approve(
        self,
        *,
        message_id: str,
        channel_id: str,
        content: str,
        approved_by: str,
        notes: str = "",
    ) -> ApprovalResult:
        redacted = redact_sensitive_text(content)
        digest = hashlib.sha256(redacted.text.encode("utf-8")).hexdigest()
        embedding = self._vector_literal(
            await self._generate_embedding(redacted.text)
        )
        from rag.db import COLLECTION_NAME

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                command_tag = await conn.execute(
                    """
                    UPDATE langchain_pg_embedding AS e
                    SET document = $2,
                        embedding = $3::vector,
                        cmetadata = e.cmetadata || jsonb_build_object(
                          'approval_status', 'approved',
                          'source_kind', 'approved_chat',
                          'trusted', true
                        )
                    FROM langchain_pg_collection AS c
                    WHERE e.collection_id = c.uuid
                      AND e.cmetadata->>'message_id' = $1
                      AND c.name = $4
                      AND e.cmetadata->>'source_kind' IN (
                        'discord_chat', 'approved_chat'
                      )
                    """,
                    message_id,
                    redacted.text,
                    embedding,
                    COLLECTION_NAME,
                )
                self._require_exactly_one(command_tag, "chat embedding")
                await conn.execute(
                    """
                    INSERT INTO approved_chat_sources (
                      message_id, channel_id, approved_by, approval_status,
                      redacted_content_hash, notes
                    )
                    VALUES ($1, $2, $3, 'approved', $4, $5)
                    ON CONFLICT (message_id) DO UPDATE SET
                      channel_id = EXCLUDED.channel_id,
                      approved_by = EXCLUDED.approved_by,
                      approved_at = NOW(),
                      approval_status = 'approved',
                      redacted_content_hash = EXCLUDED.redacted_content_hash,
                      notes = EXCLUDED.notes
                    """,
                    message_id,
                    channel_id,
                    approved_by,
                    digest,
                    notes,
                )

        await self._refresh_bm25(require_documents=True)
        return ApprovalResult(
            message_id=message_id,
            approval_status="approved",
            redacted_content=redacted.text,
            redaction_count=redacted.redaction_count,
        )

    async def revoke(
        self,
        *,
        message_id: str,
        approved_by: str,
    ) -> ApprovalResult:
        from rag.db import COLLECTION_NAME

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                command_tag = await conn.execute(
                    """
                    UPDATE langchain_pg_embedding AS e
                    SET cmetadata = e.cmetadata || jsonb_build_object(
                      'approval_status', 'revoked',
                      'trusted', false
                    )
                    FROM langchain_pg_collection AS c
                    WHERE e.collection_id = c.uuid
                      AND e.cmetadata->>'message_id' = $1
                      AND c.name = $2
                      AND e.cmetadata->>'source_kind' IN (
                        'discord_chat', 'approved_chat'
                      )
                    """,
                    message_id,
                    COLLECTION_NAME,
                )
                self._require_exactly_one(command_tag, "chat embedding")
                source_command_tag = await conn.execute(
                    """
                    UPDATE approved_chat_sources
                    SET approval_status = 'revoked',
                        approved_by = $2,
                        approved_at = NOW()
                    WHERE message_id = $1
                    """,
                    message_id,
                    approved_by,
                )
                self._require_exactly_one(
                    source_command_tag,
                    "approved chat source",
                )

        await self._refresh_bm25(require_documents=False)
        return ApprovalResult(
            message_id=message_id,
            approval_status="revoked",
            redacted_content="",
            redaction_count=0,
        )
