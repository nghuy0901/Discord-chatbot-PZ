import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from knowledge.redaction import redact_sensitive_text


@dataclass(frozen=True)
class ApprovalResult:
    message_id: str
    approval_status: str
    redacted_content: str
    redaction_count: int


class ChatApprovalService:
    def __init__(self, pool: Optional[Any] = None, chat_bm25: Optional[Any] = None):
        self._pool = pool
        self._chat_bm25 = chat_bm25

    async def _get_pool(self):
        if self._pool is not None:
            return self._pool
        from rag.db import get_pool

        return await get_pool()

    def _get_chat_bm25(self):
        if self._chat_bm25 is not None:
            return self._chat_bm25
        from rag.bm25_search import get_chat_bm25

        return get_chat_bm25()

    async def _refresh_bm25(self) -> None:
        await self._get_chat_bm25().refresh_from_db()

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
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
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
                await conn.execute(
                    """
                    UPDATE langchain_pg_embedding
                    SET document = $2,
                        cmetadata = cmetadata || jsonb_build_object(
                          'approval_status', 'approved',
                          'source_kind', 'approved_chat',
                          'trusted', true
                        )
                    WHERE cmetadata->>'message_id' = $1
                    """,
                    message_id,
                    redacted.text,
                )
        await self._refresh_bm25()
        return ApprovalResult(
            message_id=message_id,
            approval_status="approved",
            redacted_content=redacted.text,
            redaction_count=redacted.redaction_count,
        )

    async def revoke(self, *, message_id: str, approved_by: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
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
                await conn.execute(
                    """
                    UPDATE langchain_pg_embedding
                    SET cmetadata = cmetadata || jsonb_build_object(
                      'approval_status', 'revoked',
                      'trusted', false
                    )
                    WHERE cmetadata->>'message_id' = $1
                    """,
                    message_id,
                )
        await self._refresh_bm25()
