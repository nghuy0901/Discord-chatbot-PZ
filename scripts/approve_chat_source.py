import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from admin.chat_approval import ChatApprovalService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approve or revoke a chat source.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    approve = subparsers.add_parser("approve")
    approve.add_argument("--message-id", required=True)
    approve.add_argument("--channel-id", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--content-file", required=True, type=Path)
    approve.add_argument("--notes", default="")

    revoke = subparsers.add_parser("revoke")
    revoke.add_argument("--message-id", required=True)
    revoke.add_argument("--approved-by", required=True)
    return parser


async def run(
    argv: Optional[Sequence[str]] = None,
    *,
    service: Optional[ChatApprovalService] = None,
) -> int:
    args = build_parser().parse_args(argv)
    approval_service = service or ChatApprovalService()

    try:
        if args.command == "approve":
            content = args.content_file.read_text(encoding="utf-8")
            result = await approval_service.approve(
                message_id=args.message_id,
                channel_id=args.channel_id,
                content=content,
                approved_by=args.approved_by,
                notes=args.notes,
            )
            payload = {
                "message_id": result.message_id,
                "approval_status": result.approval_status,
                "redacted_content": result.redacted_content,
                "redaction_count": result.redaction_count,
            }
        else:
            result = await approval_service.revoke(
                message_id=args.message_id,
                approved_by=args.approved_by,
            )
            payload = {
                "message_id": result.message_id,
                "approval_status": result.approval_status,
                "redaction_count": result.redaction_count,
            }
    except Exception:
        print(json.dumps({"error": "chat approval failed"}))
        return 1

    print(json.dumps(payload, ensure_ascii=False))
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
