import argparse
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from admin.chat_approval import ChatApprovalService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Approve or revoke a Discord chat message as a trusted RAG source."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    approve = subcommands.add_parser("approve")
    approve.add_argument("--message-id", required=True)
    approve.add_argument("--channel-id", required=True)
    approve.add_argument("--content", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--notes", default="")

    revoke = subcommands.add_parser("revoke")
    revoke.add_argument("--message-id", required=True)
    revoke.add_argument("--approved-by", required=True)

    return parser


async def main() -> None:
    args = build_parser().parse_args()
    service = ChatApprovalService()

    if args.command == "approve":
        result = await service.approve(
            message_id=args.message_id,
            channel_id=args.channel_id,
            content=args.content,
            approved_by=args.approved_by,
            notes=args.notes,
        )
        print(
            "approved "
            f"message_id={result.message_id} "
            f"redactions={result.redaction_count}"
        )
        return

    await service.revoke(message_id=args.message_id, approved_by=args.approved_by)
    print(f"revoked message_id={args.message_id}")


if __name__ == "__main__":
    asyncio.run(main())
