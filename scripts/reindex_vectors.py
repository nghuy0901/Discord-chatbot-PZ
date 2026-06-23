import argparse
import asyncio


async def main():
    parser = argparse.ArgumentParser(description="Reindex NomNom vector collections after embedding changes.")
    parser.add_argument("--collection", required=True, choices=["discord_messages", "knowledge_base"])
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != "REINDEX":
        raise SystemExit("Pass --confirm REINDEX to run.")
    raise SystemExit("Reindex command scaffolded; wire to ingestion source before production rollout.")


if __name__ == "__main__":
    asyncio.run(main())
