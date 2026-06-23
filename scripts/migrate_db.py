import asyncio
import os
from pathlib import Path

import asyncpg


MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


async def migrate() -> None:
    postgres_url = os.environ["POSTGRES_URL"]
    conn = await asyncpg.connect(postgres_url)
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              version TEXT PRIMARY KEY,
              applied_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = path.stem
            already = await conn.fetchval(
                "SELECT 1 FROM schema_migrations WHERE version = $1",
                version,
            )
            if already:
                continue
            async with conn.transaction():
                await conn.execute(path.read_text(encoding="utf-8"))
                await conn.execute(
                    "INSERT INTO schema_migrations(version) VALUES ($1)",
                    version,
                )
            print(f"applied {version}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
