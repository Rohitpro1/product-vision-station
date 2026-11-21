#!/usr/bin/env python3
"""Apply PostgreSQL migrations in order (psql), tracked in schema_migrations.

  PVS_PG_DSN=postgresql://user@host:5432/pvs python scripts/migrate.py [--check]

--check verifies every migration file's recorded checksum (detects edited, already-applied migrations).
Restore: pg_dump -Fc "$PVS_PG_DSN" > pvs.dump ; pg_restore --clean -d "$PVS_PG_DSN" pvs.dump
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "migrations" / "postgres"


def psql(dsn: str, sql: str) -> str:
    return subprocess.run(["psql", dsn, "-v", "ON_ERROR_STOP=1", "-qAt", "-c", sql], check=True, capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    dsn = os.environ.get("PVS_PG_DSN")
    if not dsn:
        print("PVS_PG_DSN not set: skipped (no PostgreSQL available)", file=sys.stderr)
        return 2
    psql(dsn, "CREATE TABLE IF NOT EXISTS schema_migrations(name text PRIMARY KEY, sha256 text NOT NULL, "
              "applied_at timestamptz NOT NULL DEFAULT now())")
    applied = dict(line.split("|") for line in psql(dsn, "SELECT name, sha256 FROM schema_migrations").splitlines() if line)
    rc = 0
    for f in sorted(ROOT.glob("*.sql")):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if f.name in applied:
            if applied[f.name] != h:
                print(f"CHECKSUM MISMATCH {f.name}")
                rc = 1
            continue
        if "--check" in sys.argv:
            print(f"PENDING {f.name}")
            rc = 1
            continue
        sql = f.read_text() + f"\nINSERT INTO schema_migrations(name, sha256) VALUES ('{f.name}', '{h}');"
        subprocess.run(["psql", dsn, "-v", "ON_ERROR_STOP=1", "-q", "-1", "-c", sql], check=True)
        print(f"applied {f.name}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
