"""SQLite persistence for seen and scored jobs.

Tracks which job postings have already been processed (`seen_jobs`) and stores
their scoring results (`scored_jobs`). Standard library only — no dependencies.

Typical pipeline, integrating with scorer.py:

    import database, scorer

    if not database.is_seen(url):
        sid = database.mark_seen(url, title=title, company=company,
                                 location=location, description=description)
        result = scorer.score(description)          # a scorer.Score
        database.save_result({**result.model_dump(), "seen_job_id": sid})

`scorer.Score` (score, reasoning, green_flags, red_flags) maps directly onto the
keys save_result() expects, via `model_dump()`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

DEFAULT_DB_PATH = Path(__file__).with_name("jobs.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url_hash TEXT UNIQUE NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    company TEXT,
    location TEXT,
    description TEXT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    score_status TEXT DEFAULT 'pending'  -- 'pending', 'scored', 'failed'
);

CREATE TABLE IF NOT EXISTS scored_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seen_job_id INTEGER NOT NULL REFERENCES seen_jobs(id),
    score INTEGER NOT NULL,
    reasoning TEXT,
    green_flags TEXT,
    red_flags TEXT,
    notified INTEGER DEFAULT 0,
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _url_hash(url: str) -> str:
    """Stable dedup key for a URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection, enforce foreign keys, and ensure the schema exists."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


@contextmanager
def _connection(conn: Optional[sqlite3.Connection]) -> Iterator[sqlite3.Connection]:
    """Yield a usable connection.

    If `conn` is None we own a short-lived connection (commit + close on exit).
    If the caller passed one, we use it and leave commit/close to them — writes
    are still visible to later reads on the same connection.
    """
    if conn is not None:
        yield conn
        return

    owned = connect()
    try:
        yield owned
        owned.commit()
    finally:
        owned.close()


def init_db(conn: Optional[sqlite3.Connection] = None) -> None:
    """Ensure the schema exists (no-op if it already does)."""
    with _connection(conn) as c:
        c.executescript(SCHEMA)


def is_seen(url: str, conn: Optional[sqlite3.Connection] = None) -> bool:
    """Return True if this URL has already been recorded in seen_jobs."""
    with _connection(conn) as c:
        row = c.execute(
            "SELECT 1 FROM seen_jobs WHERE url_hash = ?", (_url_hash(url),)
        ).fetchone()
        return row is not None


def mark_seen(
    url: str,
    title: Optional[str] = None,
    company: Optional[str] = None,
    location: Optional[str] = None,
    description: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Record a job as seen and return its seen_jobs id.

    Idempotent: calling again with the same URL returns the existing id without
    overwriting the original row's metadata.
    """
    url_hash = _url_hash(url)
    with _connection(conn) as c:
        c.execute(
            """
            INSERT INTO seen_jobs (url_hash, url, title, company, location, description)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(url_hash) DO NOTHING
            """,
            (url_hash, url, title, company, location, description),
        )
        row = c.execute(
            "SELECT id FROM seen_jobs WHERE url_hash = ?", (url_hash,)
        ).fetchone()
        return int(row["id"])


def save_result(
    job_dict: Dict[str, Any], conn: Optional[sqlite3.Connection] = None
) -> int:
    """Persist a scoring result and mark the seen job as scored.

    `job_dict` must contain `seen_job_id` (from mark_seen) plus the score fields
    `score`, `reasoning`, `green_flags`, `red_flags` (the last two as lists, as
    produced by scorer.Score). Returns the new scored_jobs id.
    """
    seen_job_id = job_dict["seen_job_id"]
    green_flags = json.dumps(job_dict.get("green_flags", []))
    red_flags = json.dumps(job_dict.get("red_flags", []))

    with _connection(conn) as c:
        cur = c.execute(
            """
            INSERT INTO scored_jobs (seen_job_id, score, reasoning, green_flags, red_flags)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                seen_job_id,
                job_dict["score"],
                job_dict.get("reasoning"),
                green_flags,
                red_flags,
            ),
        )
        c.execute(
            "UPDATE seen_jobs SET score_status = 'scored' WHERE id = ?",
            (seen_job_id,),
        )
        return int(cur.lastrowid)


def mark_failed(
    seen_job_id: int, conn: Optional[sqlite3.Connection] = None
) -> None:
    """Mark a seen job as failed to score."""
    with _connection(conn) as c:
        c.execute(
            "UPDATE seen_jobs SET score_status = 'failed' WHERE id = ?",
            (seen_job_id,),
        )
