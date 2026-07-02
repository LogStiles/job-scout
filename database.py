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

    # Later, notify on high scorers and record that they were alerted:
    for job in database.unnotified_results(min_score=70):
        notify(job)                                 # your delivery mechanism
        database.mark_notified(job["id"])

`scorer.Score` (score, reasoning, green_flags, red_flags) maps directly onto the
keys save_result() expects, via `model_dump()`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

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
    """Open a connection, enforce foreign keys, and ensure the schema exists.

    ``check_same_thread=False`` keeps a shared/cached connection usable across
    threads (SQLite serializes writes itself). Callers doing heavy concurrent
    writes should instead pass their own per-thread connection.
    """
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# One reusable connection for the default (conn=None) path, so hot-loop calls
# like is_seen() per scraped URL don't reopen the DB and re-run the schema DDL
# every time. Created lazily; lives for the process (see close()).
_default_conn: Optional[sqlite3.Connection] = None


def _default_connection() -> sqlite3.Connection:
    global _default_conn
    if _default_conn is None:
        _default_conn = connect(DEFAULT_DB_PATH)
    return _default_conn


def close() -> None:
    """Close and drop the cached default connection (for shutdown or tests)."""
    global _default_conn
    if _default_conn is not None:
        _default_conn.close()
        _default_conn = None


@contextmanager
def _connection(conn: Optional[sqlite3.Connection]) -> Iterator[sqlite3.Connection]:
    """Yield a usable connection and commit the operation on success.

    When `conn` is None the cached default connection is reused. Whether the
    connection is the default or caller-supplied, each operation is committed on
    success and rolled back on error, so writes always persist and a failed
    write never leaks a partial transaction into the next call. Connections are
    never closed here — the default is reused (see close()); caller-supplied
    connections are the caller's to manage.
    """
    c = _default_connection() if conn is None else conn
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise


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


def mark_notified(
    scored_job_id: int, conn: Optional[sqlite3.Connection] = None
) -> None:
    """Mark a scored job as already notified so it isn't alerted on twice."""
    with _connection(conn) as c:
        c.execute(
            "UPDATE scored_jobs SET notified = 1 WHERE id = ?",
            (scored_job_id,),
        )


def unnotified_results(
    min_score: int = 0, conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """Return scored jobs not yet notified, joined with their seen-job info.

    Rows are ordered by score (highest first) and filtered to `score >= min_score`.
    `green_flags`/`red_flags` are decoded back into lists. Each dict includes the
    `scored_jobs.id` to pass to mark_notified().
    """
    with _connection(conn) as c:
        rows = c.execute(
            """
            SELECT
                sj.id AS id,
                sj.seen_job_id AS seen_job_id,
                sj.score AS score,
                sj.reasoning AS reasoning,
                sj.green_flags AS green_flags,
                sj.red_flags AS red_flags,
                sj.scored_at AS scored_at,
                s.url AS url,
                s.title AS title,
                s.company AS company,
                s.location AS location
            FROM scored_jobs sj
            JOIN seen_jobs s ON s.id = sj.seen_job_id
            WHERE sj.notified = 0 AND sj.score >= ?
            ORDER BY sj.score DESC, sj.id ASC
            """,
            (min_score,),
        ).fetchall()

    results: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["green_flags"] = json.loads(item["green_flags"]) if item["green_flags"] else []
        item["red_flags"] = json.loads(item["red_flags"]) if item["red_flags"] else []
        results.append(item)
    return results
