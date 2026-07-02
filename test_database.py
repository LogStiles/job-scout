"""Unit tests for database.py (in-memory SQLite, no API calls)."""

from __future__ import annotations

import json

import pytest

import database


@pytest.fixture
def conn():
    """A shared in-memory connection passed to every call in a test.

    :memory: databases are per-connection, so the whole test must reuse one
    connection for writes to be visible to later reads.
    """
    c = database.connect(":memory:")
    try:
        yield c
    finally:
        c.close()


# --- is_seen / mark_seen ----------------------------------------------------


def test_is_seen_false_then_true(conn):
    url = "https://jobs.example.com/123"
    assert database.is_seen(url, conn=conn) is False
    database.mark_seen(url, conn=conn)
    assert database.is_seen(url, conn=conn) is True


def test_mark_seen_returns_id(conn):
    sid = database.mark_seen("https://jobs.example.com/1", conn=conn)
    assert isinstance(sid, int) and sid > 0


def test_mark_seen_is_idempotent(conn):
    url = "https://jobs.example.com/dup"
    first = database.mark_seen(url, title="Original", conn=conn)
    second = database.mark_seen(url, title="Changed", conn=conn)
    assert first == second

    rows = conn.execute("SELECT COUNT(*) AS n FROM seen_jobs").fetchone()
    assert rows["n"] == 1
    # Original metadata is preserved (ON CONFLICT DO NOTHING).
    row = conn.execute(
        "SELECT title FROM seen_jobs WHERE id = ?", (first,)
    ).fetchone()
    assert row["title"] == "Original"


def test_mark_seen_distinct_urls_distinct_ids(conn):
    a = database.mark_seen("https://jobs.example.com/a", conn=conn)
    b = database.mark_seen("https://jobs.example.com/b", conn=conn)
    assert a != b


def test_mark_seen_persists_metadata_and_default_status(conn):
    sid = database.mark_seen(
        "https://jobs.example.com/full",
        title="Senior Engineer",
        company="Acme",
        location="NYC",
        description="Build things.",
        conn=conn,
    )
    row = conn.execute("SELECT * FROM seen_jobs WHERE id = ?", (sid,)).fetchone()
    assert row["title"] == "Senior Engineer"
    assert row["company"] == "Acme"
    assert row["location"] == "NYC"
    assert row["description"] == "Build things."
    assert row["score_status"] == "pending"


# --- save_result ------------------------------------------------------------


def test_save_result_inserts_and_marks_scored(conn):
    sid = database.mark_seen("https://jobs.example.com/s", conn=conn)
    scored_id = database.save_result(
        {
            "seen_job_id": sid,
            "score": 88,
            "reasoning": "Strong match.",
            "green_flags": ["Java", "fintech"],
            "red_flags": ["onsite only"],
        },
        conn=conn,
    )
    assert isinstance(scored_id, int) and scored_id > 0

    scored = conn.execute(
        "SELECT * FROM scored_jobs WHERE id = ?", (scored_id,)
    ).fetchone()
    assert scored["seen_job_id"] == sid
    assert scored["score"] == 88
    assert scored["reasoning"] == "Strong match."
    assert scored["notified"] == 0
    # Flags round-trip through JSON back to the original lists.
    assert json.loads(scored["green_flags"]) == ["Java", "fintech"]
    assert json.loads(scored["red_flags"]) == ["onsite only"]

    status = conn.execute(
        "SELECT score_status FROM seen_jobs WHERE id = ?", (sid,)
    ).fetchone()
    assert status["score_status"] == "scored"


def test_save_result_defaults_missing_flags_to_empty(conn):
    sid = database.mark_seen("https://jobs.example.com/min", conn=conn)
    scored_id = database.save_result({"seen_job_id": sid, "score": 50}, conn=conn)
    scored = conn.execute(
        "SELECT * FROM scored_jobs WHERE id = ?", (scored_id,)
    ).fetchone()
    assert json.loads(scored["green_flags"]) == []
    assert json.loads(scored["red_flags"]) == []
    assert scored["reasoning"] is None


def test_save_result_score_model_dump_shape(conn):
    # Mirrors the intended scorer.Score.model_dump() + seen_job_id integration.
    sid = database.mark_seen("https://jobs.example.com/int", conn=conn)
    score_dump = {
        "score": 72,
        "reasoning": "Decent.",
        "green_flags": ["remote"],
        "red_flags": [],
    }
    scored_id = database.save_result({**score_dump, "seen_job_id": sid}, conn=conn)
    row = conn.execute(
        "SELECT score FROM scored_jobs WHERE id = ?", (scored_id,)
    ).fetchone()
    assert row["score"] == 72


# --- mark_failed ------------------------------------------------------------


def test_mark_failed_sets_status(conn):
    sid = database.mark_seen("https://jobs.example.com/f", conn=conn)
    database.mark_failed(sid, conn=conn)
    row = conn.execute(
        "SELECT score_status FROM seen_jobs WHERE id = ?", (sid,)
    ).fetchone()
    assert row["score_status"] == "failed"


# --- helpers / schema -------------------------------------------------------


def test_url_hash_stable_and_distinct():
    assert database._url_hash("https://a.com") == database._url_hash("https://a.com")
    assert database._url_hash("https://a.com") != database._url_hash("https://b.com")


def test_connect_is_idempotent():
    # Creating the schema twice on the same connection must not error.
    c = database.connect(":memory:")
    try:
        c.executescript(database.SCHEMA)  # re-run: IF NOT EXISTS keeps it safe
        database.init_db(conn=c)
    finally:
        c.close()
