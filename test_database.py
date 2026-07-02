"""Unit tests for database.py (in-memory SQLite, no API calls)."""

from __future__ import annotations

import json
import sqlite3

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


# --- commit / rollback semantics (fix #1) -----------------------------------


def test_injected_connection_writes_are_committed(tmp_path):
    # A caller-supplied connection's writes must persist even if the caller
    # closes without an explicit commit — the library commits each operation.
    db = tmp_path / "jobs.db"
    c1 = database.connect(db)
    sid = database.mark_seen("https://x/1", title="T", conn=c1)
    database.save_result({"seen_job_id": sid, "score": 90}, conn=c1)
    c1.close()  # no explicit c1.commit() by the caller

    c2 = database.connect(db)
    try:
        assert database.is_seen("https://x/1", conn=c2) is True
        row = c2.execute(
            "SELECT score FROM scored_jobs WHERE seen_job_id = ?", (sid,)
        ).fetchone()
        assert row["score"] == 90
    finally:
        c2.close()


def test_failed_write_rolls_back_and_connection_stays_usable(conn):
    # FK violation (seen_job_id 999 doesn't exist; foreign_keys is ON) must roll
    # back cleanly and leave the shared connection usable for the next call.
    with pytest.raises(sqlite3.IntegrityError):
        database.save_result({"seen_job_id": 999, "score": 50}, conn=conn)

    assert conn.execute("SELECT COUNT(*) AS n FROM scored_jobs").fetchone()["n"] == 0
    sid = database.mark_seen("https://x/after", conn=conn)
    assert database.is_seen("https://x/after", conn=conn) is True


# --- cached default connection (fix #2) -------------------------------------


def test_default_connection_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DEFAULT_DB_PATH", tmp_path / "jobs.db")
    database.close()  # drop any stale cached connection
    try:
        assert database.is_seen("https://x/def") is False
        database.mark_seen("https://x/def")
        assert database.is_seen("https://x/def") is True
        # The same connection object backs every default-path call.
        first = database._default_connection()
        database.is_seen("https://x/def")
        assert database._default_connection() is first
    finally:
        database.close()


def test_close_resets_default_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DEFAULT_DB_PATH", tmp_path / "jobs.db")
    database.close()
    try:
        database.mark_seen("https://x/c")
        conn_a = database._default_connection()
        database.close()
        assert database._default_conn is None
        conn_b = database._default_connection()
        assert conn_b is not conn_a
    finally:
        database.close()


# --- notification workflow (fix #3) -----------------------------------------


def test_mark_notified_sets_flag(conn):
    sid = database.mark_seen("https://x/n", conn=conn)
    scored_id = database.save_result({"seen_job_id": sid, "score": 80}, conn=conn)
    before = conn.execute(
        "SELECT notified FROM scored_jobs WHERE id = ?", (scored_id,)
    ).fetchone()
    assert before["notified"] == 0

    database.mark_notified(scored_id, conn=conn)
    after = conn.execute(
        "SELECT notified FROM scored_jobs WHERE id = ?", (scored_id,)
    ).fetchone()
    assert after["notified"] == 1


def test_unnotified_results_filters_decodes_and_orders(conn):
    hi = database.mark_seen("https://x/hi", title="Hi", company="Acme", conn=conn)
    r_hi = database.save_result(
        {"seen_job_id": hi, "score": 90, "reasoning": "great",
         "green_flags": ["Java"], "red_flags": []},
        conn=conn,
    )
    mid = database.mark_seen("https://x/mid", conn=conn)
    r_mid = database.save_result({"seen_job_id": mid, "score": 75}, conn=conn)
    lo = database.mark_seen("https://x/lo", conn=conn)
    database.save_result({"seen_job_id": lo, "score": 40}, conn=conn)  # below min
    done = database.mark_seen("https://x/done", conn=conn)
    r_done = database.save_result({"seen_job_id": done, "score": 95}, conn=conn)
    database.mark_notified(r_done, conn=conn)  # already notified -> excluded

    results = database.unnotified_results(min_score=70, conn=conn)
    # Only un-notified jobs with score >= 70, highest first: hi(90), mid(75).
    assert [r["id"] for r in results] == [r_hi, r_mid]
    assert [r["score"] for r in results] == [90, 75]

    top = results[0]
    assert top["url"] == "https://x/hi"
    assert top["title"] == "Hi"
    assert top["company"] == "Acme"
    assert top["green_flags"] == ["Java"]  # decoded back into a list
    assert top["red_flags"] == []
