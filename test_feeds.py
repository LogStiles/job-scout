"""Unit tests for feeds.py (pure URL/parse logic + fetch with mocked network)."""

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import feeds


def _feed(*entries):
    return SimpleNamespace(entries=list(entries), bozo=0)


# --- We Work Remotely -------------------------------------------------------


def test_wwrr_urls_builds_category_feeds():
    search = feeds.WWRRSearch(categories=("remote-back-end-programming-jobs", "x"))
    assert feeds.wwrr_urls(search) == [
        "https://weworkremotely.com/categories/remote-back-end-programming-jobs.rss",
        "https://weworkremotely.com/categories/x.rss",
    ]


def test_parse_wwrr_entry_splits_company_and_uses_region():
    job = feeds.parse_wwrr_entry(
        {
            "title": "1Password: Senior Web Developer",
            "link": "https://weworkremotely.com/remote-jobs/1password-senior-web-developer",
            "region": "Anywhere in the World",
            "summary": "<p>Build things.</p>",
        }
    )
    assert job["company"] == "1Password"
    assert job["title"] == "Senior Web Developer"
    assert job["location"] == "Anywhere in the World"
    assert job["url"].endswith("1password-senior-web-developer")
    assert job["description"] == "Build things."
    assert job["source"] == "weworkremotely"


def test_parse_wwrr_entry_title_may_contain_colon():
    # Only the first ": " separates company from title.
    job = feeds.parse_wwrr_entry(
        {"title": "Acme: Engineer: Backend", "link": "x", "region": "USA Only"}
    )
    assert job["company"] == "Acme"
    assert job["title"] == "Engineer: Backend"


def test_parse_wwrr_entry_without_company_separator():
    job = feeds.parse_wwrr_entry({"title": "Executive Director", "link": "x"})
    assert job["company"] is None
    assert job["title"] == "Executive Director"


def test_parse_wwrr_entry_missing_region():
    job = feeds.parse_wwrr_entry({"title": "Co: Eng", "link": "x", "summary": ""})
    assert job["location"] is None


def test_parse_wwrr_entry_strips_html():
    job = feeds.parse_wwrr_entry(
        {"title": "Co: Eng", "link": "x", "summary": "<p>Java &amp; <b>Spring</b>.</p>"}
    )
    assert "<" not in job["description"] and ">" not in job["description"]
    assert "Spring" in job["description"]


def test_fetch_wwrr_dedupes_across_categories(monkeypatch):
    dup = {"title": "Co: Eng", "link": "https://wwr/1", "region": "Remote", "summary": ""}
    uniq = {"title": "Co: Dev", "link": "https://wwr/2", "region": "Remote", "summary": ""}

    calls = {"n": 0}

    def fake_parse(url):
        calls["n"] += 1
        return _feed(dup, uniq) if calls["n"] == 1 else _feed(dup)

    monkeypatch.setattr(feeds.feedparser, "parse", fake_parse)
    search = feeds.WWRRSearch(categories=("a", "b"))
    jobs = feeds.fetch_wwrr(search)

    assert calls["n"] == 2  # one fetch per category
    assert [j["url"] for j in jobs] == ["https://wwr/1", "https://wwr/2"]


# --- Indeed (stub) ----------------------------------------------------------


def _query(url: str) -> dict:
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def test_indeed_url_includes_expected_params():
    search = feeds.IndeedSearch(
        query="backend engineer java",
        locations=("New York, NY",),
        radius=25,
        fromage=1,
        sort="date",
        job_type="fulltime",
        limit=25,
    )
    q = _query(feeds.indeed_url(search, "New York, NY"))
    assert q["q"] == "backend engineer java"
    assert q["l"] == "New York, NY"
    assert q["radius"] == "25"
    assert q["fromage"] == "1"
    assert q["sort"] == "date"
    assert q["jt"] == "fulltime"
    assert q["limit"] == "25"


def test_indeed_url_encodes_query_and_omits_radius_for_remote():
    search = feeds.IndeedSearch(query="c++ & java", radius=25)
    url = feeds.indeed_url(search, "New York, NY")
    assert " " not in url  # spaces are percent-encoded
    assert _query(url)["q"] == "c++ & java"
    assert "radius" not in _query(feeds.indeed_url(search, "Remote"))


def test_indeed_url_includes_start_when_paginating():
    search = feeds.IndeedSearch()
    assert "start" not in _query(feeds.indeed_url(search, "Remote", start=0))
    assert _query(feeds.indeed_url(search, "Remote", start=25))["start"] == "25"


def test_parse_indeed_entry_splits_title_company_location():
    job = feeds.parse_indeed_entry(
        {
            "title": "Full-Stack Engineer - Beta LLC - New York, NY",
            "link": "https://indeed.com/viewjob?jk=1",
            "summary": "Build things.",
        }
    )
    assert job["title"] == "Full-Stack Engineer"  # internal hyphen preserved
    assert job["company"] == "Beta LLC"
    assert job["location"] == "New York, NY"
    assert job["url"] == "https://indeed.com/viewjob?jk=1"
    assert job["source"] == "indeed"


def test_parse_indeed_entry_missing_company_and_location():
    job = feeds.parse_indeed_entry({"title": "Backend Engineer", "link": "x"})
    assert job["company"] is None
    assert job["location"] is None


# --- aggregate fetch() ------------------------------------------------------


def test_fetch_aggregates_sources_and_dedupes(monkeypatch):
    job_a = {"url": "https://j/a", "title": "A"}
    job_b = {"url": "https://j/b", "title": "B"}
    job_c = {"url": "https://j/c", "title": "C"}
    # Two sources; job_b overlaps and must appear once, order preserved.
    monkeypatch.setattr(
        feeds, "SOURCES", (lambda: [job_a, job_b], lambda: [job_b, job_c])
    )
    assert [j["url"] for j in feeds.fetch()] == ["https://j/a", "https://j/b", "https://j/c"]


def test_fetch_skips_entries_without_url(monkeypatch):
    monkeypatch.setattr(
        feeds, "SOURCES", (lambda: [{"url": "", "title": "x"}, {"url": "https://j/ok"}],)
    )
    assert [j["url"] for j in feeds.fetch()] == ["https://j/ok"]


def test_fetch_wwrr_tolerates_empty_or_malformed_feed(monkeypatch):
    monkeypatch.setattr(
        feeds.feedparser, "parse", lambda url: SimpleNamespace(entries=[], bozo=1)
    )
    assert feeds.fetch_wwrr(feeds.WWRRSearch(categories=("a", "b"))) == []
