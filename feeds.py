"""Job-board feeds: fetch postings and normalize them into job dicts.

The ingestion layer for the pipeline. Each posting is normalized into the dict
shape the rest of the app consumes:

    {"url", "title", "company", "location", "description", "source"}

`url`/`title`/`company`/`location`/`description` are exactly the keys
`database.mark_seen` and `scorer.score` expect, so `fetch()` output drops
straight into the pipeline.

Sources:
  - **We Work Remotely** (working) — real per-category RSS feeds. The active
    source; `fetch()` aggregates these.
  - **Indeed** (stub) — its public RSS returns HTTP 403 (deprecated/blocked), so
    `fetch_indeed()` yields nothing in practice. Kept callable for when a working
    Indeed path exists; intentionally left out of `SOURCES`.

When config.py exists, the per-source search parameters here should move there
(per CLAUDE.md, config.py is the single place for parameters).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence
from urllib.parse import urlencode

import feedparser

# --- shared helpers ---------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Collapse an HTML summary into plain-ish text."""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()


def _collect(
    urls: Sequence[str], parser: Callable[[Mapping[str, Any]], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Fetch each feed URL, parse its entries, and dedupe by URL.

    Empty or malformed feeds contribute nothing rather than raising.
    """
    jobs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for url in urls:
        parsed = feedparser.parse(url)
        for entry in getattr(parsed, "entries", []) or []:
            job = parser(entry)
            u = job["url"]
            if not u or u in seen:
                continue
            seen.add(u)
            jobs.append(job)
    return jobs


# --- We Work Remotely (working) ---------------------------------------------

WWRR_CATEGORY_RSS = "https://weworkremotely.com/categories/{slug}.rss"


@dataclass
class WWRRSearch:
    """We Work Remotely categories to pull (one RSS feed each)."""

    categories: Sequence[str] = (
        "remote-back-end-programming-jobs",
        "remote-full-stack-programming-jobs",
        "remote-devops-sysadmin-jobs",
    )


DEFAULT_WWRR = WWRRSearch()


def wwrr_urls(search: WWRRSearch = DEFAULT_WWRR) -> List[str]:
    """The RSS feed URL for each configured category."""
    return [WWRR_CATEGORY_RSS.format(slug=slug) for slug in search.categories]


def parse_wwrr_entry(
    entry: Mapping[str, Any], source: str = "weworkremotely"
) -> Dict[str, Any]:
    """Normalize one We Work Remotely entry.

    WWRR puts "Company: Job Title" in the title and the location in `region`.
    """
    raw_title = (entry.get("title") or "").strip()
    company, sep, title = raw_title.partition(": ")
    if not sep:  # no "Company: Title" separator
        company, title = "", raw_title

    region = (entry.get("region") or "").strip()
    summary = entry.get("summary") or entry.get("description") or ""
    return {
        "url": (entry.get("link") or "").strip(),
        "title": title.strip(),
        "company": company.strip() or None,
        "location": region or None,
        "description": _strip_html(summary),
        "source": source,
    }


def fetch_wwrr(search: WWRRSearch = DEFAULT_WWRR) -> List[Dict[str, Any]]:
    """Fetch and normalize postings from the configured WWRR categories."""
    return _collect(wwrr_urls(search), parse_wwrr_entry)


# --- Indeed (stub — public RSS returns HTTP 403) ----------------------------

INDEED_RSS_BASE = "https://www.indeed.com/rss"
REMOTE = "Remote"


@dataclass
class IndeedSearch:
    """Indeed search parameters. One feed is fetched per location."""

    query: str = "backend engineer java"          # q
    locations: Sequence[str] = (                    # l (one feed each)
        REMOTE,
        "New York, NY",
        "Jersey City, NJ",                          # North Jersey anchor
    )
    radius: int = 25                                # miles (skipped for Remote)
    fromage: int = 1                                # max posting age, days
    sort: str = "date"                              # 'date' | 'relevance'
    job_type: str = "fulltime"                      # jt
    limit: int = 25                                 # results per page (max 50)


DEFAULT_INDEED = IndeedSearch()


def indeed_url(search: IndeedSearch, location: str, start: int = 0) -> str:
    """Build the Indeed RSS search URL for one location.

    `radius` is omitted for remote searches (it is meaningless there).
    """
    params: Dict[str, Any] = {
        "q": search.query,
        "l": location,
        "fromage": search.fromage,
        "sort": search.sort,
        "jt": search.job_type,
        "limit": search.limit,
    }
    if location.strip().lower() != REMOTE.lower():
        params["radius"] = search.radius
    if start:
        params["start"] = start
    return f"{INDEED_RSS_BASE}?{urlencode(params)}"


def parse_indeed_entry(
    entry: Mapping[str, Any], source: str = "indeed"
) -> Dict[str, Any]:
    """Normalize one Indeed entry.

    Indeed packs "Job Title - Company - Location" into the entry title, so split
    from the right into at most three parts and degrade gracefully when company
    or location is missing.
    """
    raw_title = (entry.get("title") or "").strip()
    parts = [p.strip() for p in raw_title.rsplit(" - ", 2)]
    title = parts[0] if parts else raw_title
    company = parts[1] if len(parts) >= 3 else None
    location = parts[-1] if len(parts) >= 2 else None

    summary = entry.get("summary") or entry.get("description") or ""
    return {
        "url": (entry.get("link") or "").strip(),
        "title": title,
        "company": company,
        "location": location,
        "description": _strip_html(summary),
        "source": source,
    }


def fetch_indeed(search: IndeedSearch = DEFAULT_INDEED) -> List[Dict[str, Any]]:
    """Fetch and normalize Indeed postings across all configured locations.

    In practice this returns nothing — Indeed's public RSS responds 403.
    """
    return _collect(
        [indeed_url(search, loc) for loc in search.locations], parse_indeed_entry
    )


# --- aggregate --------------------------------------------------------------

# Feed sources aggregated by fetch(). Indeed is intentionally omitted: its RSS
# returns HTTP 403, so including it would just spend requests for zero jobs. Add
# `fetch_indeed` here once a working Indeed path exists.
SOURCES: Sequence[Callable[[], List[Dict[str, Any]]]] = (fetch_wwrr,)


def fetch() -> List[Dict[str, Any]]:
    """Fetch all active sources and return the merged, URL-deduped job dicts."""
    jobs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for source in SOURCES:
        for job in source():
            u = job["url"]
            if not u or u in seen:
                continue
            seen.add(u)
            jobs.append(job)
    return jobs
