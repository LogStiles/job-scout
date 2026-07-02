"""Unit tests for scorer's pure/file helpers (no API calls)."""

from __future__ import annotations

import os

import pytest

import scorer

# The two score() tests below make a real LLM call (cost + network + some
# nondeterminism). Skip them unless an API key is configured.
requires_api = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live scoring tests",
)

# A concrete candidate so these tests don't depend on the local profile.txt.
CANDIDATE = scorer.Profile(
    text=(
        "- Title target: Senior Software Engineer\n"
        "- Core skills: Java, Spring Boot, Kubernetes, AWS\n"
        "- Years of experience: 4\n"
        "- Location: NYC or North Jersey (remote/hybrid/onsite-NYC acceptable)\n"
        "- Preferred domains: fintech, healthtech, enterprise SaaS\n"
        "- Hard disqualifiers: requires 10+ years, C#/.NET only stack, "
        "fully onsite outside NYC/NJ"
    )
)

STRONG_MATCH_POSTING = """Senior Software Engineer — Payments Platform (fintech)
Location: New York, NY (hybrid, 2 days/week; remote considered)
We're hiring a backend engineer with 3-6 years of experience to build our
payments platform in Java and Spring Boot, deployed on Kubernetes and AWS.
You'll own microservices end to end in a fast-growing fintech company.
"""

WEAK_MATCH_POSTING = """Principal .NET Engineer (12+ years required)
Location: Austin, TX — fully onsite, no remote or relocation assistance.
We need a principal engineer with 12+ years building enterprise desktop
software in a C#/.NET-only stack (WPF, WinForms). No Java or cloud work.
This is a hands-on, fully in-office role.
"""


@pytest.fixture
def temp_profile(tmp_path, monkeypatch):
    """Point scorer.PROFILE_PATH at a temp file so tests never touch profile.txt."""
    path = tmp_path / "profile.txt"
    monkeypatch.setattr(scorer, "PROFILE_PATH", path)
    return path


# --- load_default_profile ---------------------------------------------------


def test_load_default_profile_reads_file(temp_profile):
    temp_profile.write_text("- Title target: Widget Engineer\n", encoding="utf-8")
    assert scorer.load_default_profile() == "- Title target: Widget Engineer"


def test_load_default_profile_falls_back_when_missing(temp_profile):
    assert not temp_profile.exists()
    assert scorer.load_default_profile() == scorer.BUILTIN_PROFILE


def test_load_default_profile_falls_back_when_empty(temp_profile):
    temp_profile.write_text("   \n\t\n", encoding="utf-8")
    assert scorer.load_default_profile() == scorer.BUILTIN_PROFILE


def test_load_default_profile_strips_surrounding_whitespace(temp_profile):
    temp_profile.write_text("\n\n- Title target: X\n\n", encoding="utf-8")
    assert scorer.load_default_profile() == "- Title target: X"


# --- save_default_profile ---------------------------------------------------


def test_save_default_profile_writes_file(temp_profile):
    scorer.save_default_profile("- Title target: Saver")
    assert temp_profile.read_text(encoding="utf-8") == "- Title target: Saver\n"


def test_save_default_profile_strips_then_adds_single_newline(temp_profile):
    scorer.save_default_profile("\n  - Title target: Trimmed  \n\n")
    assert temp_profile.read_text(encoding="utf-8") == "- Title target: Trimmed\n"


def test_save_then_load_round_trips(temp_profile):
    scorer.save_default_profile("- Title target: RoundTrip\n- Core skills: Go")
    assert scorer.load_default_profile() == "- Title target: RoundTrip\n- Core skills: Go"


# --- build_prompt -----------------------------------------------------------


def test_build_prompt_includes_profile_and_posting():
    profile = scorer.Profile(text="- Title target: Custom Role")
    prompt = scorer.build_prompt("Senior Go role, remote", profile=profile)
    assert "- Title target: Custom Role" in prompt
    assert "JOB POSTING:\nSenior Go role, remote" in prompt
    # The required JSON scaffold is present.
    assert '"score": <integer 0-100>' in prompt


def test_build_prompt_uses_default_profile_when_none_given(temp_profile):
    temp_profile.write_text("- Title target: Default Role\n", encoding="utf-8")
    prompt = scorer.build_prompt("Some posting")
    assert "- Title target: Default Role" in prompt


def test_build_prompt_strips_inputs():
    profile = scorer.Profile(text="\n\n- Title target: Padded\n\n")
    prompt = scorer.build_prompt("\n\n  a posting  \n\n", profile=profile)
    # Both inputs are stripped: no padding blank lines leak into the prompt.
    assert prompt.endswith("a posting")
    assert "with this profile:\n- Title target: Padded\n\nScore" in prompt


def test_build_prompt_matches_template_shape():
    profile = scorer.Profile(text="P")
    prompt = scorer.build_prompt("J", profile=profile)
    expected = scorer.PROMPT_TEMPLATE.format(profile="P", job_posting="J")
    assert prompt == expected


# --- score (live LLM call) --------------------------------------------------


@requires_api
def test_score_strong_match_is_high():
    result = scorer.score(STRONG_MATCH_POSTING, profile=CANDIDATE)
    assert 65 <= result.score <= 100, (
        f"expected a high score for a strong match, got {result.score}: "
        f"{result.reasoning}"
    )


@requires_api
def test_score_weak_match_is_low():
    result = scorer.score(WEAK_MATCH_POSTING, profile=CANDIDATE)
    assert 0 <= result.score <= 35, (
        f"expected a low score for a mismatch, got {result.score}: "
        f"{result.reasoning}"
    )
