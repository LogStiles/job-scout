"""Score a resume against a job posting using an LLM.

Given a resume (candidate profile) and a job posting, this builds a scoring
prompt and asks Claude to rate how strong a match the candidate is for the
posting, returning a structured 0-100 score with reasoning and flags.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import anthropic
from pydantic import BaseModel, Field

MODEL = "claude-opus-4-8"

# The default candidate profile lives in profile.txt next to this module so it
# can be swapped out (e.g. via `resume.py --set-default`) without editing code.
# The file is gitignored — this constant is the built-in fallback used when it
# is absent (e.g. a fresh clone).
PROFILE_PATH = Path(__file__).with_name("profile.txt")

BUILTIN_PROFILE = """-Title target: Software Engineer
- Core skills: Java, Spring Boot, Kubernetes, AWS
- Soft skills: PostgreSQL, Javascript, Python, CI/CD pipelines
- Years of experience: 4
- Location: NYC or North Jersey (remote or hybrid acceptable, fully onsite in NYC acceptable)
- Preferred domains: fintech, healthtech, enterprise SaaS
- Hard disqualifiers: requires 10+ years, C#/.NET only stack, fully onsite outside NYC/NJ"""



def load_default_profile() -> str:
    """Return the current default profile text (from profile.txt, or built-in)."""
    if PROFILE_PATH.exists():
        text = PROFILE_PATH.read_text(encoding="utf-8").strip()
        if text:
            return text
    return BUILTIN_PROFILE


def save_default_profile(text: str) -> None:
    """Persist `text` as the new default profile in profile.txt."""
    PROFILE_PATH.write_text(text.strip() + "\n", encoding="utf-8")

PROMPT_TEMPLATE = """You are evaluating a job posting for a candidate with this profile:
{profile}

Score the following job posting from 0 to 100 based on how strong a candidate match this person is.
Return ONLY a JSON object in this exact format, no other text:
{{
  "score": <integer 0-100>,
  "reasoning": "<2-3 sentence explanation>",
  "green_flags": ["<flag1>", "<flag2>"],
  "red_flags": ["<flag1>", "<flag2>"]
}}

JOB POSTING:
{job_posting}"""


class Score(BaseModel):
    """Structured scoring result returned by the model."""

    score: int = Field(ge=0, le=100, description="Match score from 0 to 100")
    reasoning: str = Field(description="2-3 sentence explanation")
    green_flags: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)


@dataclass
class Profile:
    """A candidate profile. `text` is inserted verbatim into the prompt.

    Defaults to the current default profile (profile.txt, or the built-in).
    """

    text: str = field(default_factory=load_default_profile)


def build_prompt(job_posting: str, profile: Profile | None = None) -> str:
    """Construct the scoring prompt for a job posting and candidate profile."""
    profile = profile or Profile()
    return PROMPT_TEMPLATE.format(
        profile=profile.text.strip(),
        job_posting=job_posting.strip(),
    )


def score(
    job_posting: str,
    profile: Profile | None = None,
    client: anthropic.Anthropic | None = None,
) -> Score:
    """Score a job posting for the candidate and return a structured result.

    Requires ANTHROPIC_API_KEY in the environment (or a preconfigured client).
    """
    client = client or anthropic.Anthropic()
    prompt = build_prompt(job_posting, profile)

    # Structured outputs guarantee the response validates against the schema,
    # so we never have to parse free-form text or handle stray prose.
    response = client.messages.parse(
        model=MODEL,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
        output_format=Score,
    )

    result = response.parsed_output
    if result is None:
        raise RuntimeError(
            f"Model did not return a parseable score (stop_reason={response.stop_reason})"
        )
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "job_posting",
        nargs="?",
        help="Path to a job posting file. Reads stdin if omitted.",
    )
    parser.add_argument(
        "--resume",
        metavar="RESUME.docx",
        help="Path to a .docx resume to distill and score against. "
        "Defaults to the built-in candidate profile if omitted.",
    )
    args = parser.parse_args()

    if args.job_posting:
        with open(args.job_posting, "r", encoding="utf-8") as f:
            posting = f.read()
    else:
        posting = sys.stdin.read()

    if not posting.strip():
        parser.error("No job posting provided (empty file or stdin).")

    profile = None
    if args.resume:
        from resume import profile_from_docx  # lazy: avoids a hard dependency

        profile = profile_from_docx(args.resume)

    result = score(posting, profile=profile)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
