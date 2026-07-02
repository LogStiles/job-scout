"""Pytest bootstrap: load .env so live tests can find ANTHROPIC_API_KEY.

Minimal, dependency-free parser. Existing environment variables take
precedence over .env values.
"""

from __future__ import annotations

from pathlib import Path


def _load_dotenv() -> None:
    import os

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Don't clobber a value already set in the real environment.
        os.environ.setdefault(key, value)


_load_dotenv()
