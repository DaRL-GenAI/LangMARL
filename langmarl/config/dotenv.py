"""Read API keys from a ``.env`` file instead of the shell.

A key pasted into a shell lands in the history file and in every process listing
on the machine; one exported in a script gets committed. A ``.env`` file, which
this repository ignores by default, keeps it in one place that is easy to rotate
and hard to leak.

Importing :mod:`langmarl` loads the nearest ``.env`` automatically. Variables
already present in the environment always win, so an explicit ``export`` or a
CI secret still overrides the file, and ``LANGMARL_NO_DOTENV=1`` turns the
lookup off entirely.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Walking up stops here, so a stray .env far above the project is never read.
_ROOT_MARKERS = (".git", "pyproject.toml", "setup.py")

_LOADED: set[str] = set()


def find_dotenv(start: Path | str | None = None) -> Path | None:
    """Locate the nearest ``.env``, searching upwards from ``start``.

    The search stops at the first directory holding a repository marker, so it
    finds the project's own file and not one belonging to a parent checkout.
    """
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
        if any((directory / marker).exists() for marker in _ROOT_MARKERS):
            break
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines, tolerating comments, blanks and ``export``."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()

        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # An unquoted trailing comment is not part of the value.
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_dotenv(
    path: Path | str | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Load a ``.env`` into :data:`os.environ` and return what it set.

    Args:
        path: a specific file, or None to search upwards from the working
            directory.
        override: replace variables that are already set. Off by default, so
            the real environment beats the file.

    Returns:
        The variables this call actually applied; empty if there was no file.
    """
    target = Path(path) if path else find_dotenv()
    if target is None or not target.is_file():
        return {}

    applied = {}
    for key, value in parse_dotenv(target.read_text()).items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    _LOADED.add(str(target.resolve()))
    return applied


def _autoload() -> None:
    """Load the nearest ``.env`` once, at import, unless switched off."""
    if os.environ.get("LANGMARL_NO_DOTENV"):
        return
    try:
        load_dotenv()
    except OSError:
        # An unreadable .env should never stop the library from importing.
        pass
