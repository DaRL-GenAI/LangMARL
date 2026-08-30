"""No credential may be committed.

API keys belong in environment variables. This test is the backstop: it scans
every tracked file for live key shapes, so a hardcoded key fails CI instead of
reaching a public repository.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Live-key shapes, not the placeholder "sk-..." used throughout the docs.
SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-[A-Za-z0-9]{32,}"),      # OpenRouter
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"),     # OpenAI project key
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),      # Anthropic
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),            # AWS access key id
    re.compile(r"\bhf_[A-Za-z0-9]{34,}\b"),         # Hugging Face
    re.compile(r"\bghp_[A-Za-z0-9]{36,}\b"),        # GitHub PAT
]


def _tracked_files():
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return [REPO / p for p in out.stdout.split("\0") if p]


@pytest.mark.skipif(
    not (REPO / ".git").exists(), reason="not a git checkout"
)
def test_no_api_keys_in_tracked_files():
    offenders = []
    for path in _tracked_files():
        if path == Path(__file__) or not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(REPO)}: {pattern.pattern}")

    assert not offenders, "Credentials found in tracked files:\n" + "\n".join(offenders)


def test_examples_read_the_key_from_the_environment():
    for path in sorted((REPO / "examples").glob("*.py")):
        text = path.read_text()
        assert 'os.environ[\'OPENAI_API_KEY\'] = "sk-' not in text, (
            f"{path.name} hardcodes an API key"
        )
