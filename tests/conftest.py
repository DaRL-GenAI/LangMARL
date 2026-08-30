"""Shared fixtures. No test in this suite makes a network call."""


import pytest

import langmarl


@pytest.fixture(autouse=True)
def _dummy_api_key(monkeypatch):
    """Let components that build an OpenAI client construct without a real key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")


@pytest.fixture
def llm():
    return langmarl.LLMConfig(
        name="test-model",
        model_string="gpt-4o-mini",
        api_key="test-key-not-used",
    )
