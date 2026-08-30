"""Unified LLM client wrapping OpenAI-compatible APIs."""

from __future__ import annotations

from openai import OpenAI

from ..config.llm import LLMConfig


class LLMClient:
    """Unified LLM client for all OpenAI-compatible providers."""

    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config
        api_key = llm_config.get_api_key()
        if llm_config.base_url:
            self._client = OpenAI(base_url=llm_config.base_url, api_key=api_key)
        else:
            self._client = OpenAI(api_key=api_key)
        self.model = llm_config.model_string

    @property
    def raw_client(self) -> OpenAI:
        """Access the underlying OpenAI client."""
        return self._client

    def chat(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        max_tokens = max_tokens or self.config.max_tokens
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        params = {"model": self.model, "messages": messages}
        model_lower = self.model.lower()
        if "o1" in model_lower or "o3" in model_lower or "gpt-5" in model_lower:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()

    def chat_with_usage(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int = None,
    ) -> tuple[str, dict[str, int]]:
        """Chat and return (response_text, {input: N, output: N})."""
        max_tokens = max_tokens or self.config.max_tokens
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        params = {"model": self.model, "messages": messages}
        model_lower = self.model.lower()
        if "o1" in model_lower or "o3" in model_lower or "gpt-5" in model_lower:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**params)
        text = response.choices[0].message.content.strip()

        if hasattr(response, 'usage') and response.usage:
            tokens = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
            }
        else:
            tokens = {
                "input": len((system_prompt + user_input).split()) * 2,
                "output": len(text.split()) * 2,
            }
        return text, tokens
