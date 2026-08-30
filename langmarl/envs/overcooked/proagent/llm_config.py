"""
LLM Configuration loader for OpenAI and OpenRouter support.

Usage:
    from proagent.llm_config import get_llm_config, create_openai_client

    config = get_llm_config()
    client = create_openai_client()
"""

import os
import json
from openai import OpenAI

# Default config file path
CONFIG_FILE = os.path.join(os.getcwd(), "llm_config.json")

# Cached config
_config = None


def load_config(config_path: str = None, use_cache: bool = True) -> dict:
    """Load LLM configuration from JSON file.

    Args:
        config_path: Path to the config file. If None, uses CONFIG_FILE.
        use_cache: If True, use global cache. Set to False for multiprocessing.

    Returns:
        Configuration dictionary.
    """
    global _config

    # Use cache only if no specific path is given and caching is enabled
    if use_cache and _config is not None and config_path is None:
        return _config

    path = config_path or CONFIG_FILE

    if not os.path.exists(path):
        # Fallback to environment variables or openai_key.txt
        return _load_fallback_config()

    with open(path, 'r') as f:
        config = json.load(f)

    # Only cache if using default path and caching is enabled
    if use_cache and config_path is None:
        _config = config

    return config


def _load_fallback_config() -> dict:
    """Fallback to old openai_key.txt or environment variables."""
    config = {
        "provider": "openai",
        "openai": {"api_key": ""},
        "openrouter": {
            "api_key": "",
            "base_url": "https://openrouter.ai/api/v1",
            "site_url": "",
            "site_name": ""
        },
        "model": "gpt-4o-mini"
    }

    # Try environment variables first
    if os.getenv("USE_OPENROUTER", "").lower() == "true":
        config["provider"] = "openrouter"
        config["openrouter"]["api_key"] = os.getenv("OPENROUTER_API_KEY", "")
        config["openrouter"]["base_url"] = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        config["openrouter"]["site_url"] = os.getenv("OPENROUTER_SITE_URL", "")
        config["openrouter"]["site_name"] = os.getenv("OPENROUTER_SITE_NAME", "")
    elif os.getenv("OPENAI_API_KEY"):
        config["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
    else:
        # Try openai_key.txt
        openai_key_file = os.path.join(os.getcwd(), "openai_key.txt")
        if os.path.exists(openai_key_file):
            with open(openai_key_file, 'r') as f:
                keys = f.read().strip().split('\n')
                if keys:
                    config["openai"]["api_key"] = keys[0]

    return config


def get_llm_config() -> dict:
    """Get the current LLM configuration."""
    return load_config()


def is_openrouter() -> bool:
    """Check if OpenRouter is configured as the provider."""
    config = get_llm_config()
    return config.get("provider", "openai").lower() == "openrouter"


def get_api_key() -> str:
    """Get the API key for the configured provider."""
    config = get_llm_config()
    if is_openrouter():
        return config.get("openrouter", {}).get("api_key", "")
    return config.get("openai", {}).get("api_key", "")


def get_base_url() -> str:
    """Get the base URL (only for OpenRouter)."""
    config = get_llm_config()
    if is_openrouter():
        return config.get("openrouter", {}).get("base_url", "https://openrouter.ai/api/v1")
    return None


def get_extra_headers() -> dict:
    """Get extra headers for OpenRouter."""
    if not is_openrouter():
        return {}

    config = get_llm_config()
    openrouter_config = config.get("openrouter", {})

    headers = {}
    site_url = openrouter_config.get("site_url", "")
    site_name = openrouter_config.get("site_name", "")

    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    return headers


def get_model() -> str:
    """Get the configured model name."""
    config = get_llm_config()
    return config.get("model", "gpt-4o-mini")


def create_openai_client(timeout: float = 60.0) -> OpenAI:
    """Create an OpenAI client configured for the current provider."""
    config = get_llm_config()

    if is_openrouter():
        return OpenAI(
            base_url=get_base_url(),
            api_key=get_api_key(),
            timeout=timeout
        )
    else:
        return OpenAI(
            api_key=get_api_key(),
            timeout=timeout
        )


def reload_config(config_path: str = None):
    """Force reload the configuration."""
    global _config
    _config = None
    return load_config(config_path)


# ============================================================================
# Config-aware functions for multiprocessing support
# These functions accept an optional config dict/path to avoid global state
# ============================================================================

def is_openrouter_from_config(config: dict) -> bool:
    """Check if OpenRouter is configured as the provider (config-aware version)."""
    return config.get("provider", "openai").lower() == "openrouter"


def get_api_key_from_config(config: dict) -> str:
    """Get the API key for the configured provider (config-aware version)."""
    if is_openrouter_from_config(config):
        return config.get("openrouter", {}).get("api_key", "")
    return config.get("openai", {}).get("api_key", "")


def get_base_url_from_config(config: dict) -> str:
    """Get the base URL (only for OpenRouter) (config-aware version)."""
    if is_openrouter_from_config(config):
        return config.get("openrouter", {}).get("base_url", "https://openrouter.ai/api/v1")
    return None


def create_openai_client_from_config(config: dict, timeout: float = 60.0) -> OpenAI:
    """Create an OpenAI client configured for the specific config (config-aware version).

    Args:
        config: Configuration dictionary
        timeout: Request timeout in seconds

    Returns:
        Configured OpenAI client
    """
    if is_openrouter_from_config(config):
        return OpenAI(
            base_url=get_base_url_from_config(config),
            api_key=get_api_key_from_config(config),
            timeout=timeout
        )
    else:
        return OpenAI(
            api_key=get_api_key_from_config(config),
            timeout=timeout
        )


def set_process_config(config_path: str = None):
    """Set the LLM config for the current process (useful in multiprocessing workers).

    This loads the config and caches it globally for the current process.

    Args:
        config_path: Path to the config file. If None, uses default CONFIG_FILE.
    """
    global _config
    _config = None  # Clear existing cache
    return load_config(config_path, use_cache=True)
