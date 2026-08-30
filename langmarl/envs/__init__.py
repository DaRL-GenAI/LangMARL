"""Environment registry and plugin system."""

from __future__ import annotations

from ..core.base import BaseEnvironment

_ENV_REGISTRY: dict[str, type[BaseEnvironment]] = {}


def register_env(name: str):
    """Decorator to register an environment."""
    def decorator(cls):
        _ENV_REGISTRY[name] = cls
        return cls
    return decorator


def make_env(name: str, config) -> BaseEnvironment:
    """Create an environment by name."""
    # Trigger lazy registration of built-in envs
    _ensure_builtins_loaded()

    if name not in _ENV_REGISTRY:
        if name in _UNAVAILABLE:
            extra = _BUILTINS.get(name)
            hint = (
                f" Install its dependencies with: pip install 'langmarl[{extra}]'"
                f"{_EXTRA_NOTES.get(name, '')}"
                if extra else ""
            )
            raise ValueError(
                f"Environment {name!r} is built in but could not be imported: "
                f"{_UNAVAILABLE[name]}.{hint}"
            )
        raise ValueError(f"Unknown environment: {name}. Available: {list(_ENV_REGISTRY)}")
    return _ENV_REGISTRY[name](config)


def list_envs() -> list[str]:
    """List registered environments."""
    _ensure_builtins_loaded()
    return list(_ENV_REGISTRY.keys())


#: Built-in environments -> the optional extra that provides their dependencies.
#: The language task needs nothing beyond the base install.
_BUILTINS = {"language": None, "pistonball": "pistonball", "overcooked": "overcooked"}

#: Extra setup an extra alone does not cover.
_EXTRA_NOTES = {
    "overcooked": (
        " Overcooked also needs the planner patch under env/overcooked_ai/; "
        "see docs/environments.md."
    ),
}

_BUILTINS_LOADED = False


def _ensure_builtins_loaded():
    """Import built-in environments so their @register_env decorators run.

    An environment whose optional dependency is missing is skipped with a debug
    message rather than silently vanishing, so `Unknown environment: pistonball`
    can be traced back to the missing extra.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    import importlib
    import logging

    logger = logging.getLogger(__name__)
    for name in _BUILTINS:
        try:
            importlib.import_module(f".{name}", __package__)
        except ImportError as exc:
            logger.debug("Environment %r unavailable: %s", name, exc)
            _UNAVAILABLE[name] = str(exc)


#: name -> why it could not be imported, for a useful make_env() error
_UNAVAILABLE: dict[str, str] = {}
