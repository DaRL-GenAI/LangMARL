"""LangMARL — Language-space Multi-Agent Reinforcement Learning.

Applies multi-agent credit assignment and policy gradient optimization
to LLM-based systems via Centralized Training with Decentralized Execution (CTDE).

Quick start:

    import langmarl

    config = langmarl.LanguageTaskConfig(
        task_type="qa",
        paradigm="central_credit",
        llm=langmarl.LLMConfig.from_preset("gpt-4o-mini"),
    )

    env = langmarl.make_env("language", config)
    trainer = langmarl.MonteCarloTrainer(
        config=config,
        env=env,
        critic=langmarl.CentralizedCritic(config, env=env),
        optimizer=langmarl.PolicyGradientOptimizer(config.get_optimizer_llm()),
    )
    trainer.train()
"""

try:  # keep in lockstep with pyproject.toml rather than duplicating it
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("langmarl")
except Exception:  # not installed (e.g. running from a source checkout)
    __version__ = "1.0.0"

# Core abstractions
# Configuration
from langmarl.config.base import (
    BaseConfig,
    LanguageTaskConfig,
    OvercookedConfig,
    PistonballConfig,
)
from langmarl.config.llm import LLMConfig, get_llm_config, list_available_models
from langmarl.core.base import (
    BaseAgent,
    BaseCritic,
    BaseEnvironment,
    BaseOptimizer,
    BaseReward,
    Trajectory,
)
from langmarl.core.critic import CentralizedCritic
from langmarl.core.optimizer import PolicyGradientOptimizer
from langmarl.core.policy import AgentPolicy, default_agent_policy
from langmarl.core.trajectory import TrajectoryFormatter

# Environment registry
from langmarl.envs import list_envs, make_env, register_env

# LLM client
from langmarl.llm.client import LLMClient
from langmarl.llm.token_tracker import TokenTracker

# Store
from langmarl.store import LocalStore, PolicyCheckpoint, RunLogger, TrajectoryStore
from langmarl.trainer.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)

# Trainer
from langmarl.trainer.monte_carlo import MonteCarloTrainer


def train(config_path: str, **overrides):
    """One-line training: load config, create components, run training.

    Args:
        config_path: Path to a JSON config file.
        **overrides: Key=value overrides for the config.

    Example:
        langmarl.train("configs/language_task/qa_central_credit.json")
    """
    config = load_config(config_path, overrides=overrides)
    env_name = _env_name_of(config_path)

    env = make_env(env_name, config)
    # The critic reads the environment's prompts and trajectory format
    critic = CentralizedCritic(config, env=env)
    optimizer = PolicyGradientOptimizer(config.get_optimizer_llm())

    trainer = MonteCarloTrainer(
        config=config,
        env=env,
        critic=critic,
        optimizer=optimizer,
    )
    return trainer.train()


def load_config(path: str, overrides: dict = None):
    """Load configuration from a JSON file.

    Args:
        path: Path to JSON config file.
        overrides: Optional dictionary of overrides.

    Returns:
        BaseConfig or subclass instance.
    """
    return _CONFIG_FOR_ENV.get(_env_name_of(path), BaseConfig).from_json(
        path, overrides=overrides
    )


#: Which config dataclass each built-in environment expects.
_CONFIG_FOR_ENV = {
    "language": LanguageTaskConfig,
    "overcooked": OvercookedConfig,
    "pistonball": PistonballConfig,
}


def _env_name_of(config_path: str) -> str:
    """Read the "env" key from a config file, defaulting to the language task."""
    import json
    with open(config_path) as f:
        return json.load(f).get("env", "language")


__all__ = [
    # Core
    "Trajectory",
    "BaseEnvironment",
    "BaseAgent",
    "BaseCritic",
    "BaseOptimizer",
    "BaseReward",
    "PolicyGradientOptimizer",
    "AgentPolicy",
    "default_agent_policy",
    "CentralizedCritic",
    "TrajectoryFormatter",
    # Config
    "BaseConfig",
    "LanguageTaskConfig",
    "OvercookedConfig",
    "PistonballConfig",
    "LLMConfig",
    "get_llm_config",
    "list_available_models",
    # LLM
    "LLMClient",
    "TokenTracker",
    # Trainer
    "MonteCarloTrainer",
    "Callback",
    "LoggingCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    # Envs
    "make_env",
    "register_env",
    "list_envs",
    # Store
    "LocalStore",
    "PolicyCheckpoint",
    "TrajectoryStore",
    "RunLogger",
    # Convenience
    "train",
    "load_config",
]
