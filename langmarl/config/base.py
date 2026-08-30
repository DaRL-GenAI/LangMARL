"""Unified configuration system with hierarchical configs."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, fields
from typing import Optional

from .llm import LLMConfig, get_llm_config


@dataclass
class BaseConfig:
    """Shared configuration fields across all environments."""

    # Experiment
    exp_name: str = "experiment"
    paradigm: str = "central_credit"  # "central_global" | "central_credit"

    # Training
    num_iterations: int = 5
    trajectories_per_iteration: int = 10
    mini_batch_size: Optional[int] = None
    start_iteration: int = 0

    # LLM backends
    llm: Optional[LLMConfig] = None
    actor_llm: Optional[LLMConfig] = None
    critic_llm: Optional[LLMConfig] = None
    optimizer_llm: Optional[LLMConfig] = None

    # Agents
    num_agents: int = 2

    # I/O
    experiment_dir: str = "./experiments"
    checkpoint_dir: str = "./ckpt_policy"

    # Parallelism
    max_workers: int = 5

    # Logging
    log_level: str = "INFO"

    def __post_init__(self):
        valid_paradigms = ['central_global', 'central_credit']
        if self.paradigm not in valid_paradigms:
            raise ValueError(f"Invalid paradigm '{self.paradigm}'. Must be one of {valid_paradigms}")
        if self.num_agents < 1:
            raise ValueError("num_agents must be at least 1")

    def get_actor_llm(self) -> LLMConfig:
        """Get LLM config for actors (fallback: llm)."""
        return self.actor_llm or self.llm

    def get_critic_llm(self) -> LLMConfig:
        """Get LLM config for critic (fallback: llm)."""
        return self.critic_llm or self.llm

    def get_optimizer_llm(self) -> LLMConfig:
        """Get LLM config for optimizer (fallback: llm)."""
        return self.optimizer_llm or self.llm

    @classmethod
    def from_json(cls, path: str, overrides: dict = None) -> 'BaseConfig':
        """Load config from JSON file with optional overrides."""
        with open(path) as f:
            data = json.load(f)
        if overrides:
            data.update(overrides)

        # Resolve LLM configs
        for llm_field in ['llm', 'actor_llm', 'critic_llm', 'optimizer_llm']:
            if llm_field in data:
                val = data[llm_field]
                if isinstance(val, str):
                    data[llm_field] = get_llm_config(val)
                elif isinstance(val, dict):
                    if 'preset' in val:
                        data[llm_field] = get_llm_config(val['preset'])
                    else:
                        data[llm_field] = LLMConfig.from_dict(val)

        # Filter to valid fields for this class. Warn rather than drop
        # silently: a typo'd or stale key used to disable a setting with no
        # sign that anything was wrong.
        valid_fields = {f.name for f in fields(cls)}
        ignored = sorted(set(data) - valid_fields - {"env"})
        if ignored:
            warnings.warn(
                f"{cls.__name__} ignoring unknown config keys: {', '.join(ignored)}",
                stacklevel=2,
            )
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_json(self, path: str):
        """Save config to JSON file."""
        data = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, LLMConfig):
                val = val.to_dict()
            data[f.name] = val
        with open(path, 'w') as fh:
            json.dump(data, fh, indent=2)


@dataclass
class LanguageTaskConfig(BaseConfig):
    """Language task specific config."""

    task_type: str = "qa"  # "qa" | "math" | "writing" | "coding"
    benchmark_path: str = ""
    data_limit: Optional[int] = None
    use_verified_reward: bool = False
    verified_reward_model: str = "gpt-4o-mini"
    verified_reward_writing_model: str = "gpt-4o"
    episode_generation_workers: int = 8
    optimizer_workers: int = 1

    def __post_init__(self):
        super().__post_init__()
        valid_task_types = ['qa', 'math', 'writing', 'coding']
        if self.task_type not in valid_task_types:
            raise ValueError(f"Invalid task_type '{self.task_type}'. Must be one of {valid_task_types}")


@dataclass
class OvercookedConfig(BaseConfig):
    """Overcooked specific config."""

    layout: str = "cramped_room"
    episode_horizon: int = 400
    p0_agent: str = "ProAgent"
    p1_agent: str = "ProAgent"
    prompt_level: str = "L1"
    belief_revision: bool = True
    retrieval_method: str = "recent"
    retrieval_k: int = 3

    def __post_init__(self):
        super().__post_init__()
        if self.num_agents != 2:
            raise ValueError("Overcooked is a two-player game; num_agents must be 2")


@dataclass
class PistonballConfig(BaseConfig):
    """Pistonball specific config."""

    num_pistons: int = 20
    # One policy per piston, so this tracks num_pistons rather than BaseConfig's 2.
    num_agents: int = 20
    max_cycles: int = 125
    frame_size: int = 84
    stack_size: int = 4
    action_mode: str = "discrete"  # "discrete" | "continuous"
    random_drop: bool = True
    pistons_start_low: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.action_mode not in ("discrete", "continuous"):
            raise ValueError(
                f"Invalid action_mode {self.action_mode!r}. "
                "Must be 'discrete' or 'continuous'"
            )
        # Every piston is an agent with its own policy, so the checkpoint store
        # and the environment must agree on how many there are.
        if self.num_agents != self.num_pistons:
            raise ValueError(
                f"num_agents ({self.num_agents}) must equal num_pistons "
                f"({self.num_pistons}); each piston carries its own policy."
            )
