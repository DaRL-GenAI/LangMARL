"""Core abstract base classes for LangMARL."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Trajectory:
    """A single episode trajectory."""

    task: dict
    steps: list[dict]  # [{agent_id, observation, action, ...}, ...]
    reward: float
    metadata: dict = field(default_factory=dict)


class BaseEnvironment(ABC):
    """Thin adapter over a third-party environment.

    Besides the three abstract rollout methods, an environment tells the
    centralized critic how to read its trajectories, via three hooks with
    language-task defaults: :attr:`ENV_NAME`, :meth:`prompts_dir`,
    :meth:`format_trajectory` and :meth:`critic_prompt_vars`.
    """

    #: Selects ``prompts/evaluation/game_contexts/<ENV_NAME>.json``.
    ENV_NAME: str = "language_task"

    #: Prefix for ``prompts/evaluation/templates/<TEMPLATE_PREFIX>_<paradigm>.json``,
    #: falling back to ``<paradigm>.json``.
    TEMPLATE_PREFIX: str = "language"

    #: Agent names this environment rolls out, e.g. ``piston_0`` vs ``agent_1``.
    AGENT_PREFIX: str = "agent"

    @abstractmethod
    def reset(self, task: dict) -> dict:
        """Reset environment, return initial observations."""
        ...

    @abstractmethod
    def step(self, agent_id: str, action: str) -> tuple[dict, float, bool, dict]:
        """Execute agent action, return (obs, reward, done, info)."""
        ...

    @abstractmethod
    def collect_trajectory(self, policies: dict[str, str], task: dict) -> Trajectory:
        """Run a full episode with given policies, return trajectory."""
        ...

    def prompts_dir(self) -> Path:
        """Directory holding this environment's critic prompts.

        An environment ships its prompts next to its code, at
        ``<package of the concrete subclass>/prompts/evaluation``. A custom
        environment that ships none falls back to the language-task prompts,
        which are generic enough to evaluate any sequential collaboration.
        """
        own = Path(inspect.getfile(type(self))).parent / "prompts" / "evaluation"
        if own.is_dir():
            return own
        return Path(__file__).parent.parent / "envs" / "language" / "prompts" / "evaluation"

    def format_trajectory(self, episode: dict, paradigm: str) -> str:
        """Render a trajectory as the text the critic will read."""
        from .trajectory import TrajectoryFormatter

        if paradigm == "central_credit":
            return TrajectoryFormatter.format_for_credit_assignment(episode)
        return TrajectoryFormatter.format_trajectory(episode)

    def critic_prompt_vars(self, paradigm: str, agents: list[str]) -> dict:
        """Extra ``str.format`` variables this environment's templates expect.

        The critic always supplies ``trajectory``, ``num_agents``, ``task_type``,
        ``role_context``, ``agent_evaluation_sections`` and
        ``agent_specific_criteria``; anything returned here is merged on top.
        """
        return {}


class BaseAgent(ABC):
    """An agent with a language policy."""

    @abstractmethod
    def act(self, observation: str, policy: str) -> str:
        """Given observation and policy, produce an action (text)."""
        ...


class BaseCritic(ABC):
    """Evaluates trajectories and assigns credit."""

    @abstractmethod
    def evaluate(self, trajectory: Trajectory, policies: dict[str, str]) -> dict:
        """Evaluate a trajectory. Returns evaluation dict with per-agent credits."""
        ...


class BaseReward(ABC):
    """Computes reward signals from trajectories."""

    @abstractmethod
    def compute(self, trajectory: Trajectory) -> float:
        """Compute reward for a trajectory."""
        ...


class BaseOptimizer(ABC):
    """Generates and applies language gradients."""

    @abstractmethod
    def generate_gradient(self, policy: str, evaluation: str, context: str) -> str:
        """Generate a language gradient (improvement instruction)."""
        ...

    @abstractmethod
    def apply_gradient(self, policy: str, gradient: str) -> str:
        """Apply gradient to policy, return updated policy."""
        ...

    @abstractmethod
    def aggregate_gradients(self, gradients: list[str]) -> str:
        """Aggregate multiple gradients into one."""
        ...

    def synthesize_policy(
        self,
        base_policy,
        gradient,
        agent_name: str = "agent",
    ) -> str:
        """Fold an aggregated gradient into the previous policy.

        The default appends it through :meth:`apply_gradient`; an optimizer
        that can rewrite a policy semantically should override this. A list is
        aggregated first, for callers that have not done so already.
        """
        if isinstance(gradient, (list, tuple)):
            gradient = self.aggregate_gradients(list(gradient))
        if not gradient:
            return base_policy
        return self.apply_gradient(base_policy, gradient)
