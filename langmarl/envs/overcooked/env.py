"""Overcooked environment adapter.

Wraps ``overcooked_ai_py`` into the unified :class:`BaseEnvironment` interface.
Both players are ProAgents whose system prompt is the language policy being
optimized, so a training iteration updates the two prompts that drive planning.

Ported from the standalone ``src/overcooked/`` trainer; the episode loop, LLM
plumbing and checkpointing now come from :mod:`langmarl.trainer`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Optional

from ...core.base import BaseEnvironment, Trajectory
from ...envs import register_env
from .trajectory import OvercookedTrajectoryFormatter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# Imported at module scope on purpose: when the overcooked extra is missing the
# ImportError reaches the environment registry, which skips registration and
# turns it into an actionable message at make_env() time. .utils pulls in
# overcooked_ai_py too, so it has to be guarded the same way.
from overcooked_ai_py.agents.agent import AgentGroup  # noqa: E402
from overcooked_ai_py.mdp.actions import Action  # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as _RawEnv  # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld  # noqa: E402

from .utils import NEW_LAYOUTS, make_agent, resolve_layout  # noqa: E402


@register_env("overcooked")
class OvercookedEnv(BaseEnvironment):
    """Two-player Overcooked, with a language policy per player."""

    ENV_NAME = "cramped_room"  # overridden per-layout in __init__
    TEMPLATE_PREFIX = "overcooked"
    AGENT_PREFIX = "agent"

    def __init__(self, config):
        """
        Args:
            config: :class:`OvercookedConfig` with layout, episode_horizon,
                p0_agent, p1_agent and llm fields.
        """
        self.layout = getattr(config, "layout", "cramped_room")
        if self.layout not in NEW_LAYOUTS:
            raise ValueError(
                f"Unknown layout {self.layout!r}. Available: {sorted(NEW_LAYOUTS)}"
            )
        # game_contexts/<layout>.json is how the critic learns this layout's map
        self.ENV_NAME = self.layout

        self.horizon = getattr(config, "episode_horizon", 400)
        self.p0_agent = getattr(config, "p0_agent", "ProAgent")
        self.p1_agent = getattr(config, "p1_agent", "ProAgent")
        self.num_agents = 2
        self.agent_names = ["agent_0", "agent_1"]

        llm = getattr(config, "actor_llm", None) or config.llm
        self.model = llm.model_string
        self.prompt_level = getattr(config, "prompt_level", "L1")
        self.belief_revision = getattr(config, "belief_revision", True)
        self.retrieval_method = getattr(config, "retrieval_method", "recent")
        self.K = getattr(config, "retrieval_k", 3)

        self.mdp = OvercookedGridworld.from_layout_name(resolve_layout(self.layout))
        # overcooked_ai 1.1.0 builds an env from an mdp through from_mdp();
        # 0.0.1, which the paper's experiments ran on, took it in the
        # constructor. Support whichever is installed.
        if hasattr(_RawEnv, "from_mdp"):
            self.env = _RawEnv.from_mdp(self.mdp, horizon=self.horizon)
        else:
            self.env = _RawEnv(self.mdp, horizon=self.horizon)

        # Alternating-yield state for centralized conflict resolution
        self.last_yielding_agent = None
        self.conflict_count = 0

        self.logger = logging.getLogger(__name__)

    # -------------------------------------------------------------- interface

    def reset(self, task: dict) -> dict:
        self.env.reset()
        self.last_yielding_agent = None
        self.conflict_count = 0
        return {"state": self.env.state}

    def step(self, agent_id: str, action: str) -> tuple[dict, float, bool, dict]:
        # Overcooked advances both players with one joint action; the episode
        # loop lives in collect_trajectory.
        raise NotImplementedError(
            "OvercookedEnv steps both players jointly; use collect_trajectory()."
        )

    def sample_tasks(self, num_samples: int, seed: Optional[int] = None) -> list[dict]:
        """Overcooked has no dataset: a 'task' is one episode on this layout."""
        return [
            {
                "task_id": f"episode_{i}",
                "layout": self.layout,
                "question": (
                    f"Cook and deliver as many soups as possible on the "
                    f"{self.layout} layout within {self.horizon} steps."
                ),
            }
            for i in range(num_samples)
        ]

    def collect_trajectory(self, policies: dict[str, str], task: dict) -> Trajectory:
        """Run one full episode with the given per-player policies."""
        with self._policy_file(policies) as policy_path:
            agents_list = self._build_agents(policy_path)
            team = AgentGroup(*agents_list)
            team.reset()
            self.reset(task)

            total_reward = 0.0
            transitions: list[dict] = []
            tokens = {"input": 0, "output": 0}

            for t in range(self.horizon):
                state = self.env.state
                state_string = self.env.mdp.state_string(state).replace("ø", "o")

                joint_action = team.joint_action(state)
                joint_action = self._resolve_conflict(state, joint_action)

                agents_info = {}
                for idx, agent in enumerate(agents_list):
                    if hasattr(agent, "get_transition_info"):  # ProAgent
                        info = agent.get_transition_info(state, joint_action[idx])
                        agents_info[str(idx)] = info
                        if info.get("gpt_query_response"):
                            tokens["input"] += _estimate_tokens(info.get("observation_to_gpt", ""))
                            tokens["output"] += _estimate_tokens(info.get("gpt_query_response", ""))
                    else:
                        player = state.players[idx]
                        agents_info[str(idx)] = {
                            "observation_to_gpt": None,
                            "gpt_query_response": None,
                            "parsed_ml_action": None,
                            "low_level_action": Action.to_char(joint_action[idx]),
                            "low_level_state": {
                                "position": list(player.position),
                                "orientation": list(player.orientation),
                                "held_object": (
                                    player.held_object.name if player.held_object else "nothing"
                                ),
                            },
                        }

                obs, reward, done, _env_info = self.env.step(joint_action)
                total_reward += reward

                transitions.append({
                    "timestep": t,
                    "state_string": state_string,
                    "agents": agents_info,
                    "ml_actions": getattr(obs, "ml_actions", None),
                    "instant_reward": reward,
                    "cumulative_reward": total_reward,
                    "done": done,
                })

                if done:
                    break

        metadata = {
            "layout": self.layout,
            "horizon": self.horizon,
            "p0_agent": self.p0_agent,
            "p1_agent": self.p1_agent,
            "total_reward": total_reward,
            "num_steps": len(transitions),
            "conflicts_resolved": self.conflict_count,
            "tokens": tokens,
        }
        return Trajectory(task=task, steps=transitions, reward=total_reward, metadata=metadata)

    # ------------------------------------------------------------ critic hooks

    def format_trajectory(self, episode: dict, paradigm: str) -> str:
        if paradigm == "central_credit":
            return OvercookedTrajectoryFormatter.format_for_credit_assignment(episode)
        return OvercookedTrajectoryFormatter.format_trajectory(episode)

    def critic_prompt_vars(self, paradigm: str, agents: list[str]) -> dict:
        """Supply the layout's role descriptions to the evaluation template."""
        roles_file = self.prompts_dir() / "role_descriptions.json"
        if not roles_file.exists():
            return {}
        with open(roles_file) as f:
            roles = json.load(f)
        return {"role_context": roles.get("role_context_summary", "")}

    # --------------------------------------------------------------- internals

    def _build_agents(self, policy_path):
        """Construct both players, handing ProAgents the current policy file."""
        agents = []
        for idx, agent_type in enumerate([self.p0_agent, self.p1_agent]):
            if agent_type == "ProAgent":
                agent = make_agent(
                    agent_type,
                    self.mdp,
                    self.layout,
                    model=self.model,
                    prompt_level=self.prompt_level,
                    belief_revision=self.belief_revision,
                    retrieval_method=self.retrieval_method,
                    K=self.K,
                    policy_path=policy_path,
                )
            else:
                agent = make_agent(agent_type, self.mdp, self.layout)
            agents.append(agent)
        return agents

    class _PolicyFile:
        """Context manager writing policies into the JSON layout ProAgent reads."""

        def __init__(self, policies: dict[str, str]):
            self.policies = policies
            self.path = None

        def __enter__(self):
            payload = {
                "current_iteration": 0,
                "policies": {
                    "iteration_0": {
                        f"agent_{i}": {"policy": self.policies.get(f"agent_{i}", "")}
                        for i in range(2)
                    }
                },
            }
            fd, self.path = tempfile.mkstemp(suffix="_policy.json", prefix="langmarl_")
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            return self.path

        def __exit__(self, *exc):
            if self.path and os.path.exists(self.path):
                os.unlink(self.path)
            return False

    def _policy_file(self, policies: dict[str, str]):
        return self._PolicyFile(policies)

    def _next_position(self, pos_and_or, action):
        """Where a player ends up after taking ``action``."""
        from overcooked_ai_py.mdp.actions import Direction

        pos = pos_and_or[0]

        # overcooked_ai 1.1.0 pairs an action with an info dict
        if isinstance(action, tuple) and len(action) == 2 and isinstance(action[1], dict):
            action = action[0]

        if action in (Action.STAY, Action.INTERACT):
            return pos
        if isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], int):
            return (pos[0] + action[0], pos[1] + action[1])
        if action == Direction.NORTH:
            return (pos[0], pos[1] - 1)
        if action == Direction.SOUTH:
            return (pos[0], pos[1] + 1)
        if action == Direction.EAST:
            return (pos[0] + 1, pos[1])
        if action == Direction.WEST:
            return (pos[0] - 1, pos[1])
        return pos

    def _resolve_conflict(self, state, actions):
        """Make one player yield when both target the same cell or swap places.

        Which player yields alternates, so neither is systematically penalized.
        """
        def split(action):
            if isinstance(action, tuple) and len(action) == 2 and isinstance(action[1], dict):
                return action[0], action[1], True
            return action, {}, False

        real_0, info_0, has_info_0 = split(actions[0])
        real_1, info_1, has_info_1 = split(actions[1])

        pos_or_0, pos_or_1 = state.players_pos_and_or[0], state.players_pos_and_or[1]
        cur_0, cur_1 = pos_or_0[0], pos_or_1[0]
        next_0 = self._next_position(pos_or_0, real_0)
        next_1 = self._next_position(pos_or_1, real_1)

        same_cell = next_0 == next_1 and next_0 != cur_0 and next_1 != cur_1
        swap = next_0 == cur_1 and next_1 == cur_0
        if not (same_cell or swap):
            return actions

        self.conflict_count += 1
        yielding = 1 if self.last_yielding_agent == 0 else 0
        self.last_yielding_agent = yielding
        self.logger.debug(
            "Conflict (%s): agent 0 -> %s, agent 1 -> %s; agent %d stays (#%d)",
            "swap" if swap else "same_cell", next_0, next_1, yielding, self.conflict_count,
        )

        if yielding == 0:
            return ((Action.STAY, info_0) if has_info_0 else Action.STAY, actions[1])
        return (actions[0], (Action.STAY, info_1) if has_info_1 else Action.STAY)


def _estimate_tokens(text: str) -> int:
    """Rough token count for text, used when the provider reports no usage."""
    if not text:
        return 0
    return max(len(text) // 4, int(len(text.split()) / 0.75))
