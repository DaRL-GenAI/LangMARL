"""Pistonball environment adapter.

Wraps PettingZoo's ``pistonball_v6`` into the unified :class:`BaseEnvironment`
interface: every piston is an agent driven by its own language policy, and the
team shares one reward for moving the ball toward the left wall.

Ported from the standalone ``src/pistonball/`` trainer; the episode loop, LLM
plumbing and checkpointing now come from :mod:`langmarl.trainer`.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.base import BaseEnvironment, Trajectory
from ...envs import register_env
from ...llm.client import LLMClient
from .observation import (
    PistonballObservationFormatter,
    make_env,
    parse_action_from_response,
    reset_pistons_to_lowest,
)
from .trajectory import PistonballTrajectoryFormatter

DEFAULT_POLICY = "Push up when the ball is near, retract after it passes."


def _to_native(obj):
    """Make numpy values JSON-serialisable."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


@register_env("pistonball")
class PistonballEnv(BaseEnvironment):
    """Cooperative Pistonball with one language policy per piston."""

    ENV_NAME = "pistonball"
    TEMPLATE_PREFIX = "pistonball"
    AGENT_PREFIX = "piston"

    def __init__(self, config):
        """
        Args:
            config: :class:`PistonballConfig` with num_pistons, max_cycles,
                frame_size, action_mode, num_agents and llm fields.
        """
        self.num_pistons = getattr(config, "num_pistons", 20)
        self.max_cycles = getattr(config, "max_cycles", 125)
        self.action_mode = getattr(config, "action_mode", "discrete")
        frame_size = getattr(config, "frame_size", 84)
        self.frame_size = (frame_size, frame_size) if isinstance(frame_size, int) else tuple(frame_size)
        self.stack_size = getattr(config, "stack_size", 4)
        self.random_drop = getattr(config, "random_drop", True)
        self.pistons_start_low = getattr(config, "pistons_start_low", False)

        if self.action_mode not in ("discrete", "continuous"):
            raise ValueError(
                f"action_mode must be 'discrete' or 'continuous', got {self.action_mode!r}"
            )

        # num_agents is derived from the environment, not set independently
        self.num_agents = self.num_pistons
        self.agent_names = [f"piston_{i}" for i in range(self.num_pistons)]

        llm = getattr(config, "actor_llm", None) or config.llm
        self.llm_client = LLMClient(llm)

        self.obs_formatter = PistonballObservationFormatter()
        self.logger = logging.getLogger(__name__)
        self._env = None

    # ------------------------------------------------------------------ setup

    @property
    def env(self):
        """The wrapped PettingZoo environment, built on first use."""
        if self._env is None:
            self._env = make_env(
                num_pistons=self.num_pistons,
                max_cycles=self.max_cycles,
                frame_size=self.frame_size,
                stack_size=self.stack_size,
                continuous=(self.action_mode == "continuous"),
                random_drop=self.random_drop,
            )
        return self._env

    # -------------------------------------------------------------- interface

    def reset(self, task: dict) -> dict:
        obs_dict, _info = self.env.reset(seed=task.get("seed"))
        if self.pistons_start_low:
            reset_pistons_to_lowest(self.env)
        return obs_dict

    def step(self, agent_id: str, action: str) -> tuple[dict, float, bool, dict]:
        # Pistonball steps all pistons simultaneously; collect_trajectory drives
        # the joint step, so this single-agent hook is intentionally unused.
        raise NotImplementedError(
            "PistonballEnv steps all agents jointly; use collect_trajectory()."
        )

    def sample_tasks(self, num_samples: int, seed: Optional[int] = None) -> list[dict]:
        """Pistonball has no dataset: a 'task' is one episode with its own seed."""
        base = 0 if seed is None else seed
        return [
            {
                "task_id": f"episode_{i}",
                "seed": base + i,
                "question": (
                    f"Coordinate {self.num_pistons} pistons to push the ball to the "
                    f"left wall within {self.max_cycles} steps."
                ),
            }
            for i in range(num_samples)
        ]

    def collect_trajectory(self, policies: dict[str, str], task: dict) -> Trajectory:
        """Run one full episode with the given per-piston policies."""
        env = self.env
        obs_dict = self.reset(task)

        total_reward = 0.0
        transitions: list[dict] = []
        tokens = {"input": 0, "output": 0}

        for t in range(self.max_cycles):
            # Global state is what the centralized critic will read.
            global_state = self.obs_formatter.get_env_state(env)
            global_state_text = self.obs_formatter.format_global_state(env)

            actions = {}
            agent_infos = {}

            for agent_name in env.possible_agents:
                if agent_name not in obs_dict:
                    continue
                agent_idx = env.possible_agents.index(agent_name)

                # Local observation is all the actor sees (decentralized execution).
                local_obs_text = self.obs_formatter.format_local_observation(env, agent_idx)
                policy = policies.get(agent_name, DEFAULT_POLICY)
                action, response, prompt, step_tokens = self._act(env, agent_name, policy)
                tokens["input"] += step_tokens["input"]
                tokens["output"] += step_tokens["output"]

                agent_infos[agent_name] = {
                    "action": action,
                    "action_name": self.obs_formatter.format_action(action, self.action_mode),
                    "response": response,
                    "prompt": prompt,
                    "local_observation": local_obs_text,
                }
                actions[agent_name] = self._convert_action_for_env(action)

            obs_dict, rewards, terms, truncs, _infos = env.step(actions)

            step_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            total_reward += step_reward
            done = any(terms.values()) or any(truncs.values())

            transitions.append({
                "timestep": t,
                "global_state": {
                    "text": global_state_text,
                    "ball_position": _to_native(global_state["ball_position"]),
                    "ball_velocity": _to_native(global_state["ball_velocity"]),
                    "pistons": _to_native(global_state["pistons"]),
                },
                "agents": {
                    name: {
                        "action": _to_native(info["action"]),
                        "action_name": info["action_name"],
                        "local_observation": info["local_observation"],
                        "observation_to_gpt": info["prompt"],
                        "gpt_query_response": info["response"],
                    }
                    for name, info in agent_infos.items()
                },
                "instant_reward": step_reward,
                "cumulative_reward": total_reward,
                "done": done,
            })

            if done:
                break

        metadata = {
            "num_pistons": self.num_pistons,
            "max_cycles": self.max_cycles,
            "action_mode": self.action_mode,
            "total_reward": total_reward,
            "num_steps": len(transitions),
            "tokens": tokens,
        }
        return Trajectory(task=task, steps=transitions, reward=total_reward, metadata=metadata)

    # ------------------------------------------------------------ critic hooks

    def format_trajectory(self, episode: dict, paradigm: str) -> str:
        return PistonballTrajectoryFormatter.format_trajectory(episode)

    def critic_prompt_vars(self, paradigm: str, agents: list[str]) -> dict:
        """Supply the pistonball-specific template variables."""
        piston_eval_format = "\n".join(
            f"[{name.upper()} EVALUATION]\n- Performance assessment:\n- Specific improvements:"
            for name in agents
        )
        third = max(1, len(agents) // 3)
        groups = {
            "left": agents[:third],
            "middle": agents[third:2 * third],
            "right": agents[2 * third:],
        }
        role_context_credit = "\n".join(
            f"* **{name.capitalize()} Pistons** ({', '.join(members)}): "
            f"closest to {'the goal wall' if name == 'left' else 'the ball spawn' if name == 'right' else 'the centre of the field'}"
            for name, members in groups.items() if members
        )
        role_context_credit += f"\n* Total {len(agents)} pistons, each needs individual evaluation"

        return {
            "num_pistons": len(agents),
            "all_agents": ", ".join(agents),
            "piston_eval_format": piston_eval_format,
            "role_context_credit": role_context_credit,
        }

    # --------------------------------------------------------------- internals

    def _act(self, env, agent_name: str, policy: str):
        """Query the LLM for one piston's action."""
        prompt = self.obs_formatter.format_agent_prompt(
            env=env,
            agent_name=agent_name,
            policy=policy,
            action_mode=self.action_mode,
        )
        response, step_tokens = self.llm_client.chat_with_usage(
            "You control a single piston in a cooperative Pistonball game. "
            "Answer with your action only.",
            prompt,
        )
        action = parse_action_from_response(response, self.action_mode)
        action_str = self.obs_formatter.format_action(action, self.action_mode)
        return action, f"Action: {action_str}", prompt, step_tokens

    def _convert_action_for_env(self, action):
        """Map a parsed action into the shape pistonball_v6 expects."""
        if self.action_mode == "continuous":
            return np.array([action], dtype=np.float32)
        return action
