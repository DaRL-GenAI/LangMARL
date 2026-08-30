"""Centralized critic for evaluating multi-agent trajectories."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Formatter
from typing import Optional

from openai import OpenAI

from .base import BaseCritic, Trajectory
from .optimizer import PolicyGradientOptimizer


class PromptLoader:
    """Loader for evaluation prompts from external JSON files."""

    DEFAULT_DIR = Path(__file__).parent.parent / "envs" / "language" / "prompts" / "evaluation"

    def __init__(self, prompts_dir: Optional[Path] = None):
        self.prompts_dir = Path(prompts_dir) if prompts_dir else self.DEFAULT_DIR
        self._cache = {}

    def _load_json(self, filepath: Path) -> dict:
        str_path = str(filepath)
        if str_path not in self._cache:
            with open(filepath, 'r') as f:
                self._cache[str_path] = json.load(f)
        return self._cache[str_path]

    def load_game_context(self, game_name: str = "language_task") -> str:  # noqa: D401
        game_file = self.prompts_dir / "game_contexts" / f"{game_name}.json"
        if not game_file.exists():
            game_file = self.prompts_dir / "game_contexts" / "default.json"
        if not game_file.exists():
            return ""
        data = self._load_json(game_file)
        return data.get("game_context", "")

    def load_evaluation_template(self, paradigm: str, task_type: str = "language") -> str:
        template_file = self.prompts_dir / "templates" / f"{task_type}_{paradigm}.json"
        if not template_file.exists():
            template_file = self.prompts_dir / "templates" / f"{paradigm}.json"
        if not template_file.exists():
            raise ValueError(f"No template file found for paradigm '{paradigm}'")
        data = self._load_json(template_file)
        return data.get("template", "")

    def clear_cache(self):
        self._cache.clear()


class CentralizedCritic(BaseCritic):
    """Centralized critic supporting central_global and central_credit paradigms.

    Evaluates N-agent sequential collaboration trajectories using LLM-as-judge.
    """

    def __init__(self, config, env=None, prompts_dir: Optional[Path] = None):
        """
        Args:
            config: BaseConfig instance with paradigm, num_agents, critic_llm/llm fields.
            env: Optional BaseEnvironment. When given, its prompts directory,
                trajectory formatting and template variables are used, which is
                what makes the critic work for non-language environments.
            prompts_dir: Optional custom path, overriding the environment's.
        """
        self.paradigm = config.paradigm
        self.num_agents = config.num_agents
        self.task_type = getattr(config, 'task_type', 'qa')
        self.env = env

        self.env_name = getattr(env, 'ENV_NAME', 'language_task')
        self.template_prefix = getattr(env, 'TEMPLATE_PREFIX', 'language')
        agent_prefix = getattr(env, 'AGENT_PREFIX', 'agent')
        if agent_prefix == 'agent':
            self.all_agents = [f"agent_{i + 1}" for i in range(self.num_agents)]
        else:
            self.all_agents = [f"{agent_prefix}_{i}" for i in range(self.num_agents)]

        if prompts_dir is None and env is not None:
            prompts_dir = env.prompts_dir()
        self.prompt_loader = PromptLoader(prompts_dir)
        self._load_prompts()

        llm = getattr(config, 'critic_llm', None) or config.llm
        api_key = llm.get_api_key()
        if llm.base_url:
            self._client = OpenAI(base_url=llm.base_url, api_key=api_key)
        else:
            self._client = OpenAI(api_key=api_key)
        self._model = llm.model_string
        self.logger = logging.getLogger(__name__)
        self._warned_unused = False

    def _load_prompts(self):
        self.game_context = self.prompt_loader.load_game_context(self.env_name)
        self.evaluation_template = self.prompt_loader.load_evaluation_template(
            self.paradigm, self.template_prefix
        )

    def evaluate(self, trajectory: Trajectory, policies: dict[str, str]) -> dict:
        """Evaluate a trajectory. Returns dict with 'raw_response' and 'per_agent' credits."""
        # Convert Trajectory to episode dict for formatting
        episode = self._trajectory_to_episode(trajectory)

        if self.env is not None:
            traj_str = self.env.format_trajectory(episode, self.paradigm)
        else:
            from .trajectory import TrajectoryFormatter
            traj_str = (
                TrajectoryFormatter.format_trajectory(episode)
                if self.paradigm == "central_global"
                else TrajectoryFormatter.format_for_credit_assignment(episode)
            )

        if self.paradigm == "central_global":
            eval_prompt = self._create_global_prompt(traj_str, list(policies))
        else:
            eval_prompt = self._create_credit_prompt(traj_str, list(policies))

        eval_response = self._call_llm(eval_prompt)

        result = {"raw_response": eval_response, "paradigm": self.paradigm}

        if self.paradigm == "central_credit":
            agent_names = list(policies.keys())
            result["per_agent"] = PolicyGradientOptimizer.parse_credit_response(
                eval_response, agent_names
            )
        else:
            result["per_agent"] = {agent: eval_response for agent in policies}

        return result

    def _trajectory_to_episode(self, trajectory: Trajectory) -> dict:
        """Convert a Trajectory dataclass to legacy episode dict for formatting."""
        episode = {
            "task": trajectory.task,
            "task_type": trajectory.metadata.get("task_type", self.task_type),
            "transitions": trajectory.steps,
            "final_answer": trajectory.steps[-1].get("output", "") if trajectory.steps else "",
            "ground_truth": trajectory.task.get("ground_truth", ""),
            "reward": trajectory.reward,
            "total_reward": trajectory.reward,
            "episode_id": trajectory.metadata.get("episode_id", 0),
        }
        for k, v in trajectory.metadata.items():
            episode.setdefault(k, v)
        if "verified_reward" in trajectory.metadata:
            episode["verified_reward"] = trajectory.metadata["verified_reward"]
        if "verification_details" in trajectory.metadata:
            episode["verification_details"] = trajectory.metadata["verification_details"]
        if "evaluation_feedback" in trajectory.metadata:
            episode["evaluation_feedback"] = trajectory.metadata["evaluation_feedback"]
        return episode

    def _get_agent_role(self, agent_name: str, idx: Optional[int] = None,
                        n: Optional[int] = None) -> tuple[str, str]:
        if idx is None:
            idx = self._agent_index(agent_name)
        n = n or self.num_agents
        position = idx + 1
        is_first = (idx == 0)
        is_last = (idx == n - 1)

        if n == 1:
            short = "Sole Agent"
            detailed = (
                f"Agent {position} is the only agent. It receives the task directly and "
                "produces the final evaluated answer."
            )
        elif is_first:
            short = "First Responder"
            later = ", ".join([f"Agent {j + 1}" for j in range(1, n)])
            detailed = (
                f"Agent {position} receives the task first and provides an initial response. "
                f"This response is added to the shared message pool and seen by {later}. "
                f"Agent {n}'s output will be the final answer."
            )
        elif is_last:
            prev = ", ".join([f"Agent {j + 1}" for j in range(idx)])
            short = "Final Responder"
            detailed = (
                f"Agent {position} sees the original task and all previous responses from "
                f"{prev} in the shared message pool. "
                f"Agent {position} produces the FINAL answer that will be evaluated."
            )
        else:
            prev = ", ".join([f"Agent {j + 1}" for j in range(idx)])
            later = ", ".join([f"Agent {j + 1}" for j in range(position, n)])
            short = f"Intermediate Agent {position}"
            detailed = (
                f"Agent {position} sees the original task and all previous responses from "
                f"{prev} in the shared message pool. "
                f"Its response is in turn seen by {later}. "
                f"Agent {n}'s output will be the final answer."
            )
        return short, detailed

    def _get_agent_criteria(self, agent_name: str, idx: Optional[int] = None,
                            n: Optional[int] = None) -> str:
        if idx is None:
            idx = self._agent_index(agent_name)
        n = n or self.num_agents
        position = idx + 1
        is_first = (idx == 0)
        is_last = (idx == n - 1)
        short, _ = self._get_agent_role(agent_name, idx, n)

        lines = [f"**For Agent {position} ({short}):**"]
        if is_first:
            lines += [
                "- Did it correctly understand the task?",
                "- Did it provide useful initial analysis or reasoning?",
                "- Did it set up subsequent agents for success?",
                "- Was the level of detail appropriate?",
                "- Did it identify key aspects of the problem?",
            ]
        elif is_last:
            lines += [
                "- Did it effectively use previous agents' responses?",
                "- Did it produce an accurate/high-quality final answer?",
                "- Did it appropriately refine or extend earlier agents' work?",
                "- Did it catch and correct any errors from previous agents?",
                "- Did it add value beyond earlier responses?",
            ]
        else:
            lines += [
                "- Did it correctly process the task and previous agents' responses?",
                "- Did it add meaningful value to the collaborative chain?",
                "- Did it effectively bridge earlier and later agents?",
                "- Did it correct any errors from earlier agents?",
                "- Was its response at an appropriate level of detail for subsequent agents?",
            ]
        return "\n".join(lines)

    def _agent_index(self, agent_name: str) -> int:
        """Best-effort 0-based index for an agent name like agent_1 / piston_0."""
        if agent_name in self.all_agents:
            return self.all_agents.index(agent_name)
        try:
            idx = int(agent_name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 0
        # agent_N is 1-based by convention, everything else is 0-based
        return idx - 1 if agent_name.startswith("agent_") else idx

    def _build_common_vars(self, trajectory: str, agents: list[str]) -> dict:
        """Template variables every environment's critic templates may use."""
        n = len(agents)
        agent_lines = []
        for i, agent_name in enumerate(agents):
            short, detailed = self._get_agent_role(agent_name, i, n)
            agent_lines.append(f"- {agent_name} ({short}): {detailed}")
        role_context = (
            f"This is a {len(agents)}-agent cooperative system:\n"
            + "\n".join(agent_lines)
        )

        agent_keys_lines = ",\n".join(
            f'  "{name}": "(contribution assessment, what worked well, what could improve, '
            f'specific policy suggestions)"'
            for name in agents
        )
        agent_evaluation_sections = (
            "Your response MUST be a JSON dictionary with exactly these keys:\n"
            "```json\n{\n" + agent_keys_lines + "\n}\n```"
        )
        agent_specific_criteria = "\n\n".join(
            self._get_agent_criteria(name, i, n) for i, name in enumerate(agents)
        )

        return {
            "trajectory": trajectory,
            "num_agents": len(agents),
            "task_type": self.task_type,
            "role_context": role_context,
            "agent_evaluation_sections": agent_evaluation_sections,
            "agent_specific_criteria": agent_specific_criteria,
        }

    def _render(self, trajectory: str, agents: list[str]) -> str:
        """Fill the evaluation template, letting the environment add variables."""
        variables = self._build_common_vars(trajectory, agents)
        if self.env is not None:
            supplied = self.env.critic_prompt_vars(self.paradigm, agents)
            variables.update(supplied)
            self._warn_unused(supplied)
        template = f"{self.game_context}\n\n{self.evaluation_template}"
        try:
            return template.format(**variables)
        except KeyError as exc:
            raise KeyError(
                f"Evaluation template for env '{self.env_name}' (paradigm "
                f"'{self.paradigm}') needs variable {exc}, which the environment "
                f"did not supply via critic_prompt_vars(). Available: "
                f"{sorted(variables)}"
            ) from None

    def _warn_unused(self, supplied: dict) -> None:
        """Point out environment variables this paradigm's template never reads.

        Templates differ between paradigms -- central_global takes role_context,
        central_credit does not -- so guidance written for one silently vanishes
        in the other. Say so once rather than letting it disappear.
        """
        if not supplied or self._warned_unused:
            return
        self._warned_unused = True

        template = f"{self.game_context}\n\n{self.evaluation_template}"
        used = {name for _, name, _, _ in Formatter().parse(template) if name}
        unused = sorted(set(supplied) - used)
        if unused:
            self.logger.warning(
                "Environment %r supplied critic variables the %r template does not "
                "use, so they will not reach the critic: %s",
                self.env_name, self.paradigm, ", ".join(unused),
            )

    def _create_global_prompt(self, trajectory: str, agents: Optional[list] = None) -> str:
        return self._render(trajectory, agents or self.all_agents)

    def _create_credit_prompt(self, trajectory: str, agents: Optional[list] = None) -> str:
        return self._render(trajectory, agents or self.all_agents)

    def _call_llm(self, prompt: str, max_tokens: int = 1500) -> str:
        params = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        m = self._model.lower()
        if "o1" in m or "o3" in m:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        resp = self._client.chat.completions.create(**params)
        return resp.choices[0].message.content.strip()
