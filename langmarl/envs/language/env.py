"""Language task environment adapter.

Wraps env/lang_benchmark/ datasets (HotPotQA, MATH, HumanEval)
into the unified BaseEnvironment interface.

Implements sequential N-agent collaboration via a shared message pool:
    Task Question -> Agent 1 -> Agent 2 -> ... -> Agent N (Final Answer)
"""

from __future__ import annotations

import logging
from typing import Optional

from ...core.base import BaseEnvironment, Trajectory
from ...core.policy import AgentPolicy, default_agent_policy
from ...envs import register_env
from ...llm.client import LLMClient
from .reward import VerifiedRewardGenerator
from .task_loader import TaskLoader


@register_env("language")
class LanguageTaskEnv(BaseEnvironment):
    """Environment for language benchmark tasks (QA, Math, Coding).

    Delegates to env/lang_benchmark/ for task data.
    """

    def __init__(self, config):
        """
        Args:
            config: LanguageTaskConfig with task_type, benchmark_path, llm, num_agents.
        """
        self.task_type = getattr(config, "task_type", "qa")
        self.num_agents = config.num_agents
        self.agent_names = [f"agent_{i + 1}" for i in range(self.num_agents)]

        llm = getattr(config, "actor_llm", None) or config.llm
        self.llm_client = LLMClient(llm)

        # Reward evaluator
        judge_model = getattr(config, "judge_model", "gpt-4o-mini")
        code_timeout = getattr(config, "code_timeout", 10.0)
        self.reward_gen = VerifiedRewardGenerator(
            judge_model=judge_model, code_timeout=code_timeout,
        )
        self.reward_gen.set_client(self.llm_client.raw_client)

        # Task data
        benchmark_path = getattr(config, "benchmark_path", "")
        data_limit = getattr(config, "data_limit", None)
        train_test_split = getattr(config, "train_test_split", 1.0)
        split_seed = getattr(config, "split_seed", 42)
        if benchmark_path:
            self.task_loader = TaskLoader(
                self.task_type,
                benchmark_path,
                data_limit=data_limit,
                train_test_split=train_test_split,
                split_seed=split_seed,
            )
        else:
            self.task_loader = None

        self.logger = logging.getLogger(__name__)

    def reset(self, task: dict) -> dict:
        return {"task": task}

    def step(self, agent_id: str, action: str) -> tuple[dict, float, bool, dict]:
        return {}, 0.0, False, {}

    def collect_trajectory(self, policies: dict, task: dict) -> Trajectory:
        """Run N-agent sequential collaboration on a single task.

        Args:
            policies: dict mapping agent_name -> AgentPolicy (or legacy str).
            task: task dict with question, ground_truth, etc.
        """
        task_prompt = self._get_task_prompt(task)

        steps = []
        message_pool = []

        for i, agent_name in enumerate(self.agent_names):
            is_last = i == self.num_agents - 1

            policy = policies.get(
                agent_name,
                default_agent_policy(i, self.num_agents),
            )
            if isinstance(policy, str):
                policy = AgentPolicy.from_legacy(policy)
            agent_system = policy.combined

            # Build user input: task + all messages in the pool
            user_input = f"Task:\n{task_prompt}"
            for msg in message_pool:
                display = msg["agent"].replace("_", " ").title()
                user_input += f"\n\n{display}'s Response:\n{msg['content']}"
            if is_last:
                user_input += "\n\nNow provide your FINAL answer:"

            response, tokens = self.llm_client.chat_with_usage(agent_system, user_input)

            steps.append({
                "agent": agent_name,
                "agent_id": agent_name,
                "system_prompt": agent_system,
                "input": user_input,
                "output": response,
                "action": response,
                "tokens": tokens,
            })

            message_pool.append({"agent": agent_name, "content": response})

        final_answer = message_pool[-1]["content"]

        # Build trajectory with placeholder reward, then compute verified reward
        metadata = {
            "task_type": self.task_type,
            "final_answer": final_answer,
        }
        trajectory = Trajectory(task=task, steps=steps, reward=0.0, metadata=metadata)
        reward = self.reward_gen.compute(trajectory)
        trajectory.reward = reward
        metadata["evaluation_feedback"] = f"reward={reward}"

        return trajectory

    def sample_tasks(self, num_samples: int, seed: Optional[int] = None, split: str = "train") -> list[dict]:
        if self.task_loader:
            return self.task_loader.sample_tasks(num_samples, seed=seed, split=split)
        raise ValueError("No task_loader available. Provide benchmark_path in config.")

    def _get_task_prompt(self, task: dict) -> str:
        if self.task_loader:
            return self.task_loader.get_task_prompt(task)
        question = task.get("question", task.get("problem", ""))
        context = task.get("context", "")
        if context:
            return f"Context: {context}\n\nQuestion: {question}"
        return question
