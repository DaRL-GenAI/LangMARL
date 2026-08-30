"""Generic Monte Carlo trainer for any LangMARL environment."""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from ..config.base import BaseConfig
from ..core.base import BaseCritic, BaseEnvironment, BaseOptimizer, BaseReward, Trajectory
from ..llm.token_tracker import TokenTracker
from ..store.base import BaseStore
from ..store.checkpoint import PolicyCheckpoint
from ..store.local import LocalStore
from ..store.run_logger import RunLogger
from ..store.trajectory_store import TrajectoryStore
from .callbacks import Callback


class MonteCarloTrainer:
    """Generic Monte Carlo trainer for any LangMARL environment.

    Five-phase iteration:
      1. Load policies from checkpoint
      2. Generate trajectories (parallel)
      3. Evaluate & generate gradients
      4. Aggregate & apply gradients
      5. Save updated policies
    """

    def __init__(
        self,
        config: BaseConfig,
        env: BaseEnvironment,
        critic: BaseCritic,
        optimizer: BaseOptimizer,
        reward_fn: Optional[BaseReward] = None,
        store: Optional[BaseStore] = None,
        callbacks: Optional[list[Callback]] = None,
    ):
        self.config = config
        self.env = env
        self.critic = critic
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.callbacks = callbacks or []

        # Storage
        self.store = store or LocalStore(config.experiment_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = self.store.create_run(f"{config.exp_name}_{timestamp}", config)
        self.checkpoint = PolicyCheckpoint(self.store, self.run_id, config.num_agents)
        self.trajectory_store = TrajectoryStore(self.store, self.run_id)
        self.run_logger = RunLogger(self.store, self.run_id)

        # Token tracking
        llm = config.llm
        self.token_tracker = TokenTracker(
            model=llm.model_string if llm else "gpt-4o-mini",
            input_price=llm.input_price_per_million if llm else None,
            output_price=llm.output_price_per_million if llm else None,
        )

        self._stats_lock = threading.Lock()
        self._should_stop = False

    def train(self, num_iterations: Optional[int] = None):
        """Main training loop."""
        n = num_iterations or self.config.num_iterations

        # Auto-detect resume point
        latest = self.checkpoint.store.latest_checkpoint(self.run_id)
        start = (latest + 1) if latest is not None else self.config.start_iteration

        self.run_logger.info(f"Starting training from iteration {start} to {n}")

        for i in range(start, n):
            if self._should_stop:
                self.run_logger.info("Training stopped early")
                break

            for cb in self.callbacks:
                cb.on_iteration_start(i, self)

            stats = self.train_one_iteration(i)

            for cb in self.callbacks:
                cb.on_iteration_end(i, stats, self)

        self.run_logger.info("Training complete!")
        return self.store.load_metrics(self.run_id)

    def train_one_iteration(self, iteration: int) -> dict:
        """Run a single training iteration."""
        self.run_logger.info(f"\n{'='*60}\nIteration {iteration}\n{'='*60}")

        # Phase 1: Load policies
        policies = self.checkpoint.get_policies()
        base_policies = dict(policies)
        self.run_logger.iteration_start(iteration, policies)

        # Phase 2: Collect trajectories
        num_traj = self.config.trajectories_per_iteration
        existing = self.trajectory_store.count(iteration)

        if existing >= num_traj:
            self.run_logger.info(f"Loading {num_traj} existing trajectories")
            trajectories = self.trajectory_store.load(iteration, limit=num_traj)
        else:
            self.run_logger.info(f"Generating {num_traj} trajectories")
            trajectories = self._collect_trajectories(policies, iteration)

        # Phase 2b: Compute verified rewards if available
        if self.reward_fn:
            for traj in trajectories:
                traj.reward = self.reward_fn.compute(traj)

        # Phase 3: Evaluate & generate gradients
        gradient_trajectories = trajectories
        if self.config.mini_batch_size and self.config.mini_batch_size < len(trajectories):
            gradient_trajectories = random.sample(trajectories, self.config.mini_batch_size)

        gradients = self._evaluate_and_generate_gradients(gradient_trajectories, base_policies)

        # Phase 4: Aggregate, then update. Each trajectory's gradient was
        # generated and is stored on its own; LLM_agg reconciles the list into
        # one gradient, and LLM_opt folds that into the previous policy.
        new_policies = {}
        for agent, grads in gradients.items():
            if not grads:
                new_policies[agent] = base_policies[agent]
            else:
                aggregated = self.optimizer.aggregate_gradients(grads)
                new_policies[agent] = self.optimizer.synthesize_policy(
                    base_policies[agent], aggregated, agent_name=agent
                )
                self.store.save_gradients(self.run_id, iteration, agent, grads, aggregated)
                self.run_logger.gradient_saved(iteration, agent, len(grads))

        # Phase 5: Save checkpoint
        stats = self._collect_stats(trajectories)
        self.checkpoint.save_policies(iteration + 1, new_policies, stats)
        self.run_logger.iteration_end(iteration, stats)

        return stats

    def _collect_trajectories(
        self,
        policies: dict[str, str],
        iteration: int,
    ) -> list[Trajectory]:
        """Generate trajectories using the environment."""
        num_traj = self.config.trajectories_per_iteration
        trajectories = []

        # Sample tasks from the environment
        if hasattr(self.env, 'sample_tasks'):
            tasks = self.env.sample_tasks(num_traj)
        else:
            tasks = [{"task_id": f"task_{i}", "question": f"Task {i}"} for i in range(num_traj)]

        def _run_one(idx, task):
            traj = self.env.collect_trajectory(policies, task)
            self.trajectory_store.save(iteration, idx, traj)
            self.run_logger.episode_saved(iteration, idx, traj.reward)
            return traj

        max_workers = self.config.max_workers
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_run_one, i, task): i
                    for i, task in enumerate(tasks)
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                    trajectories.append(future.result())
        else:
            for i, task in enumerate(tqdm(tasks, desc="Generating")):
                trajectories.append(_run_one(i, task))

        return trajectories

    def _evaluate_and_generate_gradients(
        self,
        trajectories: list[Trajectory],
        base_policies: dict[str, str],
    ) -> dict[str, list[str]]:
        """Evaluate trajectories and generate per-agent gradients."""
        accumulated = {agent: [] for agent in base_policies}

        def _eval_one(traj: Trajectory):
            eval_result = self.critic.evaluate(traj, base_policies)
            raw_response = eval_result.get("raw_response", "")
            episode_id = traj.metadata.get("episode_id", 0)
            self.run_logger.evaluation_done(
                0, episode_id, self.config.paradigm, raw_response
            )

            task_context = traj.task.get("question", traj.task.get("problem", ""))[:800]
            per_agent = eval_result.get("per_agent", {})

            gradients = {}
            if self.config.paradigm == "central_global":
                shared_grad = self.optimizer.generate_shared_gradient(raw_response, task_context)
                for agent in base_policies:
                    gradients[agent] = shared_grad
            else:
                for agent, agent_eval in per_agent.items():
                    gradients[agent] = self.optimizer.generate_gradient(
                        base_policies[agent], agent_eval, task_context, agent
                    )
            return gradients

        max_workers = getattr(self.config, 'optimizer_workers', 1)
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_eval_one, t): t for t in trajectories}
                for f in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                    ep_grads = f.result()
                    with self._stats_lock:
                        for agent, grad in ep_grads.items():
                            accumulated[agent].append(grad)
        else:
            for traj in tqdm(trajectories, desc="Evaluating"):
                ep_grads = _eval_one(traj)
                for agent, grad in ep_grads.items():
                    accumulated[agent].append(grad)

        return accumulated

    def _collect_stats(self, trajectories: list[Trajectory]) -> dict:
        rewards = [t.reward for t in trajectories]
        token_stats = self.token_tracker.get_stats()
        stats = {
            "paradigm": self.config.paradigm,
            "num_episodes": len(trajectories),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "min_reward": min(rewards, default=0.0),
            "max_reward": max(rewards, default=0.0),
            "rewards": rewards,
            **{k: token_stats[k] for k in ['input_tokens', 'output_tokens', 'total_tokens', 'cost_usd']},
        }
        return stats
