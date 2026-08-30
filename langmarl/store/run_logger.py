"""Structured event logger for training runs."""

from __future__ import annotations

import logging

from .base import BaseStore


class RunLogger:
    """Structured event logger for a training run."""

    def __init__(
        self,
        store: BaseStore,
        run_id: str,
        console: bool = True,
        level: str = "INFO",
    ):
        """
        Args:
            store: where the run's log file lives.
            run_id: identifies this run's logger.
            console: also mirror events to the terminal.
            level: threshold for the console only. The run's log file always
                keeps DEBUG, so quieting the terminal never loses the record.
        """
        self.store = store
        self.run_id = run_id
        self._logger = logging.getLogger(f"langmarl.{run_id}")
        self._logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers
        if not self._logger.handlers:
            log_path = store.get_log_path(run_id)
            if log_path:
                fh = logging.FileHandler(log_path)
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
                self._logger.addHandler(fh)

            if console:
                ch = logging.StreamHandler()
                ch.setLevel(getattr(logging, str(level).upper(), logging.INFO))
                ch.setFormatter(logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
                self._logger.addHandler(ch)

    def info(self, msg: str):
        self._logger.info(msg)

    def debug(self, msg: str):
        self._logger.debug(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str, exc: Exception = None):
        self._logger.error(msg, exc_info=exc)

    def iteration_start(self, iteration: int, policies: dict[str, str]):
        self._logger.info(f"{'='*60}")
        self._logger.info(f"Iteration {iteration} started | {len(policies)} agents")

    def iteration_end(self, iteration: int, stats: dict):
        self._logger.info(
            f"Iteration {iteration} done | "
            f"avg_reward={stats.get('avg_reward', 0):.3f} | "
            f"tokens={stats.get('total_tokens', 0)} | "
            f"cost=${stats.get('cost_usd', 0):.4f}"
        )
        self.store.append_metrics(self.run_id, {"type": "iteration", "iteration": iteration, **stats})

    def episode_saved(self, iteration: int, episode_id: int, reward: float):
        self._logger.debug(f"  Episode {episode_id} saved | reward={reward:.3f}")

    def evaluation_done(self, iteration: int, episode_id, paradigm: str, raw_response: str):
        self._logger.debug(f"  Eval episode {episode_id} [{paradigm}]")
        self.store.save_evaluation(self.run_id, iteration, episode_id, {
            "episode_id": episode_id,
            "paradigm": paradigm,
            "raw_response": raw_response,
        })

    def gradient_saved(self, iteration: int, agent_id: str, num_gradients: int):
        self._logger.info(f"  Gradient {agent_id} | {num_gradients} episode gradients aggregated")
