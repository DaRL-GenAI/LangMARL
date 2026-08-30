"""Task data loading for language benchmarks.

Loads and samples tasks from benchmark directories for QA, math, and coding tasks.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional


class TaskLoader:
    """Load and sample tasks from benchmark directories."""

    TASK_TYPES = ["qa", "math", "coding"]

    TASK_DIRS = {
        "qa": "HotPotQA",
        "math": "MATH",
        "coding": "coding",
    }

    HOTPOT_RAW_FILE = "hotpot_dev_fullwiki_v1.json"

    def __init__(
        self,
        task_type: str,
        benchmark_path: str,
        data_limit: Optional[int] = None,
        train_test_split: float = 1.0,
        split_seed: int = 42,
    ):
        if task_type not in self.TASK_TYPES:
            raise ValueError(f"Invalid task_type '{task_type}'. Must be one of {self.TASK_TYPES}")

        self.task_type = task_type
        self.benchmark_path = Path(benchmark_path)
        self.data_limit = data_limit
        self.tasks: List[Dict] = []
        self._load_tasks()
        if self.data_limit is not None and self.data_limit > 0:
            self.tasks = self.tasks[: self.data_limit]

        self.train_tasks: List[Dict] = []
        self.test_tasks: List[Dict] = []
        self._split_tasks(train_test_split, split_seed)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_tasks(self):
        if self.task_type == "qa":
            raw_file = self.benchmark_path / self.HOTPOT_RAW_FILE
            if not raw_file.exists():
                raw_file = self.benchmark_path.parent / self.TASK_DIRS["qa"] / self.HOTPOT_RAW_FILE
            if raw_file.exists():
                self._load_hotpot_raw_json(raw_file)
                return

        # Fallback: JSONL format
        for filename in ("tasks.jsonl", "test_tasks.jsonl"):
            task_file = self.benchmark_path / filename
            if not task_file.exists():
                task_dir = self.benchmark_path.parent / self.TASK_DIRS.get(self.task_type, self.task_type)
                task_file = task_dir / filename
            if task_file.exists():
                self._load_jsonl(task_file)
                return

        raise FileNotFoundError(
            f"No task file found under {self.benchmark_path} for task_type='{self.task_type}'"
        )

    def _load_jsonl(self, path: Path):
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    task = self._normalize_task(json.loads(line))
                    self.tasks.append(task)
        if not self.tasks:
            raise ValueError(f"No tasks loaded from {path}")

    def _load_hotpot_raw_json(self, file_path: Path):
        with open(file_path, "r") as f:
            raw_data = json.load(f)

        for idx, item in enumerate(raw_data):
            passages = []
            for p_idx, (title, sentences) in enumerate(item.get("context", []), 1):
                passage_text = " ".join(sentences)
                passages.append(f"[Passage {p_idx}] {title}\n{passage_text}")
            context_str = "\n\n".join(passages)

            self.tasks.append({
                "task_id": f"qa_{idx:05d}",
                "type": "qa",
                "question": item["question"],
                "context": context_str,
                "ground_truth": item["answer"],
                "metadata": {
                    "source": "hotpotqa",
                    "difficulty": item.get("level", ""),
                    "type": item.get("type", ""),
                    "original_id": item.get("_id", ""),
                },
            })

    def _normalize_task(self, task: Dict) -> Dict:
        normalized = {
            "task_id": task.get("task_id", "unknown"),
            "type": task.get("type", self.task_type),
            "metadata": task.get("metadata", {}),
        }

        if self.task_type == "qa":
            normalized["question"] = task.get("question", task.get("problem", ""))
            normalized["context"] = task.get("context", "")
            normalized["ground_truth"] = task.get("answer", task.get("ground_truth", ""))
        elif self.task_type == "math":
            normalized["question"] = task.get("problem", task.get("question", ""))
            normalized["context"] = ""
            normalized["ground_truth"] = task.get("answer", "")
            normalized["solution"] = task.get("solution", "")
        elif self.task_type == "coding":
            normalized["question"] = task.get("problem", task.get("prompt", ""))
            normalized["context"] = ""
            normalized["ground_truth"] = task.get("canonical_solution", "")
            normalized["test"] = task.get("test", "")
            normalized["entry_point"] = task.get("entry_point", "")

        return normalized

    # ------------------------------------------------------------------
    # Splitting / sampling
    # ------------------------------------------------------------------

    def _split_tasks(self, train_ratio: float, seed: int):
        if train_ratio >= 1.0 or not self.tasks:
            self.train_tasks = self.tasks.copy()
            self.test_tasks = []
            return
        indices = list(range(len(self.tasks)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        n_train = max(1, int(len(self.tasks) * train_ratio))
        self.train_tasks = [self.tasks[i] for i in indices[:n_train]]
        self.test_tasks = [self.tasks[i] for i in indices[n_train:]]

    def sample_tasks(self, num_samples: int, seed: Optional[int] = None, split: str = "train") -> List[Dict]:
        pool = self.train_tasks if split == "train" else self.test_tasks
        if not pool:
            pool = self.tasks
        rng = random.Random(seed) if seed is not None else random.Random()
        if num_samples >= len(pool):
            return pool.copy()
        return rng.sample(pool, num_samples)

    def get_test_tasks(self) -> List[Dict]:
        return self.test_tasks.copy()

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def get_task_prompt(self, task: Dict) -> str:
        if self.task_type == "qa":
            prompt = f"Question: {task['question']}"
            if task.get("context"):
                prompt = f"Context: {task['context']}\n\n{prompt}"
            return prompt
        elif self.task_type == "math":
            return f"Problem: {task['question']}"
        elif self.task_type == "coding":
            return f"Complete the following Python function:\n\n{task['question']}"
        return task.get("question", "")

    def get_ground_truth(self, task: Dict) -> str:
        return task.get("ground_truth", "")

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)
