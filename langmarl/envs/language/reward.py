"""Verified reward computation for language task episodes.

- QA / Math: LLM-as-judge comparing model output against ground truth (0 or 1).
- Coding: HumanEval execution-based evaluation (0 or 1).
"""

from __future__ import annotations

import re
from typing import Tuple

from ...core.base import BaseReward, Trajectory

try:
    from human_eval.execution import check_correctness as _humaneval_check_correctness
    _HUMANEVAL_AVAILABLE = True
except ImportError:
    _HUMANEVAL_AVAILABLE = False


# ── LLM-as-judge prompt templates ────────────────────────────────────

_QA_SYSTEM = (
    "You are a precise and objective judge for question-answering tasks. "
    "Follow the output format exactly."
)

_QA_USER = """\
Question: {question}{context_block}

Ground Truth Answer: {ground_truth}

Model's Answer:
{response}

Is the model's answer CORRECT?
Accept paraphrasing, synonyms, and different but equivalent formulations.

Respond EXACTLY in this format (nothing else):

VERDICT: CORRECT
or
VERDICT: INCORRECT
"""

_MATH_SYSTEM = (
    "You are a precise mathematical judge. "
    "Follow the output format exactly."
)

_MATH_USER = """\
Math Problem:
{problem}

Correct Answer: {ground_truth}

Model's Solution:
{response}

Is the model's final numerical/symbolic answer mathematically equivalent to the correct answer?
Ignore differences in formatting (e.g. fractions vs decimals, \\boxed notation).

Respond EXACTLY in this format (nothing else):

VERDICT: CORRECT
or
VERDICT: INCORRECT
"""


# ── Reward generator ─────────────────────────────────────────────────

class VerifiedRewardGenerator(BaseReward):
    """Compute verified rewards: LLM-as-judge for QA/Math, execution for Coding."""

    def __init__(self, judge_model: str = "gpt-4o-mini", code_timeout: float = 10.0):
        self.judge_model = judge_model
        self.code_timeout = code_timeout
        self._client = None  # OpenAI-compatible client, injected via set_client

    def set_client(self, client):
        """Inject an OpenAI-compatible client for LLM-as-judge calls."""
        self._client = client

    def compute(self, trajectory: Trajectory) -> float:
        task_type = trajectory.metadata.get("task_type", "")
        task = trajectory.task
        final_answer = trajectory.metadata.get("final_answer", "")
        if not final_answer and trajectory.steps:
            final_answer = trajectory.steps[-1].get("output", trajectory.steps[-1].get("action", ""))

        if task_type == "qa":
            score, _ = self._judge_qa(final_answer, task)
        elif task_type == "math":
            score, _ = self._judge_math(final_answer, task)
        elif task_type == "coding":
            score, _ = self._exec_coding(final_answer, task)
        else:
            score = trajectory.reward
        return score

    # ------------------------------------------------------------------
    # QA
    # ------------------------------------------------------------------

    def _judge_qa(self, response: str, task: dict) -> Tuple[float, str]:
        question = task.get("question", "")
        ground_truth = task.get("ground_truth", "")
        context = task.get("context", "")
        context_block = f"\nContext:\n{context}" if context else ""
        prompt = _QA_USER.format(
            question=question,
            context_block=context_block,
            ground_truth=ground_truth,
            response=response,
        )
        return self._call_judge(prompt, system=_QA_SYSTEM)

    # ------------------------------------------------------------------
    # Math
    # ------------------------------------------------------------------

    def _judge_math(self, response: str, task: dict) -> Tuple[float, str]:
        problem = task.get("question", "")
        ground_truth = task.get("ground_truth", "")
        prompt = _MATH_USER.format(
            problem=problem,
            ground_truth=ground_truth,
            response=response,
        )
        return self._call_judge(prompt, system=_MATH_SYSTEM)

    # ------------------------------------------------------------------
    # Coding (HumanEval execution)
    # ------------------------------------------------------------------

    def _exec_coding(self, response: str, task: dict) -> Tuple[float, str]:
        if not _HUMANEVAL_AVAILABLE:
            raise RuntimeError(
                "human_eval package is required for coding evaluation. "
                "Install it with: pip install human-eval"
            )

        test_code = task.get("test", "")
        if not test_code:
            return 0.0, "No test cases provided."

        entry_point = task.get("entry_point", "")
        prompt = task.get("question", "")

        # Extract code block from markdown-fenced response
        match = re.search(r"```(?:python)?[ \t]*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        completion = match.group(1) if match else response

        problem = {
            "task_id": task.get("task_id", "unknown"),
            "prompt": prompt,
            "test": test_code,
            "entry_point": entry_point,
        }
        result = _humaneval_check_correctness(problem, completion, timeout=self.code_timeout)
        if result["passed"]:
            return 1.0, "Passed"
        return 0.0, f"Failed: {result['result']}"

    # ------------------------------------------------------------------
    # Shared judge helper
    # ------------------------------------------------------------------

    def _call_judge(self, prompt: str, system: str = "") -> Tuple[float, str]:
        if self._client is None:
            raise RuntimeError("No LLM client set. Call set_client() before evaluation.")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        result = self._client.chat.completions.create(
            model=self.judge_model,
            messages=messages,
            max_tokens=64,
        )
        text = result.choices[0].message.content.strip()
        if re.search(r"\bCORRECT\b", text) and not re.search(r"\bINCORRECT\b", text):
            return 1.0, text
        return 0.0, text
