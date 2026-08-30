"""LLM-based policy gradient optimizer.

Gradient pipeline:
  1. generate_gradient()        -- per-agent improvement instruction (central_credit)
  2. generate_shared_gradient() -- shared instruction for all agents (central_global)
  3. apply_gradient()           -- base_policy + gradient (static helper)
  4. aggregate_gradients()      -- combine multiple gradients from multiple episodes
  5. parse_credit_response()    -- parse per-agent credit from evaluator response
"""

import logging
import re
from typing import Dict, List

from openai import OpenAI

from .base import BaseOptimizer

_PER_AGENT_GRADIENT_PROMPT = """\
You are coaching an AI agent to improve its problem-solving strategy.

## Agent's Current Instructions
{policy}

## Problem Being Solved
{task_context}

## Evaluation of the Agent's Last Attempt
{evaluation}

## Your Task
Write a concise, specific improvement note (2-4 sentences) that:
- Identifies the exact reasoning or strategy mistake made in this attempt
- Gives a concrete, actionable instruction for this type of problem
- Is grounded in the specific case above (not generic advice)

Write the improvement note directly (no preamble or headers):"""


_SHARED_GRADIENT_PROMPT = """\
You are coaching a multi-agent AI team to improve their problem-solving strategy.

## Problem Being Solved
{task_context}

## Team Evaluation
{evaluation}

## Your Task
Write a concise, specific improvement note for the whole team (2-4 sentences) that:
- Identifies the exact team-level reasoning or strategy mistake
- Gives a concrete, actionable instruction for this type of problem
- Is grounded in the specific case above (not generic advice)

Write the improvement note directly (no preamble or headers):"""


_SYNTHESIZE_POLICY_PROMPT = """\
You are an expert prompt engineer.  Your task is to produce an improved version
of an AI agent's system prompt (policy) by integrating concrete improvement
feedback from recent problem-solving attempts.

## Original Policy
{base_policy}

## Improvement Feedback (from one or more attempts)
{feedback}

## Instructions
Write a NEW, self-contained system prompt that:
1. Preserves the core role and responsibilities from the original policy.
2. Integrates the specific improvement suggestions from the feedback — make
   the instructions concrete and actionable.
3. Is concise: similar length to the original policy (do NOT pad with filler).
4. Reads as a standalone prompt — do NOT reference "previous attempts",
   "feedback", or "improvements".  It should look like it was always written
   this way.

Write the improved policy directly (no preamble, no markdown fences):"""


_AGGREGATE_GRADIENT_PROMPT = """\
You are consolidating language policy gradients for a single agent in a
cooperative multi-agent system.

Each gradient below was derived from a *different* trajectory. They may
reinforce each other, contradict each other, or reflect noise particular to one
episode rather than a real weakness in the agent's policy.

## Gradients
{gradients}

## Instructions
Produce ONE consolidated improvement instruction that:
1. Keeps guidance supported by several trajectories — repeated signal is the
   reliable signal.
2. Resolves contradictions.  Where gradients disagree, prefer the direction
   backed by more trajectories; if both are valid, say under which condition
   each applies.
3. Drops one-off observations that look specific to a single episode rather
   than to the policy itself.
4. Is concrete, actionable, and no longer than the longest single gradient.

Write the consolidated instruction directly (no preamble, no markdown fences):"""


class PolicyGradientOptimizer(BaseOptimizer):
    """LLM-based policy gradient optimizer."""

    def __init__(self, llm_config):
        """
        Args:
            llm_config: LLMConfig instance for the optimizer LLM.
        """
        from ..config.llm import LLMConfig
        if not isinstance(llm_config, LLMConfig):
            raise TypeError(f"Expected LLMConfig, got {type(llm_config)}")
        api_key = llm_config.get_api_key()
        if llm_config.base_url:
            self._client = OpenAI(base_url=llm_config.base_url, api_key=api_key)
        else:
            self._client = OpenAI(api_key=api_key)
        self._model = llm_config.model_string
        self.logger = logging.getLogger(__name__)

    def generate_gradient(
        self,
        policy: str,
        evaluation: str,
        context: str,
        agent_name: str = "agent",
    ) -> str:
        """Generate a case-specific improvement instruction for one agent."""
        prompt = _PER_AGENT_GRADIENT_PROMPT.format(
            policy=policy,
            task_context=context[:800],
            evaluation=evaluation,
        )
        try:
            gradient = self._llm_call(prompt, max_tokens=400)
            self.logger.debug(
                "Gradient for %s (%d chars): %.80s ...", agent_name, len(gradient), gradient
            )
            return gradient
        except Exception as exc:
            self.logger.warning(
                "Gradient generation failed for %s, using evaluation as fallback: %s",
                agent_name, exc,
            )
            return evaluation

    def generate_shared_gradient(self, evaluation: str, task_context: str) -> str:
        """Generate one shared gradient for the whole team (central_global)."""
        prompt = _SHARED_GRADIENT_PROMPT.format(
            task_context=task_context[:800],
            evaluation=evaluation,
        )
        try:
            gradient = self._llm_call(prompt, max_tokens=400)
            self.logger.debug("Shared gradient (%d chars): %.80s ...", len(gradient), gradient)
            return gradient
        except Exception as exc:
            self.logger.warning(
                "Shared gradient generation failed, using evaluation as fallback: %s", exc
            )
            return evaluation

    def synthesize_policy(
        self,
        base_policy,
        gradient,
        agent_name: str = "agent",
    ) -> str:
        """Rewrite a policy around an aggregated gradient.

        This is the paper's ``LLM_opt`` -- the second half of
        ``pi <- LLM_opt(pi, LLM_agg({d_pi}))``.  It expects the gradient that
        :meth:`aggregate_gradients` already reconciled, and folds it into the
        *previous* iteration's policy, producing a self-contained prompt rather
        than a policy with notes stapled to it.

        A list is accepted for convenience and aggregated first; the trainer
        passes the aggregated string it already computed, so the reconciliation
        call happens exactly once per agent per iteration.

        Falls back to :meth:`apply_gradient` if the call fails, so a flaky API
        degrades the update instead of losing the iteration.
        """
        from .policy import AgentPolicy

        if isinstance(base_policy, AgentPolicy):
            base_policy = base_policy.combined
        if isinstance(gradient, (list, tuple)):
            gradient = self.aggregate_gradients(list(gradient))
        if not gradient:
            return base_policy

        feedback = gradient
        prompt = _SYNTHESIZE_POLICY_PROMPT.format(
            base_policy=base_policy,
            feedback=feedback,
        )
        try:
            new_policy = self._llm_call(prompt, max_tokens=1024)
            self.logger.debug(
                "Synthesized policy for %s (%d -> %d chars)",
                agent_name, len(base_policy), len(new_policy),
            )
            return new_policy
        except Exception as exc:
            self.logger.warning(
                "Policy synthesis failed for %s, falling back to concatenation: %s",
                agent_name, exc,
            )
            return self.apply_gradient(base_policy, feedback)

    @staticmethod
    def apply_gradient(base_policy, gradient: str) -> str:
        """Apply a gradient to a policy, returning the new system prompt.

        The base is never modified and the feedback section is *replaced*, not
        appended to, so a policy stays the same size no matter how many
        iterations it has been through. ``base_policy`` may be an
        :class:`AgentPolicy` or a flat string from an older checkpoint; a flat
        string is parsed first, which also collapses any gradients a previous
        version accumulated.
        """
        from .policy import AgentPolicy

        policy = (
            base_policy if isinstance(base_policy, AgentPolicy)
            else AgentPolicy.from_legacy(base_policy)
        )
        return policy.with_gradient(gradient).combined

    @staticmethod
    def join_gradients(gradients: List[str]) -> str:
        """Concatenate gradients verbatim, with no LLM in the loop."""
        if not gradients:
            return ""
        if len(gradients) == 1:
            return gradients[0]
        return "\n\n---\n\n".join(gradients)

    def aggregate_gradients(self, gradients: List[str]) -> str:
        """Semantically integrate one iteration's gradients into a single one.

        This is the paper's ``LLM_agg``: the gradients come from different
        trajectories, so they repeat, contradict, and carry per-episode noise.
        One LLM call reconciles them, keeping what several trajectories agree
        on and dropping what only one saw.

        A single gradient needs no reconciliation and is returned untouched.
        If the call fails, the gradients are concatenated so the iteration is
        degraded rather than lost.
        """
        if not gradients:
            return ""
        if len(gradients) == 1:
            return gradients[0]

        numbered = "\n\n".join(
            f"### Gradient {i} (from trajectory {i})\n{g}"
            for i, g in enumerate(gradients, start=1)
        )
        try:
            aggregated = self._llm_call(
                _AGGREGATE_GRADIENT_PROMPT.format(gradients=numbered),
                max_tokens=600,
            )
            self.logger.debug(
                "Aggregated %d gradients into %d chars", len(gradients), len(aggregated)
            )
            return aggregated
        except Exception as exc:
            self.logger.warning(
                "Gradient aggregation failed, falling back to concatenation: %s", exc
            )
            return self.join_gradients(gradients)

    @staticmethod
    def parse_credit_response(response: str, agent_names: List[str]) -> Dict[str, str]:
        """Parse per-agent evaluations from a credit-assignment LLM response.

        Handles:
        - Overcooked: [AGENT 0 EVALUATION] / [AGENT 1 EVALUATION]
        - Language task: JSON dict {"agent_1": "...", "agent_2": "..."}
        - Pistonball: [PISTON_N EVALUATION] markers
        - Fallback: returns full response for all agents.
        """
        if not agent_names:
            return {}

        result: Dict[str, str] = {}

        # Overcooked
        overcooked_names = {'0', '1', 'agent_0', 'agent_1'}
        if set(agent_names).issubset(overcooked_names) or (
            len(agent_names) == 2 and any(n in overcooked_names for n in agent_names)
        ):
            pattern = r'\[AGENT\s+(\d+)\s+EVALUATION\](.*?)(?=\[AGENT\s+\d+\s+EVALUATION\]|$)'
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                idx_to_text = {m[0]: m[1].strip() for m in matches}
                for name in agent_names:
                    idx = name.split('_')[-1] if '_' in name else name
                    if idx in idx_to_text:
                        result[name] = idx_to_text[idx]
                if result:
                    for name in agent_names:
                        if name not in result:
                            result[name] = response
                    return result

        # Language task: JSON dict
        if any(n.startswith('agent_') and n[6:].isdigit() for n in agent_names):
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{[^{}]*"agent_\d+"[^{}]*\})', response, re.DOTALL)
            if json_match:
                try:
                    import json as _json
                    parsed = _json.loads(json_match.group(1))
                    for name in agent_names:
                        if name in parsed:
                            result[name] = str(parsed[name])
                    if result:
                        for name in agent_names:
                            if name not in result:
                                result[name] = response
                        return result
                except Exception:
                    pass

        # Pistonball
        if any(n.startswith('piston_') for n in agent_names):
            pattern = r'\[PISTON[_\s]?(\d+)\s+EVALUATION\](.*?)(?=\[PISTON[_\s]?\d+\s+EVALUATION\]|$)'
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                idx_to_text = {m[0]: m[1].strip() for m in matches}
                for name in agent_names:
                    idx = name.split('_')[-1]
                    if idx in idx_to_text:
                        result[name] = idx_to_text[idx]
                if result:
                    for name in agent_names:
                        if name not in result:
                            result[name] = response
                    return result

            third = len(agent_names) // 3
            group_assignments = {}
            for i, name in enumerate(sorted(agent_names, key=lambda x: int(x.split('_')[-1]))):
                if i < third:
                    group_assignments[name] = 'left'
                elif i < 2 * third:
                    group_assignments[name] = 'middle'
                else:
                    group_assignments[name] = 'right'

            for group in ('left', 'middle', 'right'):
                pat = rf'\b{group}\b.{{0,2000}}'
                match = re.search(pat, response, re.DOTALL | re.IGNORECASE)
                if match:
                    group_text = match.group(0)[:500]
                    for name, g in group_assignments.items():
                        if g == group:
                            result[name] = group_text

            if result:
                for name in agent_names:
                    if name not in result:
                        result[name] = response
                return result

        # Fallback
        return {name: response for name in agent_names}

    def _llm_call(self, prompt: str, max_tokens: int = 400) -> str:
        model_lower = self._model.lower()
        params = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if "o1" in model_lower or "o3" in model_lower:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        resp = self._client.chat.completions.create(**params)
        return resp.choices[0].message.content.strip()
