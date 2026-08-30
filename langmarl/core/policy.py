"""Language policy representation.

A language policy is the system prompt an agent acts under, and it has two
parts that behave very differently under training:

* a **base** — who this agent is and where it sits in the collaboration. It is
  set once and never rewritten, so the protocol the trajectory relies on cannot
  be optimized away.
* a **feedback** section — the aggregated language gradient from the most recent
  iteration. It is *replaced* on every update, not appended to.

Keeping the two apart is what makes replacement structural. When a policy is a
single flat string the optimizer cannot tell last iteration's gradient from the
base, so every update appends and the prompt grows without bound, carrying stale
gradients along with the current one.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Separates the base policy from the current gradient inside a rendered prompt.
FEEDBACK_MARKER = "[CASE-SPECIFIC FEEDBACK]"


@dataclass
class AgentPolicy:
    """One agent's language policy: a fixed base plus a replaceable gradient."""

    base: str
    feedback: str = ""

    @property
    def combined(self) -> str:
        """The full system prompt to send to the LLM."""
        if not self.feedback.strip():
            return self.base.rstrip()
        return f"{self.base.rstrip()}\n\n{FEEDBACK_MARKER}\n{self.feedback.strip()}"

    def with_gradient(self, gradient: str) -> AgentPolicy:
        """Return a policy with the feedback section replaced by ``gradient``."""
        return AgentPolicy(base=self.base, feedback=gradient)

    @classmethod
    def from_legacy(cls, text: str) -> AgentPolicy:
        """Parse a policy that was stored as one flat string.

        Checkpoints written before policies were structured hold the base and
        every appended gradient concatenated together. Splitting on the marker
        recovers the base and keeps only the newest gradient, which collapses an
        accumulated prompt back to the size it should have had.
        """
        if isinstance(text, cls):
            return text
        if FEEDBACK_MARKER not in text:
            return cls(base=text.rstrip())

        base, _, rest = text.partition(FEEDBACK_MARKER)
        # Older checkpoints stacked several gradients; the last one is current.
        newest = rest.rsplit(FEEDBACK_MARKER, 1)[-1]
        return cls(base=base.rstrip(), feedback=newest.strip())

    def __str__(self) -> str:
        return self.combined


def default_agent_prompt(agent_idx: int, num_agents: int) -> str:
    """The base prompt for the agent at ``agent_idx`` in a sequential chain.

    Describes only the collaboration protocol -- position, what this agent can
    see, and who sees its output. Strategy is left to the learned feedback.
    """
    position = agent_idx + 1

    if num_agents == 1:
        return (
            "You are the sole agent in a collaborative task system.\n"
            "- You receive the task and provide the final answer\n"
            "Please provide your response to the task."
        )

    lines = [f"You are participating in a collaborative task with {num_agents} agents."]
    is_first = agent_idx == 0
    is_last = agent_idx == num_agents - 1

    if is_first:
        later = ", ".join(f"Agent {j + 1}" for j in range(1, num_agents))
        lines.append(f"- You are Agent {position}, speaking first")
        lines.append(f"- After you respond, {later} will see the task and your response")
        lines.append(f"- Agent {num_agents}'s output will be the final answer")
    elif is_last:
        prev = ", ".join(f"Agent {j + 1}" for j in range(agent_idx))
        lines.append(f"- You are Agent {position}, speaking last")
        lines.append(f"- You can see the original task and the responses from {prev}")
        lines.append("- YOUR output is the FINAL answer that will be evaluated")
    else:
        prev = ", ".join(f"Agent {j + 1}" for j in range(agent_idx))
        later = ", ".join(f"Agent {j + 1}" for j in range(position, num_agents))
        lines.append(f"- You are Agent {position}")
        lines.append(f"- You can see the original task and the responses from {prev}")
        lines.append(f"- {later} will see your response")
        lines.append(f"- Agent {num_agents}'s output will be the final answer")

    lines.append("Please provide your response to the task.")
    return "\n".join(lines)


def default_agent_policy(agent_idx: int, num_agents: int) -> AgentPolicy:
    """A starting policy for one agent: the protocol base, no gradient yet."""
    return AgentPolicy(base=default_agent_prompt(agent_idx, num_agents))
