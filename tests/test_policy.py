"""A policy is a fixed base plus a gradient that is replaced, never appended."""

import pytest

from langmarl.core.optimizer import PolicyGradientOptimizer
from langmarl.core.policy import (
    FEEDBACK_MARKER,
    AgentPolicy,
    default_agent_policy,
    default_agent_prompt,
)


def test_a_fresh_policy_has_no_feedback_section():
    policy = default_agent_policy(0, 2)
    assert policy.feedback == ""
    assert FEEDBACK_MARKER not in policy.combined
    assert policy.combined == default_agent_prompt(0, 2)


def test_a_gradient_is_rendered_under_the_marker():
    policy = AgentPolicy(base="BASE").with_gradient("do better")
    assert policy.combined == f"BASE\n\n{FEEDBACK_MARKER}\ndo better"
    assert policy.base == "BASE"


def test_updating_replaces_the_gradient_rather_than_appending():
    policy = AgentPolicy(base="BASE")
    for i in range(5):
        policy = policy.with_gradient(f"gradient-{i}")

    assert policy.combined.count(FEEDBACK_MARKER) == 1
    assert "gradient-4" in policy.combined
    assert "gradient-3" not in policy.combined
    assert policy.base == "BASE"


def test_repeated_optimizer_updates_keep_the_prompt_bounded():
    """The trainer feeds each iteration's output back in as the next base."""
    policy = "BASE"
    for i in range(10):
        policy = PolicyGradientOptimizer.apply_gradient(policy, f"gradient-{i}")

    assert policy.count(FEEDBACK_MARKER) == 1
    assert policy == f"BASE\n\n{FEEDBACK_MARKER}\ngradient-9"


def test_legacy_flat_policy_round_trips():
    policy = AgentPolicy.from_legacy("just a plain prompt")
    assert policy.base == "just a plain prompt"
    assert policy.feedback == ""


def test_legacy_accumulated_policy_collapses_to_the_newest_gradient():
    """Checkpoints written before this fix stacked every gradient."""
    accumulated = (
        f"BASE\n\n{FEEDBACK_MARKER}\nold-1"
        f"\n\n{FEEDBACK_MARKER}\nold-2"
        f"\n\n{FEEDBACK_MARKER}\nnewest"
    )
    policy = AgentPolicy.from_legacy(accumulated)
    assert policy.base == "BASE"
    assert policy.feedback == "newest"
    assert policy.combined.count(FEEDBACK_MARKER) == 1


def test_from_legacy_passes_through_an_agent_policy():
    original = AgentPolicy(base="BASE", feedback="grad")
    assert AgentPolicy.from_legacy(original) is original


@pytest.mark.parametrize("num_agents", [1, 2, 4])
def test_default_prompts_describe_each_position(num_agents):
    prompts = [default_agent_prompt(i, num_agents) for i in range(num_agents)]
    assert len(set(prompts)) == num_agents, "each position needs its own protocol"
    if num_agents > 1:
        assert "speaking first" in prompts[0]
        assert "FINAL answer" in prompts[-1]


def test_str_renders_the_combined_prompt():
    policy = AgentPolicy(base="BASE", feedback="grad")
    assert str(policy) == policy.combined
