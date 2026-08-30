"""How one iteration's gradients become the next policy.

Each trajectory produces its own gradient and those are kept as a list. After
the iteration the list goes through two LLM steps, matching the paper's
``pi <- LLM_opt(pi, LLM_agg({d_pi}))``:

1. ``aggregate_gradients`` reconciles the K trajectory gradients into one,
   resolving contradictions and dropping single-episode noise.
2. ``synthesize_policy`` rewrites the *previous* policy around that gradient.
"""

import json

import pytest

import langmarl
from langmarl.core.optimizer import PolicyGradientOptimizer
from langmarl.core.policy import FEEDBACK_MARKER


@pytest.fixture
def optimizer(llm):
    return PolicyGradientOptimizer(llm)


@pytest.fixture
def recorder(optimizer):
    """Capture every prompt the optimizer sends, and count the calls."""
    calls = []

    def fake_llm(prompt, max_tokens=None):
        calls.append(prompt)
        return f"RESPONSE-{len(calls)}"

    optimizer._llm_call = fake_llm
    optimizer.calls = calls
    return optimizer


# --- LLM_agg ---------------------------------------------------------------

def test_aggregation_sees_every_trajectory_gradient(recorder):
    recorder.aggregate_gradients(["grad-A", "grad-B", "grad-C"])

    assert len(recorder.calls) == 1, "aggregation is a single call"
    prompt = recorder.calls[0]
    for gradient in ("grad-A", "grad-B", "grad-C"):
        assert gradient in prompt, f"{gradient} never reached the aggregator"


def test_aggregation_numbers_the_gradients_so_support_can_be_counted(recorder):
    recorder.aggregate_gradients(["grad-A", "grad-B"])
    prompt = recorder.calls[0]
    assert "Gradient 1" in prompt and "Gradient 2" in prompt


def test_a_single_gradient_needs_no_reconciliation(recorder):
    assert recorder.aggregate_gradients(["only-one"]) == "only-one"
    assert recorder.calls == [], "no LLM call for a single gradient"


def test_no_gradients_aggregates_to_nothing(recorder):
    assert recorder.aggregate_gradients([]) == ""
    assert recorder.calls == []


def test_failed_aggregation_degrades_to_concatenation(optimizer):
    optimizer._llm_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API down"))
    result = optimizer.aggregate_gradients(["grad-A", "grad-B"])
    assert "grad-A" in result and "grad-B" in result


# --- LLM_opt ---------------------------------------------------------------

def test_synthesis_rewrites_the_previous_policy_around_the_gradient(recorder):
    result = recorder.synthesize_policy("PREVIOUS POLICY", "AGGREGATED GRADIENT")

    assert result == "RESPONSE-1"
    assert len(recorder.calls) == 1
    assert "PREVIOUS POLICY" in recorder.calls[0]
    assert "AGGREGATED GRADIENT" in recorder.calls[0]


def test_synthesis_returns_a_self_contained_policy(recorder):
    """The rewrite replaces the policy; it does not staple notes onto it."""
    assert FEEDBACK_MARKER not in recorder.synthesize_policy("PREVIOUS", "grad")


def test_no_gradient_leaves_the_policy_alone(recorder):
    assert recorder.synthesize_policy("PREVIOUS", "") == "PREVIOUS"
    assert recorder.calls == []


def test_a_failed_call_degrades_to_concatenation(optimizer):
    optimizer._llm_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API down"))
    result = optimizer.synthesize_policy("PREVIOUS", "grad-A")

    assert result.startswith("PREVIOUS")
    assert "grad-A" in result
    assert result.count(FEEDBACK_MARKER) == 1


def test_the_fallback_path_still_does_not_accumulate(optimizer):
    """Repeated failures must not grow the prompt without bound."""
    optimizer._llm_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError())

    policy = "BASE"
    for i in range(5):
        policy = optimizer.synthesize_policy(policy, f"grad-{i}")

    assert policy.count(FEEDBACK_MARKER) == 1
    assert policy.startswith("BASE")


def test_synthesis_accepts_an_agent_policy(recorder):
    policy = langmarl.AgentPolicy(base="BASE", feedback="old gradient")
    assert recorder.synthesize_policy(policy, "grad") == "RESPONSE-1"
    assert "BASE" in recorder.calls[0]


# --- the two steps together ------------------------------------------------

def test_a_list_is_aggregated_exactly_once(recorder):
    """Convenience path: passing the raw list must not aggregate twice."""
    recorder.synthesize_policy("PREVIOUS", ["grad-A", "grad-B"])

    assert len(recorder.calls) == 2, "one aggregation call, then one rewrite"
    assert "grad-A" in recorder.calls[0] and "grad-B" in recorder.calls[0]
    assert "RESPONSE-1" in recorder.calls[1], "the rewrite consumes the aggregate"


def test_the_iteration_costs_two_calls_per_agent(recorder):
    """What the trainer does: aggregate once, then rewrite once."""
    grads = ["grad-A", "grad-B", "grad-C"]
    aggregated = recorder.aggregate_gradients(grads)
    recorder.synthesize_policy("PREVIOUS", aggregated)

    assert len(recorder.calls) == 2, "no redundant aggregation"


def test_per_trajectory_gradients_are_stored_individually(tmp_path, llm):
    """The list is the record of the iteration, not just the merged text."""
    from langmarl.store.local import LocalStore

    store = LocalStore(str(tmp_path))
    run_id = store.create_run("grads", langmarl.BaseConfig(llm=llm))
    grads = ["grad-A", "grad-B", "grad-C"]
    store.save_gradients(run_id, 0, "agent_1", grads, "AGGREGATED")

    saved = json.loads(
        (tmp_path / "runs" / run_id / "gradients" / "iter_0" / "agent_1_gradients.json").read_text()
    )
    assert saved == grads


def test_a_custom_optimizer_without_synthesis_still_works(llm):
    """BaseOptimizer's default keeps third-party optimizers usable."""

    class Minimal(langmarl.BaseOptimizer):
        def generate_gradient(self, policy, evaluation, context):
            return "g"

        def apply_gradient(self, policy, gradient):
            return f"{policy}||{gradient}"

        def aggregate_gradients(self, gradients):
            return "+".join(gradients)

    assert Minimal().synthesize_policy("P", ["a", "b"]) == "P||a+b"
    assert Minimal().synthesize_policy("P", "g") == "P||g"
    assert Minimal().synthesize_policy("P", []) == "P"
