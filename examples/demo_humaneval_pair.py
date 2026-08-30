#!/usr/bin/env python3
"""Watch LangMARL's credit assignment work, on two agents solving HumanEval.

A Coder writes an implementation; a Tester writes unit tests for it, runs them,
and emits the final implementation -- repairing the Coder's draft when its own
tests expose a bug. The team is scored by HumanEval's official tests, so both
roles move the number: a Coder that drafts badly and a Tester that breaks
working code both cost the team.

This script exists to make the *mechanism* legible. For every trajectory it
prints what the centralized critic attributed to each agent, how those credits
are reconciled into one gradient, and how that gradient rewrites each policy.
Both policies start deliberately threadbare ("You write Python code.") so the
rewriting is visible.

It runs three HumanEval tasks that gpt-4o-mini fails under the starting policy,
so there is somewhere to go and so a whole iteration's credit fits on screen.

It is NOT a benchmark. Three tasks put every problem worth 33 points, so the
accuracy it reports is one task flipping. Measured runs -- on these three and on
a sample of ten -- came out between -30 and +33 points with no trend, and the
optimizer reliably grows both policies into verbose engineering prose. Read the
curve as a sanity check that the loop closes, not as evidence of improvement;
read the credit and policy text for what the demo is actually for.

Usage:
    echo 'OPENAI_API_KEY=sk-...' > .env      # git-ignored, loaded on import
    python examples/demo_humaneval_pair.py
    python examples/demo_humaneval_pair.py --iterations 3 --model gpt-4o
    python examples/demo_humaneval_pair.py --task-ids HumanEval/32,HumanEval/132
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    # Run straight from a clone, without `pip install -e .` first.
    sys.path.insert(0, str(REPO))

import langmarl  # noqa: E402
from langmarl.core.critic import CentralizedCritic  # noqa: E402
from langmarl.core.optimizer import PolicyGradientOptimizer  # noqa: E402

HUMANEVAL = REPO / "env" / "lang_benchmark" / "coding" / "test_tasks.jsonl"

# ── terminal styling ────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


BOLD = lambda s: _c("1", s)          # noqa: E731
DIM = lambda s: _c("2", s)           # noqa: E731
RED = lambda s: _c("31", s)          # noqa: E731
GREEN = lambda s: _c("32", s)        # noqa: E731
YELLOW = lambda s: _c("33", s)       # noqa: E731
BLUE = lambda s: _c("34", s)         # noqa: E731
MAGENTA = lambda s: _c("35", s)      # noqa: E731
CYAN = lambda s: _c("36", s)         # noqa: E731

WIDTH = 78

# The library logs progress and draws tqdm bars; both would interleave with the
# narration below, so the demo silences them and prints its own account.
sys.stdout.reconfigure(line_buffering=True)


def _quiet_library() -> None:
    """Replace the trainer's progress bars with a no-op passthrough."""
    from langmarl.trainer import monte_carlo

    monte_carlo.tqdm = lambda iterable, **kw: iterable


AGENT_LABEL = {"agent_1": "CODER", "agent_2": "TESTER"}
AGENT_COLOR = {"agent_1": CYAN, "agent_2": MAGENTA}


def rule(title: str = "", char: str = "-") -> None:
    if not title:
        print(char * WIDTH)
        return
    pad = WIDTH - len(title) - 4
    print(f"{char * 2} {BOLD(title)} {char * max(pad, 0)}")


def wrap(text: str, indent: str = "    ", limit: int | None = None) -> str:
    text = " ".join((text or "").split())
    if limit and len(text) > limit:
        text = text[:limit].rstrip() + " ..."
    return textwrap.fill(text, width=WIDTH, initial_indent=indent, subsequent_indent=indent)


# ── code execution ──────────────────────────────────────────────────────────

_HARNESS_HINTS = ("unittest", "pytest", "def test_")


def strip_test_harness(code: str) -> str:
    """Remove a self-running test harness from an extracted block.

    A block that ends in ``unittest.main()`` calls ``sys.exit()`` when imported
    alongside the official test, so the official ``check()`` never runs and a
    correct implementation is scored as a failure. Drop the ``__main__`` guard
    and any bare harness invocation.
    """
    lines, out, skipping = code.splitlines(), [], False
    for line in lines:
        stripped = line.strip()
        if re.match(r"if\s+__name__\s*==", stripped):
            skipping = True
            continue
        if skipping:
            # The guard's body is indented; the first unindented line ends it.
            if line and not line[0].isspace():
                skipping = False
            else:
                continue
        if re.match(r"(unittest|pytest)\.main\s*\(", stripped):
            continue
        out.append(line)
    return "\n".join(out)


def extract_code(text: str, entry_point: str | None = None) -> str:
    """Pull the implementation out of a markdown-fenced LLM reply.

    A Tester's reply holds its unit tests as well as the final implementation,
    so pick the last block that defines the function under test and does not
    look like a test harness, then strip any harness left inside it.
    """
    blocks = re.findall(r"```(?:python)?[ \t]*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        return strip_test_harness(text)

    candidates = blocks
    if entry_point:
        defining = [b for b in blocks
                    if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", b)]
        if defining:
            candidates = defining

    # Among the candidates, prefer one that is an implementation rather than a
    # test file; fall back to the last one either way.
    clean = [b for b in candidates
             if not any(hint in b for hint in _HARNESS_HINTS)]
    return strip_test_harness((clean or candidates)[-1])


def run_humaneval(code: str, task: dict, timeout: float = 10.0) -> tuple[bool, str]:
    """Run HumanEval's official test for one task against ``code``.

    Executed in a subprocess so a hang or a crash cannot take the demo with it.
    """
    passed, detail = _run_python(
        f"{code}\n\n{task['test']}\n\ncheck({task['entry_point']})\n", timeout
    )
    return passed, ("all official tests passed" if passed else detail)


def _run_python(program: str, timeout: float) -> tuple[bool, str]:
    """Execute a program in a subprocess; True when it exits cleanly."""
    program += "\nprint('__LANGMARL_PASS__')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(program)
        path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        if "__LANGMARL_PASS__" in proc.stdout:
            return True, "passed"
        err = (proc.stderr or proc.stdout).strip().splitlines()
        return False, err[-1] if err else "failed with no output"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"could not run: {exc}"
    finally:
        os.unlink(path)


def run_custom_tests(code: str, tests: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Run the Tester's own asserts against the Coder's implementation.

    Returns (passed, detail). A test file that cannot even run counts as failed,
    with the error as the detail, because that is what the Coder needs to see.
    """
    if not tests.strip():
        return False, "the Tester produced no runnable tests"
    return _run_python(f"{code}\n\n{tests}\n", timeout)


# ── environment ─────────────────────────────────────────────────────────────

CODER_PROMPT = """Task:
{problem}

Write a complete Python implementation. Put it in one ```python block."""

TESTER_PROMPT = """Task:
{problem}

The Coder wrote this implementation:

{coder_output}

Write unit tests for it. Use plain `assert` statements that call the function
directly -- no unittest, no pytest, no test functions, no prints. Cover the
docstring's examples and the edge cases you think are risky.

Do NOT write an implementation. Your job is to test, not to fix: if you find a
bug the Coder will repair it. Reply with one ```python block of asserts."""

CODER_REVISE_PROMPT = """Task:
{problem}

Your implementation:

{coder_output}

The Tester wrote these tests for it:

```python
{tests}
```

Result of running them against your implementation:
{test_result}

Give the FINAL implementation in one ```python block. If the tests found a real
bug, fix it. If they passed, or if a failure is the test's mistake rather than
yours, keep your implementation as it is -- do not rewrite working code."""


@langmarl.register_env("humaneval_pair")
class CoderTesterEnv(langmarl.BaseEnvironment):
    """Coder drafts, Tester validates and delivers. Scored by HumanEval."""

    ENV_NAME = "language_task"      # reuse the shipped language critic prompts
    TEMPLATE_PREFIX = "language"

    def __init__(self, config):
        self.num_agents = 2
        self.agent_names = ["agent_1", "agent_2"]
        self.llm = langmarl.LLMClient(config.get_actor_llm())
        self.tasks = load_tasks(getattr(config, "task_ids", None))

    # A fixed task set every iteration, so the accuracy curve measures the
    # policies rather than the luck of the draw.
    def sample_tasks(self, num_samples: int, seed: int | None = None) -> list[dict]:
        return list(self.tasks)

    def reset(self, task: dict) -> dict:
        return {"task": task}

    def step(self, agent_id: str, action: str):
        raise NotImplementedError("use collect_trajectory()")

    def collect_trajectory(self, policies: dict, task: dict) -> langmarl.Trajectory:
        """Coder drafts, Tester tests and reports, Coder revises.

        The Tester never writes the answer. It reports, and the Coder decides
        what to do about the report -- which is what makes the two credits
        separable: the Tester is judged on whether its tests found the truth,
        the Coder on whether it drafted well and acted on the report correctly.
        """
        problem, entry = task["problem"], task["entry_point"]
        coder_policy = str(policies.get("agent_1", "You write Python code."))
        tester_policy = str(policies.get("agent_2", "You write unit tests."))

        # 1. Coder drafts.
        draft_out, tok1 = self.llm.chat_with_usage(
            coder_policy, CODER_PROMPT.format(problem=problem)
        )
        draft_code = extract_code(draft_out, entry)

        # 2. Tester writes tests -- and they are actually run.
        tester_out, tok2 = self.llm.chat_with_usage(
            tester_policy,
            TESTER_PROMPT.format(problem=problem, coder_output=draft_out),
        )
        tests = extract_code(tester_out, entry)
        tests_passed, tests_detail = run_custom_tests(draft_code, tests)
        report = (
            "All of the Tester's tests passed."
            if tests_passed
            else f"A test failed: {tests_detail}"
        )

        # 3. Coder revises in light of the report. Its answer is the team's.
        final_out, tok3 = self.llm.chat_with_usage(
            coder_policy,
            CODER_REVISE_PROMPT.format(
                problem=problem, coder_output=draft_out,
                tests=tests, test_result=report,
            ),
        )
        final_code = extract_code(final_out, entry)

        draft_ok, _ = run_humaneval(draft_code, task)
        final_ok, final_note = run_humaneval(final_code, task)

        # Whether the Tester told the truth is checkable: compare its verdict on
        # the draft against what the official tests say about the same draft.
        if draft_ok and tests_passed:
            tester_verdict = "cleared a correct draft (right call)"
        elif not draft_ok and not tests_passed:
            tester_verdict = "caught a real bug in the draft (right call)"
        elif draft_ok and not tests_passed:
            tester_verdict = "FALSE ALARM: failed a draft that was actually correct"
        else:
            tester_verdict = "MISSED IT: passed a draft that was actually broken"

        if draft_ok and final_ok:
            coder_verdict = "drafted correctly and kept it."
        elif not draft_ok and final_ok:
            coder_verdict = "drafted a bug and repaired it after the report."
        elif draft_ok and not final_ok:
            coder_verdict = "drafted correctly, then BROKE it while revising."
        else:
            coder_verdict = "drafted a bug and did not repair it."

        steps = [
            {"agent": "agent_1", "agent_id": "agent_1", "system_prompt": coder_policy,
             "input": problem, "output": draft_out, "action": draft_out,
             "tokens": tok1, "role": "Coder, first draft"},
            {"agent": "agent_2", "agent_id": "agent_2", "system_prompt": tester_policy,
             "input": draft_out, "output": tester_out, "action": tester_out,
             "tokens": tok2, "role": f"Tester, wrote tests -> {report}"},
            {"agent": "agent_1", "agent_id": "agent_1", "system_prompt": coder_policy,
             "input": report, "output": final_out, "action": final_out,
             "tokens": tok3, "role": "Coder, revision (the team's answer)"},
        ]
        return langmarl.Trajectory(
            task=task,
            steps=steps,
            reward=1.0 if final_ok else 0.0,
            metadata={
                "task_type": "coding",
                "final_answer": final_out,
                "draft_passed": draft_ok,
                "final_passed": final_ok,
                "tester_tests_passed": tests_passed,
                "tester_verdict": tester_verdict,
                "coder_verdict": coder_verdict,
                "evaluation_feedback": (
                    f"Coder {coder_verdict} Tester {tester_verdict}. "
                    f"Official tests on the final answer: {final_note}"
                ),
            },
        )

    # Tell the critic what these two roles actually are; without this it falls
    # back to generic "first / final responder" descriptions. The two paradigms
    # read different variables -- central_credit ignores role_context -- so the
    # scoring contract goes into the criteria, which both paths render.
    def format_trajectory(self, episode: dict, paradigm: str) -> str:
        """Render the episode as three turns by two agents.

        The library's default formatter labels each step by its position, which
        would call the Coder's revision "Agent 3". Here the Coder speaks twice,
        so each turn is labelled by who took it and what it was for.
        """
        meta = episode
        lines = ["=== EPISODE ===", f"Task: {episode.get('task', {}).get('task_id', '?')}", ""]
        lines += ["[PROBLEM]", str(episode.get("task", {}).get("problem", ""))[:1200], ""]

        for step in episode.get("transitions", []):
            agent = step.get("agent", "?")
            label = AGENT_LABEL.get(agent, agent)
            lines += [f"=== {agent} ({label}) -- {step.get('role', '')} ===",
                      "Policy in force:", str(step.get("system_prompt", ""))[:500], "",
                      "Produced:", str(step.get("output", ""))[:2000], ""]

        lines += ["=== OUTCOME ===",
                  f"Coder's first draft vs the official tests: "
                  f"{'passed' if meta.get('draft_passed') else 'FAILED'}",
                  f"Tester's own tests on that draft: "
                  f"{'passed' if meta.get('tester_tests_passed') else 'FAILED'}",
                  f"Tester's verdict was: {meta.get('tester_verdict', '')}",
                  f"Coder: {meta.get('coder_verdict', '')}",
                  f"Final answer vs the official tests: "
                  f"{'PASSED' if meta.get('final_passed') else 'FAILED'}",
                  f"Team score: {episode.get('reward', 0.0):.1f}", ""]
        lines.append("Attribute this outcome to agent_1 (Coder) and agent_2 (Tester).")
        return "\n".join(lines)

    def critic_prompt_vars(self, paradigm: str, agents: list[str]) -> dict:
        roles = (
            "This is a two-agent coding team:\n"
            "- agent_1 (Coder): writes a first implementation, then -- after the "
            "Tester reports -- writes the FINAL implementation. Only the Coder "
            "ever writes code.\n"
            "- agent_2 (Tester): writes unit tests for the draft. Those tests are "
            "really executed, and the result is handed back to the Coder. The "
            "Tester never writes an implementation; its influence is entirely in "
            "whether its report is accurate and useful.\n"
            "The team is scored by HumanEval's official hidden tests."
        )
        criteria = (
            "HOW SCORING WORKS -- judge against this, not against general "
            "software-engineering taste. The hidden tests call the function "
            "with the inputs its docstring describes and compare the RETURN "
            "VALUE. Raising an exception, printing, or validating an input and "
            "refusing it are FAILURES: an empty string or an unusual value is a "
            "case to handle and return an answer for, not to reject. Input "
            "validation, defensive scaffolding and extra documentation earn "
            "nothing and often lose points. Solving the stated problem "
            "correctly is the only thing that scores.\n\n"
            "**For agent_1 (Coder):**\n"
            "- Was the first draft correct?\n"
            "- Given the Tester's report, did it revise well -- fixing a real bug, "
            "and leaving a correct draft alone when the report was a false alarm?\n"
            "- If the final answer is wrong, was it wrong from the start, or did "
            "the revision break it?\n\n"
            "**For agent_2 (Tester):**\n"
            "- Did its tests run at all, and did they exercise what the task needs?\n"
            "- Did its verdict on the draft match reality? A false alarm misleads "
            "the Coder into breaking working code; a miss leaves a bug in.\n"
            "- Was the failure it reported specific enough to act on?\n"
            "- It cannot write code, so judge only the quality of its report.\n\n"
            "IMPORTANT: each agent's policy is a standing instruction reused on "
            "problems it has never seen. Write every credit as a TRANSFERABLE "
            "working habit -- 'check the empty-input case before returning', "
            "'do not rewrite code your tests did not flag'. Never mention this "
            "problem's subject matter, function name, or specific values; a "
            "credit that only makes sense for this task is useless."
        )
        # The two paradigms read different variables, so hand each only what its
        # template renders; anything else would be dropped on the floor.
        if paradigm == "central_credit":
            return {"agent_specific_criteria": f"{roles}\n\n{criteria}"}
        return {"role_context": roles}


#: The three tasks this demo runs. gpt-4o-mini failed all three on 3 of 3
#: attempts under the threadbare Coder policy, and each fails by misreading the
#: spec -- rounding .5 away from zero, digit sums of negative numbers, swapping
#: case before substituting vowels -- which is the kind of habit a policy could
#: plausibly fix. Easier tasks put the team near its ceiling from the first
#: iteration, and more of them make the per-trajectory credit too long to read.
HARD_TASKS = ["HumanEval/99", "HumanEval/145", "HumanEval/93"]


def load_tasks(task_ids: list[str] | None = None) -> list[dict]:
    """Load the demo's tasks by id, defaulting to :data:`HARD_TASKS`."""
    if not HUMANEVAL.exists():
        sys.exit(f"missing {HUMANEVAL.relative_to(REPO)} -- run scripts/prepare_data.py")
    with open(HUMANEVAL) as f:
        by_id = {r["task_id"]: r for r in map(json.loads, filter(str.strip, f))}

    wanted = task_ids or HARD_TASKS
    missing = [t for t in wanted if t not in by_id]
    if missing:
        sys.exit(f"unknown task ids: {', '.join(missing)}")
    return [by_id[t] for t in wanted]


# ── instrumented components ─────────────────────────────────────────────────

#: Filled by LoudCritic, drained once per iteration.
_OUTCOMES: list[dict] = []


class LoudCritic(CentralizedCritic):
    """A critic that shows its per-agent attribution for every trajectory."""

    def evaluate(self, trajectory, policies):
        result = super().evaluate(trajectory, policies)
        _OUTCOMES.append(dict(trajectory.metadata))

        task_id = trajectory.task.get("task_id", "?")
        meta = trajectory.metadata
        mark = GREEN("PASS") if meta["final_passed"] else RED("FAIL")
        draft = GREEN("pass") if meta["draft_passed"] else RED("fail")
        said = "pass" if meta["tester_tests_passed"] else "fail"
        honest = "FALSE ALARM" not in meta["tester_verdict"] and \
             "MISSED IT" not in meta["tester_verdict"]
        report = (GREEN if honest else RED)(f"tester said {said}")

        print()
        print(f"  {BOLD(task_id)}  draft={draft}  {report}  final={mark}")
        print(DIM(wrap(f"CODER {meta['coder_verdict']}", indent="    ")))
        print(DIM(wrap(f"TESTER {meta['tester_verdict']}", indent="    ")))
        for agent, credit in result.get("per_agent", {}).items():
            color = AGENT_COLOR.get(agent, DIM)
            print(f"    {color('credit -> ' + AGENT_LABEL.get(agent, agent))}")
            print(DIM(wrap(credit, indent="      ", limit=420)))
        return result


class LoudOptimizer(PolicyGradientOptimizer):
    """An optimizer that narrates LLM_agg and LLM_opt."""

    def aggregate_gradients(self, gradients):
        merged = super().aggregate_gradients(gradients)
        if len(gradients) > 1:
            print(f"    {YELLOW('LLM_agg')} {len(gradients)} gradients -> 1")
            print(DIM(wrap(merged, indent="      ", limit=400)))
        return merged

    def synthesize_policy(self, base_policy, gradient, agent_name="agent"):
        # A policy is reused on unseen problems, so the rewrite must not absorb
        # the specifics of the ones it just saw. Without this the Coder ends up
        # with "you count uppercase vowels at even indices" as its policy.
        gradient = (
            f"{gradient}\n\n"
            "Constraint: the rewritten policy is a standing instruction applied to "
            "problems it has never seen. State general working habits only. Do not "
            "mention any function name, subject matter, or example value from the "
            "problems above."
        )
        new_policy = super().synthesize_policy(base_policy, gradient, agent_name)
        old = str(base_policy)
        print(f"    {YELLOW('LLM_opt')} {AGENT_LABEL.get(agent_name, agent_name)}: "
              f"{len(old)} -> {len(new_policy)} chars")
        return new_policy


# ── main ────────────────────────────────────────────────────────────────────

INITIAL_POLICIES = {
    "agent_1": "You write Python code.",
    "agent_2": "You write unit tests.",
}


def resolve_model(name: str) -> langmarl.LLMConfig:
    """Accept a preset name, a config JSON, or any raw model string.

    A raw string is passed through as-is so a provider-prefixed id such as
    ``openai/gpt-4o-mini`` reaches whatever OPENAI_BASE_URL points at.
    """
    try:
        return langmarl.LLMConfig.from_preset(name)
    except ValueError:
        return langmarl.LLMConfig(
            name=name,
            model_string=name,
            api_key_env_var="OPENAI_API_KEY",
            max_tokens=4096,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--task-ids", default=None,
                        help=f"comma-separated HumanEval ids; default {', '.join(HARD_TASKS)}")
    parser.add_argument("--workers", type=int, default=3, help="parallel trajectories")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "No API key. Put OPENAI_API_KEY=sk-... in a .env file at the project\n"
            "root (it is git-ignored), or export it. This demo makes real LLM calls."
        )

    _quiet_library()

    llm = resolve_model(args.model)
    config = langmarl.BaseConfig(
        exp_name="humaneval_pair",
        paradigm="central_credit",
        llm=llm,
        num_agents=2,
        num_iterations=args.iterations,
        trajectories_per_iteration=3,
        max_workers=args.workers,
        experiment_dir="./experiments",
        log_level="WARNING",   # the run log file still records everything
    )
    task_ids = ([t.strip() for t in args.task_ids.split(",") if t.strip()]
                if args.task_ids else HARD_TASKS)
    config.trajectories_per_iteration = len(task_ids)
    config.task_ids = task_ids
    config.optimizer_workers = 1

    env = langmarl.make_env("humaneval_pair", config)

    rule("LangMARL x HumanEval: Coder + Tester", "=")
    print(f"  model            {args.model}")
    print("  paradigm         central_credit (per-agent causal attribution)")
    print(f"  tasks/iteration  {len(env.tasks)}  (the same set every iteration)")
    print(f"  iterations       {args.iterations}")
    print()
    print(f"  {CYAN('CODER')}   starts as: {INITIAL_POLICIES['agent_1']!r}")
    print(f"  {MAGENTA('TESTER')}  starts as: {INITIAL_POLICIES['agent_2']!r}")
    print(f"  tasks: {', '.join(t['task_id'] for t in env.tasks)}")
    print()

    trainer = langmarl.MonteCarloTrainer(
        config=config,
        env=env,
        critic=LoudCritic(config, env=env),
        optimizer=LoudOptimizer(config.get_optimizer_llm()),
    )
    # Start from the threadbare prompts rather than the library's defaults.
    trainer.checkpoint.default_policy_fn = lambda idx, n: INITIAL_POLICIES[f"agent_{idx + 1}"]

    history: list[dict] = []
    for i in range(args.iterations):
        print()
        rule(f"ITERATION {i}", "=")

        before = trainer.checkpoint.get_policies()
        print(f"\n  {BOLD('Policies in play')}")
        for agent in ("agent_1", "agent_2"):
            color = AGENT_COLOR[agent]
            print(f"    {color(AGENT_LABEL[agent])}")
            print(DIM(wrap(str(before[agent]), indent="      ", limit=300)))

        print(f"\n  {BOLD('Rollouts and credit assignment')}")
        stats = trainer.train_one_iteration(i)

        after = trainer.checkpoint.get_policies()
        passed = int(round(stats["avg_reward"] * stats["num_episodes"]))

        # What each role actually contributed this iteration.
        outcomes, _OUTCOMES[:] = list(_OUTCOMES), []
        drafts_ok = sum(1 for o in outcomes if o["draft_passed"])
        repaired = sum(1 for o in outcomes if not o["draft_passed"] and o["final_passed"])
        broken = sum(1 for o in outcomes if o["draft_passed"] and not o["final_passed"])
        honest = sum(1 for o in outcomes
                     if o["draft_passed"] == o["tester_tests_passed"])
        history.append({
            "iteration": i,
            "accuracy": stats["avg_reward"],
            "passed": passed,
            "total": stats["num_episodes"],
            "drafts_ok": drafts_ok,
            "repaired": repaired,
            "broken": broken,
            "honest": honest,
        })

        cost = DIM(f"rollout cost ${stats.get('cost_usd', 0.0):.4f} cumulative")
        print(f"\n  {BOLD('Result')}  "
              f"{passed}/{stats['num_episodes']} passed  "
              f"({stats['avg_reward'] * 100:.1f}%)   {cost}")
        n = stats["num_episodes"]
        print(f"    {CYAN('CODER')}  drafts correct: {drafts_ok}/{n}   "
              f"repaired after the report: {GREEN(str(repaired))}   "
              f"broke a correct draft: {(RED if broken else DIM)(str(broken))}")
        print(f"    {MAGENTA('TESTER')} verdicts that matched reality: "
              f"{honest}/{n}  {DIM('(false alarms and misses are the rest)')}")

        print(f"\n  {BOLD('Evolved policies')}")
        for agent in ("agent_1", "agent_2"):
            color = AGENT_COLOR[agent]
            grew = len(str(after[agent])) - len(str(before[agent]))
            print(f"    {color(AGENT_LABEL[agent])} {DIM(f'({grew:+d} chars)')}")
            print(DIM(wrap(str(after[agent]), indent="      ", limit=460)))

    # ── curve ───────────────────────────────────────────────────────────────
    print()
    rule("ACCURACY CURVE", "=")
    print()
    best = max(h["accuracy"] for h in history) or 1.0
    for h in history:
        bar_len = int(round(h["accuracy"] / best * 40)) if best else 0
        bar = "#" * bar_len
        colorize = GREEN if h["accuracy"] >= best else BLUE
        print(f"  iter {h['iteration']}  {h['accuracy'] * 100:5.1f}%  "
              f"{h['passed']}/{h['total']}  {colorize(bar)}")

    print()
    print(f"  {BOLD('Where the score came from')}")
    print(DIM(f"    {'iter':<6}{'drafts ok':<12}{'repaired':<11}{'broke':<8}"
              f"{'tester accurate':<16}"))
    for h in history:
        drafts = f"{h['drafts_ok']}/{h['total']}"
        accurate = f"{h['honest']}/{h['total']}"
        print(f"    {h['iteration']:<6}{drafts:<12}"
              f"{h['repaired']:<11}{h['broken']:<8}{accurate:<16}")

    first, last = history[0]["accuracy"], history[-1]["accuracy"]
    delta = (last - first) * 100
    arrow = GREEN(f"+{delta:.1f} pts") if delta > 0 else (
        RED(f"{delta:.1f} pts") if delta < 0 else DIM("no change"))
    print(f"\n  iteration 0 -> {history[-1]['iteration']}: {arrow}")

    per_task = 100.0 / history[0]["total"]
    print(DIM(wrap(
        f"One task is worth {per_task:.0f} points here, so a run this small is "
        f"noise-dominated -- treat the curve as a sanity check that the loop "
        f"closes, not as a measurement. What this demo is for is the credit and "
        f"policy text above.", indent="  ")))

    print()
    rule("FINAL POLICIES", "=")
    final = trainer.checkpoint.get_policies()
    for agent in ("agent_1", "agent_2"):
        print()
        print(f"  {AGENT_COLOR[agent](AGENT_LABEL[agent])}  "
              f"{DIM(f'(started at {len(INITIAL_POLICIES[agent])} chars, now {len(str(final[agent]))})')}")
        print(textwrap.fill(" ".join(str(final[agent]).split()), width=WIDTH,
                            initial_indent="    ", subsequent_indent="    "))

    run_dir = Path(config.experiment_dir) / "runs" / trainer.run_id
    print()
    print(DIM(f"  Full trajectories, credits and gradients: {run_dir}"))


if __name__ == "__main__":
    main()
