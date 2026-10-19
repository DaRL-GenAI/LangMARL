#!/usr/bin/env python3
"""Watch LangMARL's credit assignment work, on two agents solving HumanEval.

A Coder writes an implementation; a Tester writes unit tests for it, which are
really executed, and reports what failed; the Coder then writes the final
implementation in light of that report. The team is scored by HumanEval's
official tests, so both roles move the number: a Coder that drafts badly and a
Tester whose false alarm talks the Coder out of working code both cost the team.

This script exists to make the *mechanism* legible. For every trajectory it
prints what the centralized critic attributed to each agent, how those credits
are reconciled into one gradient, and how that gradient rewrites each policy.
Both policies start deliberately threadbare ("You write Python code.") so the
rewriting is visible.

HumanEval is split, with a fixed seed, into a train split and a held-out test
split (--n-test, default 80). Each iteration trains on a fresh draw of --n-train
tasks (default 8), and the test split is scored before training and after every
iteration with no critic and no update, so the curve measures transfer to
problems the policies never saw. Actors decode at temperature 0 to keep that
curve from being dominated by sampling noise.

Left unchecked, the optimizer grows both policies into verbose engineering
prose, so each rewrite is held to --max-policy-words (default 50).

Usage:
    echo 'OPENAI_API_KEY=sk-...' > .env      # git-ignored, loaded on import
    python examples/demo_humaneval_pair.py
    python examples/demo_humaneval_pair.py --iterations 3 --model gpt-4o
    python examples/demo_humaneval_pair.py --n-train 12 --n-test 40 --seed 1
    python examples/demo_humaneval_pair.py --max-policy-words 30
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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


def strip_self_tests(code: str) -> str:
    """Drop test code the Coder appended to its implementation.

    Top-level asserts, ``test_*`` functions and bare calls run on import, so a
    wrong expected value in the Coder's own test fails a correct implementation
    before the official tests get to run. Testing is the Tester's job; only the
    implementation is scored.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    tree.body = [
        node for node in tree.body
        if not isinstance(node, (ast.Assert, ast.Expr))
        and not (isinstance(node, ast.FunctionDef) and node.name.startswith("test"))
    ]
    return ast.unparse(tree)


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

Write a complete Python implementation. Put it in one ```python block.
Output only the implementation: no tests, no asserts, no example calls. Testing
is the Tester's job."""

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
yours, keep your implementation as it is -- do not rewrite working code.
Output only the implementation: no tests, no asserts, no example calls."""


@langmarl.register_env("humaneval_pair")
class CoderTesterEnv(langmarl.BaseEnvironment):
    """Coder drafts, Tester validates and delivers. Scored by HumanEval."""

    ENV_NAME = "language_task"      # reuse the shipped language critic prompts
    TEMPLATE_PREFIX = "language"

    def __init__(self, config):
        self.num_agents = 2
        self.agent_names = ["agent_1", "agent_2"]
        self.llm = langmarl.LLMClient(config.get_actor_llm())
        self.train_tasks, self.test_tasks = split_tasks(config.n_test, config.split_seed)
        self._rng = random.Random(config.split_seed)

    # Each iteration trains on a fresh draw from the train split, so a policy
    # cannot score by memorising a handful of problems. The test split is never
    # sampled here; it only scores the policies (see evaluate_on).
    def sample_tasks(self, num_samples: int, seed: int | None = None) -> list[dict]:
        return self._rng.sample(self.train_tasks, min(num_samples, len(self.train_tasks)))

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
        draft_code = strip_self_tests(extract_code(draft_out, entry))

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
        final_code = strip_self_tests(extract_code(final_out, entry))

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
            "correctly is the only thing that scores. Only the Coder's "
            "implementation is scored: the Coder must not write tests, which "
            "are the Tester's job.\n\n"
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
            "problems it has never seen, and it is scored on held-out problems "
            "it never trains on. So summarize, as far as possible, the HIGH-LEVEL "
            "STRATEGY behind what went right or wrong: whether the agent "
            "followed the specification exactly, how it decided whether to "
            "revise, whether it stayed within its role. Ask what general habit "
            "would have prevented this failure on any problem, and state that "
            "habit -- 'trace every docstring example through the code before "
            "answering', 'do not add checks or cases the spec does not ask "
            "for', 'do not rewrite code unless a test shows a real bug'. "
            "Never mention this "
            "problem's subject matter, function name, or specific values; a "
            "credit that only makes sense for this task is useless."
        )
        # The two paradigms read different variables, so hand each only what its
        # template renders; anything else would be dropped on the floor.
        if paradigm == "central_credit":
            return {"agent_specific_criteria": f"{roles}\n\n{criteria}"}
        return {"role_context": roles}


def split_tasks(n_test: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Split HumanEval's 164 tasks into disjoint train and test sets.

    The split is a seeded shuffle, so every run with the same ``--seed`` holds
    out the same test problems.
    """
    if not HUMANEVAL.exists():
        sys.exit(f"missing {HUMANEVAL.relative_to(REPO)} -- run scripts/prepare_data.py")
    with open(HUMANEVAL) as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    if not 0 < n_test < len(tasks):
        sys.exit(f"--n-test must be between 1 and {len(tasks) - 1}")
    random.Random(seed).shuffle(tasks)
    return tasks[n_test:], tasks[:n_test]


def evaluate_on(env, policies: dict, tasks: list[dict], workers: int) -> float:
    """Pass rate of ``policies`` on ``tasks``: rollouts only, no critic, no update.

    A rollout that errors out (an API failure, say) counts as a fail.
    """
    def run(task):
        try:
            return env.collect_trajectory(policies, task).reward
        except Exception:
            return 0.0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rewards = list(pool.map(run, tasks))
    return sum(rewards) / len(rewards)


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
    """An optimizer that narrates LLM_agg and LLM_opt, and caps policy length."""

    #: Word budget for a rewritten policy; set from --max-policy-words.
    max_words: int = 50

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
            f"problems above. Keep it to at most {self.max_words} words: keep only the "
            "habits that matter most and drop the rest."
        )
        new_policy = self._enforce_length(
            super().synthesize_policy(base_policy, gradient, agent_name)
        )
        old = str(base_policy)
        print(f"    {YELLOW('LLM_opt')} {AGENT_LABEL.get(agent_name, agent_name)}: "
              f"{len(old)} -> {len(new_policy)} chars")
        return new_policy

    def _enforce_length(self, policy: str) -> str:
        """Hold a policy to ``max_words``: one compression call, then a hard cut.

        Left alone, LLM_opt only ever adds advice, and the policies grow into
        1000+ character essays within a few iterations.
        """
        if len(policy.split()) <= self.max_words:
            return policy
        try:
            policy = self._llm_call(
                f"Rewrite this instruction in at most {self.max_words} words. Keep "
                "its most important habits, drop the rest, and output only the "
                f"rewritten instruction.\n\n{policy}",
                max_tokens=4 * self.max_words,
            ).strip()
        except Exception:
            pass
        words = policy.split()
        return policy if len(words) <= self.max_words else " ".join(words[: self.max_words])


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
    parser.add_argument("--n-train", type=int, default=8,
                        help="training tasks drawn from the train split each iteration")
    parser.add_argument("--n-test", type=int, default=80,
                        help="held-out tasks scored before training and after every iteration")
    parser.add_argument("--seed", type=int, default=0, help="seed for the train/test split")
    parser.add_argument("--workers", type=int, default=16, help="parallel trajectories")
    parser.add_argument("--max-policy-words", type=int, default=50,
                        help="word budget for each rewritten policy (default: 50)")
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
    # Greedy decoding for the actors, so a change in test accuracy reflects the
    # policies more than the luck of a single sample.
    config.actor_llm = replace(llm, extra_params={**llm.extra_params, "temperature": 0})
    config.trajectories_per_iteration = args.n_train
    config.n_test = args.n_test
    config.split_seed = args.seed
    config.optimizer_workers = 1

    env = langmarl.make_env("humaneval_pair", config)

    rule("LangMARL x HumanEval: Coder + Tester", "=")
    print(f"  model            {args.model}")
    print("  paradigm         central_credit (per-agent causal attribution)")
    print(f"  train split      {len(env.train_tasks)} tasks, {args.n_train} drawn per iteration")
    print(f"  test split       {len(env.test_tasks)} held-out tasks, never trained on")
    print(f"  iterations       {args.iterations}")
    print(f"  policy budget    {args.max_policy_words} words")
    print()
    print(f"  {CYAN('CODER')}   starts as: {INITIAL_POLICIES['agent_1']!r}")
    print(f"  {MAGENTA('TESTER')}  starts as: {INITIAL_POLICIES['agent_2']!r}")
    print()

    trainer = langmarl.MonteCarloTrainer(
        config=config,
        env=env,
        critic=LoudCritic(config, env=env),
        optimizer=LoudOptimizer(config.get_optimizer_llm()),
    )
    trainer.optimizer.max_words = args.max_policy_words
    # Start from the threadbare prompts rather than the library's defaults.
    trainer.checkpoint.default_policy_fn = lambda idx, n: INITIAL_POLICIES[f"agent_{idx + 1}"]

    print(f"\n  {BOLD('Test accuracy before training')}  ", end="", flush=True)
    baseline = evaluate_on(env, trainer.checkpoint.get_policies(), env.test_tasks, args.workers)
    print(f"{baseline * 100:.1f}%  ({round(baseline * len(env.test_tasks))}/{len(env.test_tasks)})")

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
        test_acc = evaluate_on(env, after, env.test_tasks, args.workers)
        history.append({
            "iteration": i,
            "accuracy": stats["avg_reward"],
            "passed": passed,
            "total": stats["num_episodes"],
            "drafts_ok": drafts_ok,
            "repaired": repaired,
            "broken": broken,
            "honest": honest,
            "test": test_acc,
        })

        cost = DIM(f"rollout cost ${stats.get('cost_usd', 0.0):.4f} cumulative")
        print(f"\n  {BOLD('Train')}  "
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

        n_test = len(env.test_tasks)
        print(f"    {BOLD('Test')} with the updated policies: {test_acc * 100:.1f}%  "
              f"({round(test_acc * n_test)}/{n_test} held-out tasks)")

    # ── curve ───────────────────────────────────────────────────────────────
    n_test = len(env.test_tasks)
    print()
    rule("HELD-OUT TEST ACCURACY", "=")
    print()
    points = [("start", baseline)] + [(f"iter {h['iteration']}", h["test"]) for h in history]
    best = max(acc for _, acc in points) or 1.0
    for label, acc in points:
        bar = "#" * int(round(acc / best * 40))
        print(f"  {label:<7} {acc * 100:5.1f}%  {round(acc * n_test):>3}/{n_test}  "
              f"{(GREEN if acc >= best else BLUE)(bar)}")

    print()
    print(f"  {BOLD('Training batches (a fresh draw every iteration)')}")
    print(DIM(f"    {'iter':<6}{'train':<9}{'drafts ok':<12}{'repaired':<11}{'broke':<8}"
              f"{'tester accurate':<16}"))
    for h in history:
        train = f"{h['passed']}/{h['total']}"
        drafts = f"{h['drafts_ok']}/{h['total']}"
        accurate = f"{h['honest']}/{h['total']}"
        print(f"    {h['iteration']:<6}{train:<9}{drafts:<12}"
              f"{h['repaired']:<11}{h['broken']:<8}{accurate:<16}")

    delta = (history[-1]["test"] - baseline) * 100
    arrow = GREEN(f"+{delta:.1f} pts") if delta > 0 else (
        RED(f"{delta:.1f} pts") if delta < 0 else DIM("no change"))
    print(f"\n  held-out test, start -> iter {history[-1]['iteration']}: {arrow}")
    print(DIM(wrap(
        f"One test task is worth {100.0 / n_test:.1f} points, and every rollout "
        f"is a single sample, so differences of a task or two are noise.",
        indent="  ")))

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
