"""Every shipped evaluation template must render with no LLM involved.

This is the regression guard for the environment migration: a template that
asks for a variable its environment does not supply used to fail only after a
full training run had already spent tokens.
"""

import importlib.util
from pathlib import Path

import pytest

import langmarl
from langmarl.core.critic import CentralizedCritic

ENVS_DIR = Path(langmarl.__file__).parent / "envs"
PARADIGMS = ["central_global", "central_credit"]

TRAJECTORY = "=== EPISODE TRAJECTORY ===\nnothing happened\n=== END ==="


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _prompt_dirs():
    for path in sorted(ENVS_DIR.glob("*/prompts/evaluation")):
        yield path.parent.parent.name, path


def test_every_env_ships_both_paradigm_templates():
    found = dict(_prompt_dirs())
    assert set(found) >= {"language", "pistonball", "overcooked"}
    for name, path in found.items():
        for paradigm in PARADIGMS:
            candidates = list((path / "templates").glob(f"*{paradigm}.json"))
            assert candidates, f"{name} has no {paradigm} template"


@pytest.mark.parametrize("paradigm", PARADIGMS)
@pytest.mark.skipif(not _has("pettingzoo"), reason="pistonball extra not installed")
def test_pistonball_prompts_render(llm, paradigm):
    from langmarl.envs.pistonball import PistonballEnv

    config = langmarl.PistonballConfig(
        paradigm=paradigm, num_pistons=6, num_agents=6, llm=llm
    )
    env = PistonballEnv(config)
    critic = CentralizedCritic(config, env=env)

    prompt = critic._render(TRAJECTORY, env.agent_names)
    assert TRAJECTORY in prompt
    assert "{" not in prompt.replace("{{", "").replace("}}", "") or "json" in prompt.lower()
    if paradigm == "central_credit":
        assert "PISTON_0 EVALUATION" in prompt


@pytest.mark.parametrize("paradigm", PARADIGMS)
def test_language_prompts_render(llm, paradigm):
    """The language templates render through the no-env default path."""
    config = langmarl.LanguageTaskConfig(paradigm=paradigm, num_agents=2, llm=llm)
    critic = CentralizedCritic(config)

    prompt = critic._render(TRAJECTORY, ["agent_1", "agent_2"])
    assert TRAJECTORY in prompt
    assert "agent_1" in prompt


@pytest.mark.parametrize("paradigm", PARADIGMS)
@pytest.mark.skipif(not _has("overcooked_ai_py"), reason="overcooked extra not installed")
def test_overcooked_prompts_render(llm, paradigm):
    from langmarl.envs.overcooked import OvercookedEnv

    config = langmarl.OvercookedConfig(
        paradigm=paradigm, layout="cramped_room", num_agents=2, llm=llm
    )
    env = OvercookedEnv(config)
    critic = CentralizedCritic(config, env=env)

    prompt = critic._render(TRAJECTORY, env.agent_names)
    assert TRAJECTORY in prompt


def test_missing_template_variable_is_reported_clearly(llm, tmp_path):
    """A template needing an unsupplied variable names it, instead of KeyError: 'x'."""
    templates = tmp_path / "templates"
    templates.mkdir(parents=True)
    (templates / "central_global.json").write_text(
        '{"template": "{trajectory} and {a_variable_nobody_supplies}"}'
    )

    config = langmarl.BaseConfig(paradigm="central_global", num_agents=2, llm=llm)
    critic = CentralizedCritic(config, prompts_dir=tmp_path)

    with pytest.raises(KeyError, match="critic_prompt_vars"):
        critic._render(TRAJECTORY, ["agent_1", "agent_2"])


def test_credit_assignment_shows_why_the_episode_scored_what_it_did():
    """The critic must see the failure reason, not just the score.

    Without it, credit assignment can only observe *that* the team failed and
    invents plausible-sounding causes, which compound into harmful policies.
    """
    from langmarl.core.trajectory import TrajectoryFormatter

    episode = {
        "task": {"question": "q"},
        "transitions": [{"agent": "agent_1", "output": "a"}],
        "reward": 0.0,
        "evaluation_feedback": "AssertionError on the empty-input case",
    }
    for render in (TrajectoryFormatter.format_trajectory,
                   TrajectoryFormatter.format_for_credit_assignment):
        assert "AssertionError on the empty-input case" in render(episode), render.__name__
