"""The environment registry and its error messages."""

import importlib.util

import pytest

import langmarl


def test_the_language_env_always_registers():
    """It needs nothing beyond the base install."""
    assert "language" in langmarl.list_envs()


@pytest.mark.parametrize(
    "name,module",
    [("pistonball", "pettingzoo"), ("overcooked", "overcooked_ai_py")],
)
def test_optional_envs_register_only_with_their_extra(name, module):
    """list_envs() lists what can actually be constructed, nothing more."""
    installed = importlib.util.find_spec(module) is not None
    assert (name in langmarl.list_envs()) == installed


@pytest.mark.parametrize("name", ["pistonball", "overcooked"])
def test_a_missing_extra_is_explained_not_just_raised(name, llm):
    """A user who skipped the extra should be told which one to install."""
    if name in langmarl.list_envs():
        pytest.skip(f"{name} extra is installed")

    config = langmarl.BaseConfig(llm=llm)
    with pytest.raises(ValueError, match=f"pip install 'langmarl\\[{name}\\]'"):
        langmarl.make_env(name, config)


def test_unknown_env_lists_alternatives(llm):
    config = langmarl.BaseConfig(llm=llm)
    with pytest.raises(ValueError, match="Unknown environment"):
        langmarl.make_env("does-not-exist", config)


def test_custom_env_can_be_registered(llm):
    @langmarl.register_env("dummy-registry-test")
    class _Dummy(langmarl.BaseEnvironment):
        def __init__(self, config):
            self.config = config

        def reset(self, task):
            return {}

        def step(self, agent_id, action):
            return {}, 0.0, True, {}

        def collect_trajectory(self, policies, task):
            return langmarl.Trajectory(task=task, steps=[], reward=0.0)

    env = langmarl.make_env("dummy-registry-test", langmarl.BaseConfig(llm=llm))
    assert isinstance(env, _Dummy)
