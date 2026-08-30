"""Config validation and JSON round-tripping."""

import json

import pytest

import langmarl


def test_paradigm_is_validated():
    with pytest.raises(ValueError, match="Invalid paradigm"):
        langmarl.BaseConfig(paradigm="not-a-paradigm")


def test_task_type_is_validated():
    with pytest.raises(ValueError, match="Invalid task_type"):
        langmarl.LanguageTaskConfig(task_type="not-a-task")


def test_json_round_trip(tmp_path, llm):
    original = langmarl.LanguageTaskConfig(
        exp_name="round-trip", task_type="math", num_iterations=3, llm=llm
    )
    path = tmp_path / "config.json"
    original.to_json(str(path))

    restored = langmarl.LanguageTaskConfig.from_json(str(path))
    assert restored.exp_name == "round-trip"
    assert restored.task_type == "math"
    assert restored.num_iterations == 3


def test_overrides_win_over_file(tmp_path, llm):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"exp_name": "from-file", "num_iterations": 1}))

    config = langmarl.LanguageTaskConfig.from_json(
        str(path), overrides={"num_iterations": 42}
    )
    assert config.num_iterations == 42


@pytest.mark.parametrize(
    "env_name,expected",
    [
        ("language", langmarl.LanguageTaskConfig),
        ("overcooked", langmarl.OvercookedConfig),
        ("pistonball", langmarl.PistonballConfig),
    ],
)
def test_load_config_picks_the_env_specific_class(tmp_path, env_name, expected):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"env": env_name, "exp_name": "x"}))
    assert isinstance(langmarl.load_config(str(path)), expected)
