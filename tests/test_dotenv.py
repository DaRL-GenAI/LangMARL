""".env is how keys reach the library, so parsing and precedence are pinned."""

import os

import pytest

from langmarl.config.dotenv import find_dotenv, load_dotenv, parse_dotenv


def test_parses_the_common_shapes():
    parsed = parse_dotenv(
        "\n".join([
            "# a comment",
            "",
            "OPENAI_API_KEY=sk-plain",
            "export EXPORTED=value",
            'QUOTED="double quoted"',
            "SINGLE='single quoted'",
            "SPACED  =  padded  ",
            "WITH_COMMENT=value # trailing",
            "EQUALS_IN_VALUE=a=b=c",
            "not a variable line",
        ])
    )
    assert parsed == {
        "OPENAI_API_KEY": "sk-plain",
        "EXPORTED": "value",
        "QUOTED": "double quoted",
        "SINGLE": "single quoted",
        "SPACED": "padded",
        "WITH_COMMENT": "value",
        "EQUALS_IN_VALUE": "a=b=c",
    }


def test_a_hash_inside_a_quoted_value_is_kept():
    assert parse_dotenv('K="a # b"') == {"K": "a # b"}


def test_loading_sets_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DEMO_ONLY_KEY=from-file\n")
    monkeypatch.delenv("DEMO_ONLY_KEY", raising=False)

    applied = load_dotenv(env)
    assert applied == {"DEMO_ONLY_KEY": "from-file"}
    assert os.environ["DEMO_ONLY_KEY"] == "from-file"


def test_the_real_environment_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DEMO_ONLY_KEY=from-file\n")
    monkeypatch.setenv("DEMO_ONLY_KEY", "from-shell")

    assert load_dotenv(env) == {}
    assert os.environ["DEMO_ONLY_KEY"] == "from-shell"


def test_override_forces_the_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DEMO_ONLY_KEY=from-file\n")
    monkeypatch.setenv("DEMO_ONLY_KEY", "from-shell")

    load_dotenv(env, override=True)
    assert os.environ["DEMO_ONLY_KEY"] == "from-file"


def test_a_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_search_stops_at_the_repository_root(tmp_path):
    """A .env above the project must not be picked up."""
    (tmp_path / ".env").write_text("OUTSIDE=1\n")
    project = tmp_path / "project"
    (project / "deep" / "nested").mkdir(parents=True)
    (project / "pyproject.toml").write_text("")

    assert find_dotenv(project / "deep" / "nested") is None


def test_search_finds_the_projects_own_file(tmp_path):
    project = tmp_path / "project"
    (project / "deep").mkdir(parents=True)
    (project / "pyproject.toml").write_text("")
    (project / ".env").write_text("INSIDE=1\n")

    assert find_dotenv(project / "deep") == project / ".env"


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "TOGETHER_API_KEY"])
def test_key_names_round_trip(name):
    assert parse_dotenv(f"{name}=abc") == {name: "abc"}
