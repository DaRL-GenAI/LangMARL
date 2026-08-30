"""Per-agent credit is parsed out of each environment's critic response format."""

from langmarl.core.optimizer import PolicyGradientOptimizer

parse = PolicyGradientOptimizer.parse_credit_response


def test_language_json_format():
    response = '```json\n{"agent_1": "good setup", "agent_2": "solid finish"}\n```'
    result = parse(response, ["agent_1", "agent_2"])
    assert result["agent_1"] == "good setup"
    assert result["agent_2"] == "solid finish"


def test_pistonball_marker_format():
    response = (
        "[PISTON_0 EVALUATION]\npushed too early\n"
        "[PISTON_1 EVALUATION]\nheld position well\n"
    )
    result = parse(response, ["piston_0", "piston_1"])
    assert "pushed too early" in result["piston_0"]
    assert "held position well" in result["piston_1"]


def test_overcooked_marker_format():
    response = (
        "[AGENT 0 EVALUATION]\nfetched onions\n"
        "[AGENT 1 EVALUATION]\nplated the soup\n"
    )
    result = parse(response, ["agent_0", "agent_1"])
    assert "fetched onions" in result["agent_0"]
    assert "plated the soup" in result["agent_1"]


def test_unparseable_response_falls_back_to_the_whole_text():
    response = "The team did fine overall."
    result = parse(response, ["agent_1", "agent_2"])
    assert set(result) == {"agent_1", "agent_2"}
    assert all(response in v for v in result.values())


def test_no_agents_yields_nothing():
    assert parse("anything", []) == {}
