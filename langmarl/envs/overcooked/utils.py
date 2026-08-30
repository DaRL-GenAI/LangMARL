"""Layout tables and agent construction for the Overcooked environment."""

import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from overcooked_ai_py.agents.agent import GreedyHumanModel, RandomAgent, StayAgent
from overcooked_ai_py.planning.planners import MediumLevelPlanner

from .proagent.proagent import ProMediumLevelAgent

OLD_LAYOUTS = {
    "counter_circuit": "random3",
    "forced_coordination": "random0",
    "cramped_room": "simple",
    "coordination_ring": "random1",
    "asymmetric_advantages": "unident_s"
}
NEW_LAYOUTS = {
    "counter_circuit": "counter_circuit_o_1order",
    "forced_coordination": "forced_coordination",
    "cramped_room": "cramped_room",
    "coordination_ring": "coordination_ring",
    "asymmetric_advantages": "asymmetric_advantages"
}


def resolve_layout(layout: str) -> str:
    """Translate a LangMARL layout name into the installed package's name.

    overcooked_ai renamed its layout files between 0.0.1 -- the version the
    paper's experiments ran on, which shipped ``simple``, ``random0`` and
    friends -- and 1.1.0, which uses the descriptive names. Pick whichever one
    the installed package actually ships rather than guessing from a version
    string.
    """
    if layout not in NEW_LAYOUTS:
        raise ValueError(
            f"Unknown layout {layout!r}. Available: {sorted(NEW_LAYOUTS)}"
        )

    from overcooked_ai_py.static import LAYOUTS_DIR

    layouts_dir = Path(LAYOUTS_DIR)
    for table in (NEW_LAYOUTS, OLD_LAYOUTS):
        name = table[layout]
        if (layouts_dir / f"{name}.layout").exists():
            return name

    raise FileNotFoundError(
        f"Neither {NEW_LAYOUTS[layout]!r} nor {OLD_LAYOUTS[layout]!r} is in "
        f"{layouts_dir}. The installed overcooked_ai does not ship the "
        f"{layout!r} layout."
    )


def make_agent(alg: str, mdp, layout, **gptargs):
    if alg == "Stay":
        agent = StayAgent()

    elif alg == "Random":
        agent = RandomAgent()

    elif alg == "ProAgent" or alg == "Greedy":
        MLAM_PARAMS = {
            "start_orientations": False,
            "wait_allowed": True,
            "counter_goals": [],
            "counter_drop": [],
            "counter_pickup": [],
            "same_motion_goals": True,
        }
        counter_locations = mdp.get_counter_locations()
        MLAM_PARAMS["counter_goals"] = counter_locations
        MLAM_PARAMS["counter_drop"] = counter_locations
        MLAM_PARAMS["counter_pickup"] = counter_locations

        if alg == "ProAgent":
            mlam = MediumLevelPlanner.from_pickle_or_compute(mdp, MLAM_PARAMS, force_compute=True).ml_action_manager
            agent = ProMediumLevelAgent(mlam, layout, **gptargs)

        elif alg == "Greedy":
            mlam = MediumLevelPlanner.from_pickle_or_compute(mdp, MLAM_PARAMS, force_compute=True)
            agent = GreedyHumanModel(mlam)

    else:
        raise ValueError(f"Unsupported algorithm: {alg}")

    agent.set_mdp(mdp)

    return agent
