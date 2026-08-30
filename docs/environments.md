# Environment setup

LangMARL ships three environments. The language tasks work with the base
install; the other two need extra dependencies, and Overcooked needs one manual
step.

---

## Language tasks (QA / Math / Coding / Writing)

Nothing beyond the base install:

```bash
pip install -e .
pip install -e ".[data]"          # pandas + pyarrow, for the data builder
python scripts/prepare_data.py
```

Set your API key once, in a git-ignored `.env` at the project root. Importing
`langmarl` loads it, so every environment -- including Overcooked's ProAgent,
which reads the key through its own config chain -- sees the same value:

```bash
echo 'OPENAI_API_KEY=sk-...' > .env
```

### What `prepare_data.py` does

Only the small evaluation splits are committed to git. The large files are
rebuilt from upstream, and both `tasks.jsonl` files come out **byte-identical**
to the ones the paper's experiments used.

| Benchmark | In git | Rebuilt by the script |
|---|---|---|
| HotPotQA | — | `hotpot_dev_{distractor,fullwiki}_v1.json`, `tasks.jsonl` (7 405 tasks) |
| MATH | `test_tasks.jsonl` (500), `test-00000-of-00001.parquet` | `competition_math` parquet, `tasks.jsonl` (12 500 tasks) |
| HumanEval | `test_tasks.jsonl` (164) | — |
| Creative writing | `tasks.jsonl` | — |

HotPotQA is fetched from the Hugging Face mirror
(`hotpotqa/hotpot_qa`) rather than the canonical `curtis.ml.cmu.edu` URLs, which
have been unreachable. The mirror carries the same records in the same order —
the script reconstructs the original JSON record-for-record.

The MATH parquet is checksummed against the exact file used for the paper; the
script fails loudly if upstream changes it.

---

## Pistonball

```bash
pip install -e ".[pistonball]"
langmarl train --config configs/pistonball/central_credit.json
```

This pins `pettingzoo==1.25.0`. Earlier releases of this repo vendored a full
copy of PettingZoo under `env/PettingZoo/`; that copy's `butterfly/pistonball/`
was byte-identical to upstream 1.25.0, so the vendored tree was dropped in
favour of the pip dependency. If `env/PettingZoo/` is still in your working
tree it is ignored by git and can be deleted.

Every piston is an agent with its own language policy, so `num_agents` must
equal `num_pistons` — the config raises if they disagree.

---

## Overcooked

```bash
pip install -e ".[overcooked]"
python scripts/setup_overcooked.py
langmarl train --config configs/overcooked/cramped_room_central_credit.json
```

### Why a setup script

`overcooked_ai` cannot simply be installed from PyPI here. LangMARL's ProAgent
planner is written against **version 0.0.1** -- the version the paper's
experiments ran on -- and neither half of what it needs is installable alone:

* 0.0.1 was never published to PyPI, whose releases start at 1.0.0. The base has
  to come from the upstream repository's `neurips2019` tag.
* The local modifications live in `env/overcooked_ai/`. They add the search
  helpers ProAgent imports (`find_path`, `get_visitable_positions`,
  `get_intersect_counter`, `query_counter_states`) and extra medium-level
  actions. They also drop four high-level planner classes that `agents/agent.py`
  still imports, so the script merges those back in.

The current PyPI release is not a substitute. `overcooked-ai==1.1.0`:

| Change in 1.1.0 | Effect |
|---|---|
| `MediumLevelPlanner` renamed to `MediumLevelActionManager` | ProAgent's planner is gone |
| `OvercookedState.from_players_pos_and_or()` lost `order_list` | the 0.0.1 planner crashes |
| layouts renamed (`simple` -> `cramped_room`, ...) | layout lookup fails |
| still calls `np.Inf` | breaks on NumPy 2.x |

`scripts/setup_overcooked.py` downloads the tag, applies the modifications,
merges the planner file and editable-installs the result into the active
environment. Run it with `--check` at any time to verify an existing install,
or `--force` to re-download and rebuild.

LangMARL itself handles both versions where it can: `resolve_layout()` picks
whichever layout names the installed package ships, and the environment uses
`OvercookedEnv.from_mdp()` when it exists and the constructor otherwise.

### First run

The first run per layout computes a motion planner and caches it under the
installed package's `data/planners/*.pkl`. `cramped_room` takes about a second;
the larger layouts take longer, and it only happens once.

Layouts: `cramped_room`, `forced_coordination`, `coordination_ring`,
`counter_circuit`, `asymmetric_advantages`.

---

## Adding your own

Subclass `langmarl.BaseEnvironment` and register it:

```python
import langmarl

@langmarl.register_env("my_env")
class MyEnv(langmarl.BaseEnvironment):
    ENV_NAME = "my_env"          # picks prompts/evaluation/game_contexts/my_env.json
    TEMPLATE_PREFIX = "my_env"   # picks prompts/evaluation/templates/my_env_<paradigm>.json

    def reset(self, task): ...
    def step(self, agent_id, action): ...
    def sample_tasks(self, n): ...
    def collect_trajectory(self, policies, task) -> langmarl.Trajectory: ...
```

Ship prompts at `<your package>/prompts/evaluation/`; if you ship none, the
critic falls back to the generic language-task prompts. Two optional hooks let
the critic read your trajectories properly:

- `format_trajectory(episode, paradigm)` — render an episode as critic-readable text.
- `critic_prompt_vars(paradigm, agents)` — extra `str.format` variables your
  templates need. The critic always supplies `trajectory`, `num_agents`,
  `task_type`, `role_context`, `agent_evaluation_sections` and
  `agent_specific_criteria`.

`tests/test_critic_prompts.py` renders every shipped template and fails if one
asks for a variable nobody supplies.
