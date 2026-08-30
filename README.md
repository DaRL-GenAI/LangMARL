<div align="center">

# LangMARL: Natural Language Multi-Agent Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2604.00722-b31b1b?style=flat-square)](https://arxiv.org/abs/2604.00722)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026%20Main-blueviolet?style=flat-square)](https://arxiv.org/abs/2604.00722)

<p align="center">
  <a href="https://github.com/DaRL-GenAI/LangMARL"><img src="https://img.shields.io/github/stars/DaRL-GenAI/LangMARL?style=for-the-badge&logo=github&logoColor=white&color=red" alt="GitHub Stars"></a>
</p>

[**Quick Start**](#quick-start) · [**Key Components**](#key-components) · [**Environments**](#environments) · [**Citation**](#citation) · [**FAQ**](#faq)

</div>

LLM-based multi-agent systems struggle to autonomously evolve coordination strategies because coarse global outcomes obscure the causal signals needed for local policy refinement. We identify this as a **multi-agent credit assignment problem** -- well-studied in classical MARL but largely overlooked in LLM-based systems. LangMARL brings credit assignment and policy gradient evolution from cooperative MARL into language space, enabling principled autonomous optimization of multi-agent LLM systems.

---

### News

> **[Aug 2026]** LangMARL v1.0.0 released -- all three environments (language tasks, Pistonball, Overcooked) now live behind one plugin interface, with a reproducible data pipeline.
>
> **[Aug 2026]** Paper accepted to **EMNLP 2026 Main Conference**. Preprint: [arXiv:2604.00722](https://arxiv.org/abs/2604.00722).

---

## Overview

LangMARL applies **Centralized Training with Decentralized Execution (CTDE)** to language-space multi-agent reinforcement learning. It mirrors the syntax and abstractions of classical MARL libraries (e.g., TorchRL), redefining core components in natural language space.

| Traditional MARL | LangMARL |
|---|---|
| Neural network parameters $\theta$ | Text policy (LLM system prompt) |
| Numerical gradient $\nabla\theta$ | LLM-generated improvement note |
| Critic value function $Q(s,a)$ | Centralized Language Critic evaluating full trajectories |
| Credit assignment $r \to \{c^1, ..., c^N\}$ | Agent-specific language credits via causal attribution |
| Policy gradient $\nabla_\theta J$ | Language policy gradient estimator |
| Parameter update $\theta \leftarrow \theta - \alpha\nabla$ | Language policy optimizer: `new_policy = LLM_opt(policy, aggregated_gradient)` |
| Decentralized execution | Each agent uses only its own policy (system prompt) |

---

## Features

| Feature | Description |
|---------|-------------|
| **CTDE in Language Space** | Centralized Training with Decentralized Execution using natural language policies and gradients |
| **Credit Assignment** | Per-agent causal credit attribution via a Centralized Language Critic |
| **Two Training Paradigms** | `central_global` (shared gradient) and `central_credit` (per-agent credit assignment) |
| **Three Environments** | Language tasks (QA/Math/Coding), Pistonball physics simulation, and Overcooked cooking collaboration |
| **Flexible LLM Backend** | Supports OpenAI, Gemini, Llama, Qwen, DeepSeek, Ollama via unified `LLMConfig` |
| **Resumable Training** | Episodes cached to disk; training resumes from last completed iteration |
| **Plugin Architecture** | Register custom environments via `langmarl.register_env()` |
| **Emergent Role Specialization** | Agents self-organize into complementary roles through credit-driven optimization |

---

## Quick Start

### Installation

```bash
git clone https://github.com/DaRL-GenAI/LangMARL.git
cd LangMARL

pip install -e .                   # core library + language tasks
pip install -e ".[pistonball]"     # + Pistonball
pip install -e ".[overcooked]"     # + Overcooked (see docs/environments.md)
pip install -e ".[all]"            # everything
```

Put your API key in a `.env` file at the project root. It is git-ignored, and
importing `langmarl` picks it up automatically -- including for Overcooked's
ProAgent, so there is one place to set it and one place to rotate it:

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=sk-...
EOF
```

Any OpenAI-compatible endpoint works; add `OPENAI_BASE_URL` alongside it. A real
environment variable still overrides the file, so CI secrets and one-off
`OPENAI_API_KEY=... python ...` invocations keep working, and
`LANGMARL_NO_DOTENV=1` turns the lookup off.

Never hardcode a key in a tracked file -- `tests/test_no_secrets.py` fails the
suite if one appears.

Fetch the benchmark data. Only the small evaluation splits ship with the repo;
this rebuilds the large HotPotQA and MATH files from upstream:

```bash
pip install -e ".[data]"
python scripts/prepare_data.py
```

### See it run

`examples/demo_humaneval_pair.py` makes the credit assignment legible. Two
agents work on three HumanEval tasks: a **Coder** writes an implementation, a
**Tester** writes unit tests for it -- which are really executed -- and reports
what failed, and the Coder revises in light of that report. Only the Coder ever
writes code, so the two contributions stay separable: the Tester is judged on
whether its report was accurate and useful, the Coder on drafting and on acting
on the report. Both policies start threadbare (`"You write Python code."`,
`"You write unit tests."`) so the rewriting is visible.

```bash
python examples/demo_humaneval_pair.py
```

It narrates each iteration: the credit the critic assigns to each agent on every
trajectory, how `LLM_agg` reconciles them into one gradient, how `LLM_opt`
rewrites each policy, and a per-iteration breakdown of which role earned the
score -- including how often the Tester's verdict on a draft matched what the
official tests said about the same draft.

The three tasks are fixed and were picked by measurement: `gpt-4o-mini` fails
all three on every attempt under the starting policy, so the team does not begin
at its ceiling, and three trajectories per iteration keep a whole iteration's
credit readable. `--iterations`, `--model`, `--task-ids` and `--workers` change
the rest. A run costs a few cents.  Read the demo for how credit is assigned and how policies are rewritten.

### Minimal Example

```python
import langmarl

config = langmarl.LanguageTaskConfig(
    task_type="qa",
    paradigm="central_credit",
    llm=langmarl.LLMConfig.from_preset("gpt-4o-mini"),
)

env = langmarl.make_env("language", config)
trainer = langmarl.MonteCarloTrainer(
    config=config,
    env=env,
    critic=langmarl.CentralizedCritic(config),
    optimizer=langmarl.PolicyGradientOptimizer(config.get_optimizer_llm()),
)
trainer.train()
```

Or train from a config file in one line:

```python
langmarl.train("configs/language_task/qa_central_credit.json")
```

### CLI Usage

Every environment trains through the same command; the config's `env` field
selects which one.

```bash
# Language task (QA)
langmarl train --config configs/language_task/qa_central_credit.json

# Override any config field from the command line
langmarl train --config configs/language_task/qa_central_credit.json \
  --override num_iterations=10 trajectories_per_iteration=20

# Pistonball
langmarl train --config configs/pistonball/central_credit.json

# Overcooked
langmarl train --config configs/overcooked/cramped_room_central_credit.json
```

The `scripts/run_*.sh` wrappers do the same thing from the repo root.

---

## Key Components

LangMARL consists of four MARL-inspired components operating entirely in language space:

1. **Language Policy Actors** -- Each agent maintains a natural language policy $\pi_i^{text}$ and selects actions conditioned on textual observations: $a_t^i = \text{LLM}_{actor}(\pi_i^{text}, s_t^{text})$

2. **Centralized Language Critic** -- Observes complete episodic trajectories and performs **causal credit assignment** in natural language, attributing team outcomes to individual agents: $C_i^{text}(\tau) = \text{LLM}_{critic}(\tau, i)$

3. **Language Policy Gradient Estimator** -- Converts agent-specific language credits into language-form policy update directions (analogous to policy gradients): $\Delta\pi_i^{text}(\tau_k) = \text{LLM}_{grad}(\pi_i^{text}, C_i^{text}(\tau_k))$

4. **Language Policy Optimizer** -- Aggregates language gradients from multiple trajectories and applies semantic policy updates: $\pi_i^{text} \leftarrow \text{LLM}_{opt}(\pi_i^{text}, \text{LLM}_{agg}(\{\Delta\pi_i^{text}(\tau_k)\}_{k=1}^K))$

### Library API

| Class | Description |
|-------|-------------|
| `langmarl.MonteCarloTrainer` | Batch actor-critic training loop |
| `langmarl.CentralizedCritic` | Centralized critic with causal credit assignment |
| `langmarl.PolicyGradientOptimizer` | LLM-based policy gradient generation |
| `langmarl.LLMConfig` | Unified LLM backend configuration |
| `langmarl.BaseEnvironment` | Abstract base class for custom environments |
| `langmarl.make_env` / `langmarl.register_env` | Environment registry |

---

## Training Paradigms

### `central_credit` (Per-Agent Credit Assignment)

The Centralized Language Critic assigns **individual credit** to each agent by performing causal attribution over the full trajectory, producing per-agent gradients:

```
Episode trajectories -> CentralizedCritic (causal attribution) -> per-agent language credits -> per-agent gradients -> each agent updated independently
```

This is the core contribution of LangMARL -- decomposing team performance into agent-specific credits enables more targeted and efficient policy optimization.

### `central_global` (Shared Gradient)

A single LLM evaluates the entire team trajectory and produces one shared gradient applied to all agents:

```
Episode trajectories -> CentralizedCritic (full view) -> shared gradient -> all agents updated identically
```

### Training Loop

Each training iteration follows a batch actor-critic procedure:

1. **Rollout**: Collect $K$ Monte Carlo trajectories using current policies (decentralized execution)
2. **Credit Assignment**: For each trajectory and each agent, the Centralized Language Critic generates agent-specific language credits
3. **Gradient Estimation**: For each agent, convert credits into language policy gradients
4. **Gradient Aggregation**: Semantically integrate multiple trajectory-level gradients via $\text{LLM}_{agg}$ (resolving conflicts, suppressing noise)
5. **Policy Update**: Apply the aggregated gradient to update each agent's language policy

Steps 4 and 5 are one LLM call each, per agent, per iteration -- so an iteration
costs `K x N` actor rollouts, `K` critic evaluations, `K x N` gradient
generations, and `2N` optimizer calls, for `K` trajectories and `N` agents.
Each trajectory's gradient is written to
`runs/<id>/gradients/iter_<i>/<agent>_gradients.json` before aggregation, so
the per-trajectory signal stays inspectable after the merge.

---

## Environments

### Language Tasks

Multi-agent sequential collaboration on QA, Math, and Coding benchmarks.

Benchmarks: HotPotQA (QA), MATH (math), HumanEval (coding), and a creative
writing set. Run `python scripts/prepare_data.py` first.

```bash
langmarl train --config configs/language_task/qa_central_credit.json
```

### Pistonball

Multi-agent physics simulation from PettingZoo.

Each of the 20 pistons is an agent carrying its own language policy.

```bash
pip install -e ".[pistonball]"
langmarl train --config configs/pistonball/central_credit.json
```

### Overcooked

Cooperative cooking collaboration.

Two ProAgent players whose planning prompt is the policy being optimized.
Needs a patched `overcooked_ai_py`; see [docs/environments.md](docs/environments.md).

```bash
pip install -e ".[overcooked]"
python scripts/setup_overcooked.py    # overcooked_ai 0.0.1 is not on PyPI
langmarl train --config configs/overcooked/cramped_room_central_credit.json
```

---

## Configuration

All experiments are defined by JSON config files. Key fields:

| Field | Description |
|---|---|
| `env` | `"language"`, `"pistonball"` or `"overcooked"` (default: `"language"`) |
| `paradigm` | `"central_global"` or `"central_credit"` |
| `llm` | Model preset name (e.g. `"gpt-4o-mini"`) or an inline LLM object |
| `actor_llm` / `critic_llm` / `optimizer_llm` | Optional per-role overrides, each falling back to `llm` |
| `num_iterations` | Number of training iterations |
| `trajectories_per_iteration` | Episodes collected per iteration |
| `mini_batch_size` | Subset of episodes used for gradient (default: all) |
| `start_iteration` | Resume from this iteration |

<details>
<summary><b>Language task specific</b></summary>

| Field | Description |
|---|---|
| `task_type` | `"qa"`, `"math"`, `"writing"`, or `"coding"` |
| `benchmark_path` | Path to benchmark data directory |
| `num_agents` | Number of agents in the sequential chain |
| `episode_generation_workers` | Parallel workers for episode collection |

</details>

<details>
<summary><b>Overcooked specific</b></summary>

| Field | Description |
|---|---|
| `layout` | `"cramped_room"`, `"forced_coordination"`, `"coordination_ring"`, `"counter_circuit"`, `"asymmetric_advantages"` |
| `episode_horizon` | Max timesteps per episode (default: 400) |
| `p0_agent` / `p1_agent` | Agent types: `"ProAgent"` or `"Greedy"` |

</details>

<details>
<summary><b>Pistonball specific</b></summary>

| Field | Description |
|---|---|
| `num_pistons` | Number of pistons (default: 20). `num_agents` must match. |
| `max_cycles` | Max timesteps per episode (default: 125) |
| `action_mode` | `"discrete"` (0/1/2) or `"continuous"` ([-1, 1]) |

</details>

### Using Alternative LLM Backends

Any OpenAI-compatible endpoint is supported via the `llm` field:

```json
{
  "llm": {
    "name": "Qwen2.5-72B",
    "model_string": "Qwen/Qwen2.5-72B-Instruct",
    "base_url": "https://api.together.xyz/v1",
    "api_key_env_var": "TOGETHER_API_KEY"
  }
}
```

Use `{"llm": "gpt-4o-mini"}` for a preset, or give `actor_llm` / `critic_llm` /
`optimizer_llm` separately to mix models across roles.

Predefined model shortcuts: `gpt-4o`, `gpt-4o-mini`, `gemini-1.5-pro`, `gemini-1.5-flash`, `llama-3.1-70b`, `qwen2.5-72b`, `deepseek-v3`, `ollama-llama3`.

---

## Resuming Training

Training automatically resumes from the last completed iteration. Episodes already on disk are loaded rather than re-generated. To explicitly resume from a specific iteration:

```bash
langmarl train --config configs/language_task/qa_central_credit.json \
  --override start_iteration=3
```

---

## Extending to a New Environment

1. Subclass `langmarl.BaseEnvironment` with `reset`, `step`, `sample_tasks`, and `collect_trajectory` methods
2. Register your environment with `langmarl.register_env("my_env")(MyEnvClass)`
3. Add prompt templates and config files as needed

```python
import langmarl

class MyEnv(langmarl.BaseEnvironment):
    def __init__(self, config):
        ...
    def reset(self, task): ...
    def step(self, agent_id, action): ...
    def sample_tasks(self, n): ...
    def collect_trajectory(self, policies, task) -> langmarl.Trajectory: ...

langmarl.register_env("my_env")(MyEnv)
```

See `examples/quick_start.py` for a complete custom environment example, and
[`examples/demo_humaneval_pair.py`](#see-it-run) for a larger one.

---

## Project Structure

```
langmarl/                  # The library (pip-installable)
├── core/                  # MARL primitives: base classes, critic, optimizer, trajectory
├── config/                # Unified configuration system
├── trainer/               # Training loop (MonteCarloTrainer, callbacks)
├── llm/                   # LLM client and token tracking
├── envs/                  # Environment registry and the three environments
│   ├── language/          #   QA / Math / Coding / Writing benchmarks
│   ├── pistonball/        #   PettingZoo pistonball_v6
│   └── overcooked/        #   overcooked_ai + ProAgent planning
├── store/                 # Checkpointing, trajectory storage, logging
└── cli/                   # `langmarl train` entry point

configs/                   # Experiment configs, one per environment × paradigm
scripts/                   # prepare_data.py and run wrappers
env/                       # Benchmark data and the overcooked planner patch
examples/                  # Runnable examples
tests/                     # Test suite (pytest)
docs/                      # Environment setup notes
```

---

## FAQ

<details>
<summary><b>How to configure API key?</b></summary>

**Option 1** (recommended): a `.env` file at the project root. It is git-ignored
and loaded automatically when `langmarl` is imported.

```bash
echo 'OPENAI_API_KEY=sk-...' > .env
```

**Option 2**: a real environment variable, which overrides the file:

```bash
export OPENAI_API_KEY="sk-..."
```

**Option 3**: set `api_key_env_var` inside the config's `llm` object to read a
different variable, e.g. `TOGETHER_API_KEY`. Put that one in `.env` too.

Never put a literal key in a tracked file. `tests/test_no_secrets.py` fails the
suite if one appears.

</details>

<details>
<summary><b>What LLM models are supported?</b></summary>

Any OpenAI-compatible endpoint. Predefined presets include:
- **OpenAI**: GPT-4o, GPT-4o Mini
- **Google**: Gemini 1.5 Pro, Gemini 1.5 Flash
- **Open-weight**: Llama 3.1 70B, Qwen 2.5 72B, DeepSeek V3
- **Local**: Ollama (any model)

Use `langmarl.list_available_models()` to see all presets.

</details>

<details>
<summary><b>How to resume training after interruption?</b></summary>

Training automatically resumes. Episodes cached on disk are reused. To start from a specific iteration, add `--start_iteration N` or set `"start_iteration": N` in your config JSON.

</details>

<details>
<summary><b>What is the difference between central_credit and central_global?</b></summary>

- **`central_credit`**: The critic performs per-agent causal attribution, generating individual credits and gradients for each agent. This is the recommended paradigm.
- **`central_global`**: The critic produces a single shared gradient applied to all agents identically. Simpler but less targeted.

</details>

<details>
<summary><b>How to add a new environment?</b></summary>

Subclass `langmarl.BaseEnvironment`, implement the required methods (`reset`, `step`, `sample_tasks`, `collect_trajectory`), and register with `langmarl.register_env()`. See the [Extending to a New Environment](#extending-to-a-new-environment) section above.

</details>

---

## Citation

If LangMARL is useful in your research, please cite:

```bibtex
@inproceedings{yao2026langmarl,
  title     = {LangMARL: Natural Language Multi-Agent Reinforcement Learning},
  author    = {Yao, Huaiyuan and Da, Longchao and Liu, Xiaoou and
               Fleming, Charles and Chen, Tianlong and Wei, Hua},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.00722}
}
```

---

## Acknowledgements

LangMARL builds on [PettingZoo](https://github.com/Farama-Foundation/PettingZoo)
(Pistonball), [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai)
and [ProAgent](https://github.com/PKU-Alignment/ProAgent) (Overcooked), and the
HotPotQA, MATH and HumanEval benchmarks. Each keeps its own license.

---

## License

Released under the [MIT License](LICENSE).
