#!/usr/bin/env python3
"""Train 2-agent coding task using gpt-3.5-turbo (actor) + gpt-4o-mini (critic)."""

import os

os.environ.setdefault('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')

import langmarl
from langmarl.config.llm import LLMConfig

BASE_URL = os.environ["OPENAI_BASE_URL"]
API_KEY = os.environ["OPENAI_API_KEY"]  # export OPENAI_API_KEY=... before running

actor_llm = LLMConfig(
    name="gpt-3.5-turbo",
    model_string="openai/gpt-3.5-turbo",
    base_url=BASE_URL,
    api_key=API_KEY,
    max_tokens=4096,
)

critic_llm = LLMConfig(
    name="gpt-4o-mini",
    model_string="openai/gpt-4o-mini",
    base_url=BASE_URL,
    api_key=API_KEY,
    max_tokens=40960,
)

config = langmarl.LanguageTaskConfig(
    exp_name="coding_2agent",
    paradigm="central_credit",
    task_type="coding",
    benchmark_path=os.path.join(os.path.dirname(__file__), "..", "env", "lang_benchmark", "coding"),
    data_limit=100,
    num_agents=2,
    num_iterations=5,
    trajectories_per_iteration=100,
    actor_llm=actor_llm,
    critic_llm=critic_llm,
    optimizer_llm=critic_llm,
    experiment_dir=os.path.join(os.path.dirname(__file__), "..", "experiments"),
    checkpoint_dir=os.path.join(os.path.dirname(__file__), "..", "experiments", "ckpt_coding"),
    episode_generation_workers=16,
    optimizer_workers=16,
    max_workers=32,
    log_level="INFO",
)

env = langmarl.make_env("language", config)
critic = langmarl.CentralizedCritic(config)
optimizer = langmarl.PolicyGradientOptimizer(config.get_optimizer_llm())

callbacks = [
    langmarl.LoggingCallback(),
    langmarl.CheckpointCallback(),
]

trainer = langmarl.MonteCarloTrainer(
    config=config,
    env=env,
    critic=critic,
    optimizer=optimizer,
    callbacks=callbacks,
)

if __name__ == "__main__":
    print("=" * 60)
    print("LangMARL Coding Task Training")
    print("Actor: gpt-3.5-turbo | Critic: gpt-4o-mini")
    print("Agents: 2 | Iterations: 5 | Samples: 100")
    print("=" * 60)

    metrics = trainer.train()

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")

    if metrics:
        print("\n--- Training curve ---")
        for m in metrics:
            if m.get("type") == "iteration":
                print(f"  Iteration {m['iteration']}: avg_reward={m.get('avg_reward', 0):.3f}")

    final_policies = trainer.checkpoint.get_policies()
    for name, policy in final_policies.items():
        print(f"\n--- {name} final policy ---")
        print(policy[:500])
        if len(policy) > 500:
            print("...")
