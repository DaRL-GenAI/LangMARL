#!/usr/bin/env python3
"""
LangMARL Quick Start Demo — 2-Agent QA with Policy Gradient Optimization

This demo shows the core LangMARL loop WITHOUT benchmark data:
  1. Two LLM agents collaborate on a QA task (sequential message pool)
  2. A centralized critic evaluates the team's performance
  3. The optimizer generates a "language gradient" (improvement instruction)
  4. The gradient is applied to each agent's policy for the next round

Requirements:
  pip install langmarl
  echo 'OPENAI_API_KEY=sk-...' > .env      # git-ignored, loaded on import

Usage:
  python examples/quick_start.py
  python examples/quick_start.py --model gpt-4o --iterations 3
"""

import argparse
import os

import langmarl

os.environ.setdefault('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')



# ─── Inline task (no benchmark files needed) ────────────────────────────────
DEMO_TASK = {
    "task_id": "demo_001",
    "question": (
        "What is the name of the famous physicist who developed the theory of "
        "general relativity, and in what year was he awarded the Nobel Prize in Physics?"
    ),
    "ground_truth": "Albert Einstein, 1921",
}


# ─── Custom environment: runs 2-agent QA without benchmark data ─────────────
class DemoQAEnv(langmarl.BaseEnvironment):
    """Minimal environment that runs a 2-agent QA collaboration on a single task."""

    def __init__(self, config):
        self.num_agents = config.num_agents
        llm = config.get_actor_llm()
        self.llm_client = langmarl.LLMClient(llm)

    def reset(self, task: dict) -> dict:
        return {"task": task}

    def step(self, agent_id: str, action: str):
        return {}, 0.0, False, {}

    def sample_tasks(self, num_samples: int) -> list[dict]:
        """Always return the same demo task."""
        return [DEMO_TASK] * num_samples

    def collect_trajectory(
        self, policies: dict[str, str], task: dict
    ) -> langmarl.Trajectory:
        """Run 2-agent sequential collaboration."""
        question = task["question"]
        ground_truth = task.get("ground_truth", "")

        # Agent 1: initial analysis
        policy_1 = policies.get("agent_1", "You are Agent 1. Provide initial analysis.")
        user_1 = f"Task:\n{question}"
        response_1, tok_1 = self.llm_client.chat_with_usage(policy_1, user_1)

        # Agent 2: final answer
        policy_2 = policies.get("agent_2", "You are Agent 2. Provide the final answer.")
        user_2 = (
            f"Task:\n{question}\n\n"
            f"Agent 1's Response:\n{response_1}\n\n"
            f"Now provide your FINAL answer:"
        )
        response_2, tok_2 = self.llm_client.chat_with_usage(policy_2, user_2)

        # Simple evaluation: check if ground truth appears in the final answer
        if ground_truth.lower() in response_2.lower():
            reward = 1.0
        else:
            # Partial credit if key terms appear
            terms = ground_truth.lower().split(",")
            matched = sum(1 for t in terms if t.strip() in response_2.lower())
            reward = matched / max(len(terms), 1)

        steps = [
            {
                "agent": "agent_1",
                "agent_id": "agent_1",
                "system_prompt": policy_1,
                "input": user_1,
                "output": response_1,
                "action": response_1,
                "tokens": tok_1,
            },
            {
                "agent": "agent_2",
                "agent_id": "agent_2",
                "system_prompt": policy_2,
                "input": user_2,
                "output": response_2,
                "action": response_2,
                "tokens": tok_2,
            },
        ]

        return langmarl.Trajectory(
            task=task,
            steps=steps,
            reward=reward,
            metadata={"task_type": "qa", "final_answer": response_2},
        )


# ─── Register and run ───────────────────────────────────────────────────────
langmarl.register_env("demo_qa")(DemoQAEnv)


def main(model_name: str = "gpt-4o-mini", num_iterations: int = 2):
    print("LangMARL Quick Start Demo")
    print(f"Model: {model_name} | Iterations: {num_iterations} | Paradigm: central_global\n")

    # ── Step 1: Configure ───────────────────────────────────────────────────
    config = langmarl.BaseConfig(
        exp_name="demo_qa",
        paradigm="central_global",
        llm=langmarl.LLMConfig.from_preset(model_name),
        num_agents=2,
        num_iterations=num_iterations,
        trajectories_per_iteration=1,  # single task demo
        max_workers=1,
    )

    # ── Step 2: Create components ────────────────────────────────────────────
    env = langmarl.make_env("demo_qa", config)
    critic = langmarl.CentralizedCritic(config)
    optimizer = langmarl.PolicyGradientOptimizer(config.get_optimizer_llm())

    # ── Step 3: Train! ───────────────────────────────────────────────────────
    trainer = langmarl.MonteCarloTrainer(
        config=config,
        env=env,
        critic=critic,
        optimizer=optimizer,
    )

    metrics = trainer.train()

    # ── Step 4: Inspect results ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    # Show final policies
    final_policies = trainer.checkpoint.get_policies()
    for name, policy in final_policies.items():
        print(f"\n--- {name} final policy ---")
        print(policy[:300])
        if len(policy) > 300:
            print("...")

    # Show training curve
    if metrics:
        print("\n--- Training curve ---")
        for m in metrics:
            if m.get("type") == "iteration":
                print(f"  Iteration {m['iteration']}: avg_reward={m.get('avg_reward', 0):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangMARL Quick Start Demo")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("--iterations", type=int, default=2, help="Training iterations")
    args = parser.parse_args()
    main(args.model, args.iterations)
