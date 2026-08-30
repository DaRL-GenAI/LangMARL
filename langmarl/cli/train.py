"""CLI entry point: langmarl train --config ..."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="LangMARL Training CLI")
    parser.add_argument("command", choices=["train"], help="Command to run")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--override", nargs="*", help="Key=value overrides (e.g. num_iterations=10)")
    args = parser.parse_args()

    if args.command == "train":
        _run_train(args.config, args.override)


def _run_train(config_path: str, overrides: list = None):
    import langmarl

    # Parse overrides
    override_dict = {}
    if overrides:
        for ov in overrides:
            key, val = ov.split("=", 1)
            # Try to parse as int/float
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
            override_dict[key] = val

    config = langmarl.load_config(config_path, overrides=override_dict)
    env_name = langmarl._env_name_of(config_path)

    env = langmarl.make_env(env_name, config)
    # The critic takes its prompts and trajectory format from the environment
    critic = langmarl.CentralizedCritic(config, env=env)
    optimizer = langmarl.PolicyGradientOptimizer(config.get_optimizer_llm())

    trainer = langmarl.MonteCarloTrainer(
        config=config,
        env=env,
        critic=critic,
        optimizer=optimizer,
    )

    trainer.train()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
