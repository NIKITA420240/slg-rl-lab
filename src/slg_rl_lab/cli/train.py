from slg_rl_lab.perception.train import train


def run_cli() -> None:
    """Run perception training using its Hydra configuration."""
    train()


if __name__ == "__main__":
    run_cli()
