"""Generate model test data for DeepSymbolicOptimizer."""

import os

from pkg_resources import resource_filename

import torch
import click

from dso import DeepSymbolicOptimizer
from dso.config import load_config


# Shorter config run for parity test
CONFIG_TRAINING_OVERRIDE = {"n_samples": 1000, "batch_size": 100}


@click.command()
@click.option(
    "--stringency",
    "--t",
    default="strong",
    type=str,
    help="stringency of the test data to generate",
)
def main(stringency):
    config = load_config()
    model = DeepSymbolicOptimizer(config)

    if stringency == "strong":
        n_samples = 1000
        suffix = "_" + stringency
    elif stringency == "weak":
        n_samples = 100
        suffix = "_" + stringency
    else:
        raise ValueError("stringency must be 'strong' or 'weak'.")

    model.config_training.update({"n_samples": n_samples, "batch_size": 100})

    model.config_gp_meld.update(
        {"run_gp_meld": True, "generations": 3, "population_size": 10}
    )

    model.train()

    save_dir = resource_filename("dso.test", "data")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "test_model{}.pt".format(suffix))
    torch.save(model.policy.state_dict(), save_path)


if __name__ == "__main__":
    main()
