"""Parallelized, single-point launch script to run DSO on a set of benchmarks."""

import os
import sys
import time
import multiprocessing
from copy import deepcopy
from datetime import datetime

import click

from dso import DeepSymbolicOptimizer
from dso.logeval import LogEval
from dso.config import load_config
from dso.utils import safe_update_summary


def train_dso(config):
    """Trains DSO and returns dict of reward, expression, and traversal"""

    print(f'\n== TRAINING SEED {config["experiment"]["seed"]} START ============')

    # For some reason, for the control task, the environment needs to be instantiated
    # before creating the pool. Otherwise, gym.make() hangs during the pool initializer
    if (
        config["task"]["task_type"] == "control"
        and config["training"]["n_cores_batch"] > 1
    ):
        import gym
        import dso.task.control  # Registers custom and third-party environments

        gym.make(config["task"]["env"])

    # Train the model
    model = DeepSymbolicOptimizer(deepcopy(config))
    start = time.time()
    result = model.train()
    result["t"] = time.time() - start
    result.pop("program")

    save_path = model.config_experiment["save_path"]
    summary_path = os.path.join(save_path, "summary.csv")

    print(f'== TRAINING SEED {config["experiment"]["seed"]} END ==============')

    return result, summary_path


def print_summary(config, runs, messages):
    text = (
        "\n== EXPERIMENT SETUP START ===========\n"
        + f'Task type            : {config["task"]["task_type"]}\n'
    )
    if config["task"]["task_type"] == "regression":
        text += f'Dataset              : {config["task"]["dataset"]}\n'
    elif config["task"]["task_type"] == "control":
        text += f'Environment          : {config["task"]["env"]}\n'
    text += f'Starting seed        : {config["experiment"]["seed"]}\n'
    text += f"Runs                 : {runs}\n"
    if len(messages) > 0:
        text += "Additional context   :\n"
        for message in messages:
            text += f"      {message}\n"
    text += "== EXPERIMENT SETUP END ============="
    print(text)


@click.command()
@click.argument("config_template", default="")
@click.option(
    "--runs",
    "--r",
    default=1,
    type=int,
    help="Number of independent runs with different seeds",
)
@click.option(
    "--n_cores_task",
    "--n",
    default=1,
    help="Number of cores to spread out across tasks",
)
@click.option(
    "--seed",
    "--s",
    default=None,
    type=int,
    help="Starting seed (overwrites seed in config), incremented for each independent run",
)
@click.option("--benchmark", "--b", default=None, type=str, help="Name of benchmark")
@click.option(
    "--exp_name",
    default=None,
    type=str,
    help="Name of experiment to manually generate log path",
)
@click.option(
    "--debug",
    default=None,
    type=int,
    help=(
        "Training debug level (overwrites config). "
        "0: No debug. 1: Print initial parameter means. "
        "2: Print parameter means each step."
    ),
)
@click.option(
    "--n_samples",
    default=None,
    type=int,
    help="Total number of samples to draw (overwrites config).",
)
def main(config_template, runs, n_cores_task, seed, benchmark, exp_name, debug, n_samples):
    """Runs DSO in parallel across multiple seeds using multiprocessing."""

    messages = []

    # Load the experiment config
    config_template = config_template if config_template != "" else None
    config = load_config(config_template)

    # Overwrite named benchmark (for tasks that support them)
    task_type = config["task"]["task_type"]
    if benchmark is not None:
        # For regression, --b overwrites config["task"]["dataset"]
        if task_type == "regression":
            config["task"]["dataset"] = benchmark
        elif task_type == "control":
            config["task"]["env"] = benchmark
        else:
            raise ValueError(f"--b is not supported for task {task_type}.")

    # Update save dir if provided
    if exp_name is not None:
        config["experiment"]["exp_name"] = exp_name

    # Overwrite config seed, if specified
    if seed is not None:
        if config["experiment"]["seed"] is not None:
            messages.append(
                f'INFO: Replacing config seed {config["experiment"]["seed"]} with command-line seed {seed}.'
            )
        config["experiment"]["seed"] = seed

    # Overwrite training debug level, if specified
    if debug is not None:
        messages.append(
            f'INFO: Replacing training debug {config["training"]["debug"]} with command-line debug {debug}.'
        )
        config["training"]["debug"] = debug

    # Overwrite training sample budget, if specified
    if n_samples is not None:
        messages.append(
            f'INFO: Replacing training n_samples {config["training"]["n_samples"]} with command-line n_samples {n_samples}.'
        )
        config["training"]["n_samples"] = n_samples


    # Save starting seed and run command
    config["experiment"]["starting_seed"] = config["experiment"]["seed"]
    config["experiment"]["cmd"] = " ".join(sys.argv)

    # Set timestamp once to be used by all workers
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    config["experiment"]["timestamp"] = timestamp

    # Fix incompatible configurations
    if n_cores_task == -1:
        n_cores_task = multiprocessing.cpu_count()
    if n_cores_task > runs:
        messages.append(
            f"INFO: Setting 'n_cores_task' to {runs} because there are only {runs} runs."
        )
        n_cores_task = runs
    if config["training"]["verbose"] and n_cores_task > 1:
        messages.append("INFO: Setting 'verbose' to False for parallelized run.")
        config["training"]["verbose"] = False
    if config["training"]["n_cores_batch"] != 1 and n_cores_task > 1:
        messages.append(
            "INFO: Setting 'n_cores_batch' to 1 to avoid nested child processes."
        )
        config["training"]["n_cores_batch"] = 1
    if config["gp_meld"]["run_gp_meld"] and n_cores_task > 1 and runs > 1:
        messages.append(
            "INFO: Setting 'parallel_eval' to 'False' as we are already parallelizing."
        )
        config["gp_meld"]["parallel_eval"] = False

    # Start training
    print_summary(config, runs, messages)

    # Generate configs (with incremented seeds) for each run
    configs = [deepcopy(config) for _ in range(runs)]
    for i, config in enumerate(configs):
        config["experiment"]["seed"] += i

    # Farm out the work
    if n_cores_task > 1:
        pool = multiprocessing.Pool(n_cores_task)
        for i, (result, summary_path) in enumerate(
            pool.imap_unordered(train_dso, configs)
        ):
            if not safe_update_summary(summary_path, result):
                print(f"Warning: Could not update summary stats at {summary_path}")
            print(
                "INFO: Completed run {} of {} in {:.0f} s".format(
                    i + 1, runs, result["t"]
                )
            )
    else:
        for i, config in enumerate(configs):
            result, summary_path = train_dso(config)
            if not safe_update_summary(summary_path, result):
                print(f"Warning: Could not update summary stats at {summary_path}")
            print(
                "INFO: Completed run {} of {} in {:.0f} s".format(
                    i + 1, runs, result["t"]
                )
            )

    # Evaluate the log files
    print("\n== POST-PROCESS START =================")
    log = LogEval(config_path=os.path.dirname(summary_path))
    log.analyze_log(
        show_count=config["postprocess"]["show_count"],
        show_hof=config["logging"]["hof"] is not None and config["logging"]["hof"] > 0,
        show_pf=config["logging"]["save_pareto_front"],
        save_plots=config["postprocess"]["save_plots"],
    )
    print("== POST-PROCESS END ===================")


if __name__ == "__main__":
    main()
