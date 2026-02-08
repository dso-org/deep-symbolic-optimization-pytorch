"""CLI for generating DSO configs with LLM-assisted planning."""

import json
import os

import click

from dso.llm import plan_config


@click.command()
@click.option(
    "--task",
    "task_type",
    required=True,
    type=click.Choice(["regression", "control"]),
    help="Task type to plan for.",
)
@click.option(
    "--dataset",
    default=None,
    type=str,
    help="Regression dataset name or CSV path.",
)
@click.option(
    "--env",
    "env_name",
    default=None,
    type=str,
    help="Control environment name.",
)
@click.option(
    "--goal",
    default="",
    type=str,
    help="Natural-language goal used by the planner.",
)
@click.option(
    "--base_config",
    default=None,
    type=str,
    help="Optional base config JSON path. Defaults to task defaults.",
)
@click.option(
    "--output",
    required=True,
    type=str,
    help="Output path for the planned config JSON.",
)
@click.option(
    "--report_output",
    default=None,
    type=str,
    help="Optional output path for planner report JSON.",
)
@click.option(
    "--dry_run",
    is_flag=True,
    default=False,
    help="Use deterministic planner and skip LLM API calls.",
)
@click.option(
    "--model",
    required=True,
    type=str,
    help="LLM model for planning (for example: gpt-4o-mini).",
)
@click.option(
    "--base_url",
    default=lambda: os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    show_default="env:OPENAI_API_BASE or https://api.openai.com/v1",
    type=str,
    help="LLM API base URL.",
)
@click.option(
    "--api_key_env",
    default="OPENAI_API_KEY",
    show_default=True,
    type=str,
    help="Environment variable name containing the LLM API key.",
)
def main(
    task_type,
    dataset,
    env_name,
    goal,
    base_config,
    output,
    report_output,
    dry_run,
    model,
    base_url,
    api_key_env,
):
    """Generate a DSO config from user goals plus dataset/task metadata."""

    api_key = os.environ.get(api_key_env)
    if not dry_run and not api_key:
        raise click.ClickException(
            f"Missing API key in '{api_key_env}'. Use --dry_run or set {api_key_env}."
        )

    config, report = plan_config(
        task_type=task_type,
        dataset=dataset,
        env_name=env_name,
        user_goal=goal,
        base_config_path=base_config,
        dry_run=dry_run,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    if output_dir := os.path.dirname(output):
        os.makedirs(output_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if report_output is None:
        root, _ = os.path.splitext(output)
        report_output = f"{root}.report.json"
    if report_dir := os.path.dirname(report_output):
        os.makedirs(report_dir, exist_ok=True)
    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    click.echo(f"Saved planned config to: {output}")
    click.echo(f"Saved planner report to: {report_output}")
    click.echo(f"Planner mode: {report['planner_mode']}")


if __name__ == "__main__":
    main()
