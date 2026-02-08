"""Tests for the LLM config planner."""

from dso.llm import plan_config


def test_llm_config_planner_regression_dry_run():
    config, report = plan_config(
        task_type="regression",
        dataset="Nguyen-2",
        user_goal="Fast debug baseline with robust priors",
        dry_run=True,
    )
    assert config["task"]["task_type"] == "regression"
    assert config["task"]["dataset"] == "Nguyen-2"
    assert isinstance(config["task"]["function_set"], list)
    assert len(config["task"]["function_set"]) > 0
    assert config["training"]["n_samples"] > 0
    assert report["planner_mode"] == "dry_run"


def test_llm_config_planner_control_dry_run():
    config, report = plan_config(
        task_type="control",
        env_name="CustomCartPoleContinuous-v0",
        user_goal="Quick control smoke test",
        dry_run=True,
    )
    assert config["task"]["task_type"] == "control"
    assert config["task"]["env"] == "CustomCartPoleContinuous-v0"
    assert config["training"]["n_samples"] > 0
    assert report["planner_mode"] == "dry_run"

