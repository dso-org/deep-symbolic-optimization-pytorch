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
    profile = report["profile"]
    assert "feature_diagnostics" in profile
    assert "family_scores" in profile
    assert "top_family_hypotheses" in profile
    assert "interaction_diagnostics" in profile
    assert isinstance(profile["feature_diagnostics"], list)
    assert isinstance(profile["family_scores"], dict)
    assert isinstance(profile["top_family_hypotheses"], list)


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


def test_llm_config_planner_short_traversal_controls():
    config, report = plan_config(
        task_type="regression",
        dataset="Nguyen-2",
        user_goal="Use concise, interpretable short traversals.",
        dry_run=True,
    )
    assert config["prior"]["length"]["max_"] <= 24
    assert config["policy"]["max_length"] <= 32
    assert report["length_controls"] is not None
