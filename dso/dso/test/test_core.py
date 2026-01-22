"""Test cases for DeepSymbolicOptimizer on each Task."""

import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

from dso import DeepSymbolicOptimizer
from dso.config import load_config
from dso.program import from_tokens
from dso.memory import Batch
from dso.policy_optimizer.pg_policy_optimizer import PGPolicyOptimizer


@pytest.fixture
def model():
    config = load_config()
    config["experiment"]["logdir"] = None
    return DeepSymbolicOptimizer(config)


def _config_path(relative_path):
    return str(Path(__file__).resolve().parents[1] / relative_path)


@pytest.mark.parametrize(
    "config", ["config/config_regression.json", "config/config_control.json"]
)
def test_task(model, config):
    """Test that Tasks do not crash for various configs."""
    if (
        config == "config/config_control.json"
        and importlib.util.find_spec("gym") is None
    ):
        pytest.skip("gym not installed")
    config = load_config(_config_path(config))
    config["experiment"]["logdir"] = None
    model.set_config(config)
    model.config_training.update({"n_samples": 10, "batch_size": 5})
    model.train()


def _make_sampled_batch(policy, batch_size):
    actions, obs, priors = policy.sample(batch_size)
    programs = [from_tokens(a) for a in actions]
    lengths = np.array(
        [min(len(p.traversal), policy.max_length) for p in programs], dtype=np.int32
    )
    rewards = np.ones(batch_size, dtype=np.float32)
    on_policy = np.ones(batch_size, dtype=np.int32)
    return Batch(
        actions=actions,
        obs=obs,
        priors=priors,
        lengths=lengths,
        rewards=rewards,
        on_policy=on_policy,
    )


def test_policy_neglogp_entropy_shapes(model):
    config = load_config(_config_path("config/config_regression.json"))
    config["experiment"]["logdir"] = None
    model.set_config(config)
    model.setup()

    batch = _make_sampled_batch(model.policy, batch_size=4)
    neglogp, entropy = model.policy.make_neglogp_and_entropy(batch, entropy_gamma=1.0)
    assert neglogp.shape == (4,)
    assert entropy.shape == (4,)


def test_pg_optimizer_step_updates_params(model):
    config = load_config(_config_path("config/config_regression.json"))
    config["experiment"]["logdir"] = None
    model.set_config(config)
    model.setup()

    batch = _make_sampled_batch(model.policy, batch_size=4)
    optimizer = PGPolicyOptimizer(model.policy, learning_rate=0.01, entropy_weight=0.0)

    params_before = [p.detach().clone() for p in model.policy.parameters()]
    optimizer.train_step(0.0, batch)
    params_after = list(model.policy.parameters())

    assert any(not torch.allclose(a, b) for a, b in zip(params_before, params_after))
