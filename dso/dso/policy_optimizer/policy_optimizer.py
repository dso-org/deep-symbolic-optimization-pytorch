from abc import ABC, abstractmethod

import numpy as np
import torch

from dso.program import Program
from dso.utils import import_custom_source
from dso.policy.policy import Policy


def make_policy_optimizer(policy, policy_optimizer_type, **config_policy_optimizer):
    """Factory function for policy optimizer object."""

    if policy_optimizer_type == "pg":
        from dso.policy_optimizer.pg_policy_optimizer import PGPolicyOptimizer

        policy_optimizer_class = PGPolicyOptimizer
    elif policy_optimizer_type == "pqt":
        from dso.policy_optimizer.pqt_policy_optimizer import PQTPolicyOptimizer

        policy_optimizer_class = PQTPolicyOptimizer
    elif policy_optimizer_type == "ppo":
        from dso.policy_optimizer.ppo_policy_optimizer import PPOPolicyOptimizer

        policy_optimizer_class = PPOPolicyOptimizer
    else:
        policy_optimizer_class = import_custom_source(policy_optimizer_type)
        assert issubclass(
            policy_optimizer_class, PolicyOptimizer
        ), f"Custom policy optimizer {policy_optimizer_class} must subclass dso.policy_optimizer.PolicyOptimizer."

    policy_optimizer = policy_optimizer_class(policy, **config_policy_optimizer)
    return policy_optimizer


class PolicyOptimizer(ABC):
    """Abstract class for a policy optimizer."""

    def __init__(
        self,
        policy: Policy,
        debug: int = 0,
        summary: bool = False,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        entropy_weight: float = 0.005,
        entropy_gamma: float = 1.0,
    ) -> None:
        self.policy = policy
        self.debug = debug
        self.summary = summary
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight
        self.entropy_gamma = entropy_gamma

        self.n_choices = Program.library.L

        self.optimizer = self._setup_optimizer()

        if self.debug >= 1:
            total_parameters = 0
            print("")
            for name, param in self.policy.named_parameters():
                n_parameters = param.numel()
                total_parameters += n_parameters
                print("Variable:    ", name)
                print("  Shape:     ", tuple(param.shape))
                print("  Parameters:", n_parameters)
            print("Total parameters:", total_parameters)

    def _setup_optimizer(self):
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        if self.optimizer_name == "rmsprop":
            return torch.optim.RMSprop(
                self.policy.parameters(), lr=self.learning_rate, alpha=0.99
            )
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(self.policy.parameters(), lr=self.learning_rate)
        raise ValueError(f"Did not recognize optimizer '{self.optimizer_name}'")

    def _compute_entropy_terms(self, sampled_batch):
        neglogp, entropy = self.policy.make_neglogp_and_entropy(
            sampled_batch, self.entropy_gamma
        )
        entropy_loss = -self.entropy_weight * torch.mean(entropy)
        return neglogp, entropy, entropy_loss

    def _grad_norm(self):
        total_norm = 0.0
        for param in self.policy.parameters():
            if param.grad is None:
                continue
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def _summary_dict(self, **kwargs):
        return {
            key: (
                value.detach().cpu().item() if torch.is_tensor(value) else float(value)
            )
            for key, value in kwargs.items()
        }

    @abstractmethod
    def train_step(self, baseline: np.ndarray, sampled_batch):
        """Computes loss, trains model, and returns summaries."""
        raise NotImplementedError
