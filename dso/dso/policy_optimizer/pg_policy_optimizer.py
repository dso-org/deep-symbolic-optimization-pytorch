import numpy as np
import torch

from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy


class PGPolicyOptimizer(PolicyOptimizer):
    """Vanilla policy gradient policy optimizer."""

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
        super().__init__(
            policy,
            debug,
            summary,
            optimizer,
            learning_rate,
            entropy_weight,
            entropy_gamma,
        )

    def train_step(self, baseline, sampled_batch):
        device = self.policy.device
        self.policy.train()
        self.optimizer.zero_grad()

        neglogp, entropy, entropy_loss = self._compute_entropy_terms(sampled_batch)
        rewards = torch.as_tensor(
            sampled_batch.rewards, dtype=torch.float32, device=device
        )
        baseline_t = torch.as_tensor(baseline, dtype=torch.float32, device=device)

        pg_loss = torch.mean((rewards - baseline_t) * neglogp)
        loss = entropy_loss + pg_loss

        loss.backward()
        grad_norm = self._grad_norm()
        self.optimizer.step()

        return self._summary_dict(
            entropy_loss=entropy_loss,
            total_loss=loss,
            pg_loss=pg_loss,
            reward=np.mean(sampled_batch.rewards),
            baseline=baseline,
            grad_norm=grad_norm,
        )
