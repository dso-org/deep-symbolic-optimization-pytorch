import numpy as np
import torch

from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy


class PQTPolicyOptimizer(PolicyOptimizer):
    """Priority Queue Training policy gradient policy optimizer."""

    def __init__(
        self,
        policy: Policy,
        debug: int = 0,
        summary: bool = False,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        entropy_weight: float = 0.005,
        entropy_gamma: float = 1.0,
        pqt_k: int = 10,
        pqt_batch_size: int = 1,
        pqt_weight: float = 200.0,
        pqt_use_pg: bool = False,
    ) -> None:
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.pqt_weight = pqt_weight
        self.pqt_use_pg = pqt_use_pg
        super().__init__(
            policy,
            debug,
            summary,
            optimizer,
            learning_rate,
            entropy_weight,
            entropy_gamma,
        )

    def train_step(self, baseline, sampled_batch, pqt_batch):
        device = self.policy.device
        self.policy.train()
        self.optimizer.zero_grad()

        neglogp, entropy, entropy_loss = self._compute_entropy_terms(sampled_batch)
        loss = entropy_loss
        pg_loss = None
        if self.pqt_use_pg:
            rewards = torch.as_tensor(
                sampled_batch.rewards, dtype=torch.float32, device=device
            )
            baseline_t = torch.as_tensor(baseline, dtype=torch.float32, device=device)
            pg_loss = torch.mean((rewards - baseline_t) * neglogp)
            loss = loss + pg_loss

        pqt_neglogp, _ = self.policy.make_neglogp_and_entropy(
            pqt_batch, self.entropy_gamma
        )
        pqt_loss = self.pqt_weight * torch.mean(pqt_neglogp)
        loss = loss + pqt_loss

        loss.backward()
        grad_norm = self._grad_norm()
        self.optimizer.step()

        summary = self._summary_dict(
            entropy_loss=entropy_loss,
            total_loss=loss,
            pqt_loss=pqt_loss,
            reward=np.mean(sampled_batch.rewards),
            baseline=baseline,
            grad_norm=grad_norm,
        )
        if pg_loss is not None:
            summary["pg_loss"] = pg_loss.detach().cpu().item()
        return summary
