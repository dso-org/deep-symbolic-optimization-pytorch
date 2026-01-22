import numpy as np
import torch

from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy
from dso.memory import Batch


class PPOPolicyOptimizer(PolicyOptimizer):
    """Proximal policy optimization policy optimizer."""

    def __init__(
        self,
        policy: Policy,
        debug: int = 0,
        summary: bool = False,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        entropy_weight: float = 0.005,
        entropy_gamma: float = 1.0,
        ppo_clip_ratio: float = 0.2,
        ppo_n_iters: int = 10,
        ppo_n_mb: int = 4,
    ) -> None:
        self.ppo_clip_ratio = ppo_clip_ratio
        self.ppo_n_iters = ppo_n_iters
        self.ppo_n_mb = ppo_n_mb
        self.rng = np.random.RandomState(0)
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
        rewards = torch.as_tensor(
            sampled_batch.rewards, dtype=torch.float32, device=device
        )
        baseline_t = torch.as_tensor(baseline, dtype=torch.float32, device=device)
        n_samples = sampled_batch.rewards.shape[0]

        with torch.no_grad():
            old_neglogp, _ = self.policy.make_neglogp_and_entropy(
                sampled_batch, self.entropy_gamma
            )

        indices = np.arange(n_samples)
        summary = {}
        for _ in range(self.ppo_n_iters):
            self.rng.shuffle(indices)
            minibatches = np.array_split(indices, self.ppo_n_mb)
            for mb in minibatches:
                mb_idx = torch.as_tensor(mb, dtype=torch.long, device=device)
                sampled_batch_mb = Batch(
                    **{
                        name: array[mb]
                        for name, array in sampled_batch._asdict().items()
                    }
                )

                self.policy.train()
                self.optimizer.zero_grad()

                neglogp, entropy, entropy_loss = self._compute_entropy_terms(
                    sampled_batch_mb
                )
                ratio = torch.exp(old_neglogp[mb_idx] - neglogp)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.ppo_clip_ratio, 1.0 + self.ppo_clip_ratio
                )
                adv = rewards[mb_idx] - baseline_t
                ppo_loss = -torch.mean(adv * torch.min(ratio, clipped_ratio))
                loss = entropy_loss + ppo_loss

                loss.backward()
                grad_norm = self._grad_norm()
                self.optimizer.step()

                clipped = (ratio < (1.0 - self.ppo_clip_ratio)) | (
                    ratio > (1.0 + self.ppo_clip_ratio)
                )
                clip_fraction = clipped.float().mean()
                sample_kl = torch.mean(neglogp - old_neglogp[mb_idx])

                summary = self._summary_dict(
                    entropy_loss=entropy_loss,
                    total_loss=loss,
                    ppo_loss=ppo_loss,
                    reward=np.mean(sampled_batch.rewards),
                    baseline=baseline,
                    grad_norm=grad_norm,
                    clip_fraction=clip_fraction,
                    sample_kl=sample_kl,
                )

        return summary
