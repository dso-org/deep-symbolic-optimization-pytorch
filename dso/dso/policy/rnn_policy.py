"""Controller used to generate distribution over hierarchical, variable-length objects."""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dso.program import Program
from dso.program import _finish_tokens
from dso.policy import Policy


def safe_cross_entropy(p, logq, axis=-1):
    """Compute p * logq safely, by substituting logq=1 for p==0."""
    safe_logq = torch.where(p == 0, torch.ones_like(logq), logq)
    return -torch.sum(p * safe_logq, dim=axis)


class StackedRNN(nn.Module):
    """Stacked RNN built from cells to allow per-layer hidden sizes."""

    def __init__(self, cell_type, input_size, hidden_sizes):
        super().__init__()
        self.cell_type = cell_type
        self.hidden_sizes = hidden_sizes
        self.cells = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_sizes):
            in_size = input_size if i == 0 else hidden_sizes[i - 1]
            if cell_type == "lstm":
                self.cells.append(nn.LSTMCell(in_size, hidden_size))
            elif cell_type == "gru":
                self.cells.append(nn.GRUCell(in_size, hidden_size))
            else:
                raise ValueError(f"Did not recognize cell type '{cell_type}'")

    def init_hidden(self, batch_size, device):
        if self.cell_type == "lstm":
            return [
                (
                    torch.zeros(batch_size, h, device=device),
                    torch.zeros(batch_size, h, device=device),
                )
                for h in self.hidden_sizes
            ]
        return [torch.zeros(batch_size, h, device=device) for h in self.hidden_sizes]

    def forward(self, inputs, hidden=None):
        # inputs: (batch, time, input_size)
        batch_size, time_steps, _ = inputs.shape
        if hidden is None:
            hidden = self.init_hidden(batch_size, inputs.device)

        outputs = []
        for t in range(time_steps):
            x = inputs[:, t, :]
            new_hidden = []
            for i, cell in enumerate(self.cells):
                if self.cell_type == "lstm":
                    h, c = hidden[i]
                    h_new, c_new = cell(x, (h, c))
                    x = h_new
                    new_hidden.append((h_new, c_new))
                else:
                    h = hidden[i]
                    h_new = cell(x, h)
                    x = h_new
                    new_hidden.append(h_new)
            outputs.append(x)
            hidden = new_hidden
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden

    def step(self, x, hidden, mask=None):
        # x: (batch, input_size)
        if mask is not None:
            mask = mask.view(-1, 1).float()
        new_hidden = []
        for i, cell in enumerate(self.cells):
            if self.cell_type == "lstm":
                h, c = hidden[i]
                h_new, c_new = cell(x, (h, c))
                if mask is not None:
                    h_new = h_new * mask + h * (1 - mask)
                    c_new = c_new * mask + c * (1 - mask)
                x = h_new
                new_hidden.append((h_new, c_new))
            else:
                h = hidden[i]
                h_new = cell(x, h)
                if mask is not None:
                    h_new = h_new * mask + h * (1 - mask)
                x = h_new
                new_hidden.append(h_new)
        return x, new_hidden


class RNNPolicy(Policy):
    """Recurrent neural network (RNN) policy used to generate expressions."""

    def __init__(
        self,
        prior,
        state_manager,
        debug=0,
        max_length=30,
        action_prob_lowerbound=0.0,
        max_attempts_at_novel_batch=10,
        sample_novel_batch=False,
        cell="lstm",
        num_layers=1,
        num_units=32,
        initializer="zeros",
        device="cpu",
    ):
        super().__init__(prior, state_manager, debug, max_length, device)

        assert 0 <= action_prob_lowerbound <= 1
        self.action_prob_lowerbound = action_prob_lowerbound

        self.n_choices = Program.library.L
        self.max_attempts_at_novel_batch = max_attempts_at_novel_batch
        self.sample_novel_batch = sample_novel_batch

        self._setup_model(cell, num_layers, num_units, initializer)
        self.to(self.device)

    def _setup_model(
        self, cell="lstm", num_layers=1, num_units=32, initializer="zeros"
    ):
        if isinstance(num_units, int):
            num_units = [num_units] * num_layers
        if len(num_units) != num_layers:
            raise ValueError("num_units must be int or list of length num_layers.")

        task = Program.task
        self.state_manager.setup_manager(self)
        dummy_obs = np.zeros((1, task.OBS_DIM), dtype=np.float32)
        input_dim = self.state_manager.get_tensor_input(dummy_obs).shape[-1]

        self.rnn = StackedRNN(
            cell_type=cell, input_size=input_dim, hidden_sizes=num_units
        )
        self.output_layer = nn.Linear(num_units[-1], self.n_choices)

        self._init_parameters(initializer)

    def _init_parameters(self, initializer):
        params = list(self.rnn.parameters()) + list(self.output_layer.parameters())
        if initializer == "zeros":
            for param in params:
                nn.init.zeros_(param)
            return
        if initializer == "var_scale":
            for param in params:
                if param.dim() >= 2:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(param)
                    denom = (fan_in + fan_out) / 2.0
                    limit = math.sqrt(3.0 * 0.5 / denom) if denom > 0 else 0.0
                    nn.init.uniform_(param, a=-limit, b=limit)
                else:
                    nn.init.uniform_(param, a=-0.1, b=0.1)
            return
        raise ValueError(f"Did not recognize initializer '{initializer}'")

    def _logits_from_obs(self, obs):
        inputs = self.state_manager.get_tensor_input(obs)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        rnn_out, _ = self.rnn(inputs)
        return self.output_layer(rnn_out)

    def make_neglogp_and_entropy(self, B, entropy_gamma):
        if entropy_gamma is None:
            entropy_gamma = 1.0

        actions = torch.as_tensor(B.actions, dtype=torch.long, device=self.device)
        obs = torch.as_tensor(B.obs, dtype=torch.float32, device=self.device)
        priors = torch.as_tensor(B.priors, dtype=torch.float32, device=self.device)
        lengths = torch.as_tensor(B.lengths, dtype=torch.long, device=self.device)

        logits = self._logits_from_obs(obs)
        if self.action_prob_lowerbound != 0.0:
            logits = self.apply_action_prob_lowerbound(logits)
        logits = logits + priors

        logprobs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)

        max_len = actions.shape[1]
        mask = (
            torch.arange(max_len, device=self.device)[None, :] < lengths[:, None]
        ).float()

        actions_one_hot = F.one_hot(actions, num_classes=self.n_choices).float()
        neglogp_per_step = safe_cross_entropy(actions_one_hot, logprobs, axis=2)
        neglogp = torch.sum(neglogp_per_step * mask, dim=1)

        entropy_gamma_decay = torch.tensor(
            [entropy_gamma**t for t in range(self.max_length)],
            dtype=torch.float32,
            device=self.device,
        )
        entropy_gamma_decay = entropy_gamma_decay[:max_len]
        entropy_gamma_decay_mask = entropy_gamma_decay.unsqueeze(0) * mask
        entropy_per_step = safe_cross_entropy(probs, logprobs, axis=2)
        entropy = torch.sum(entropy_per_step * entropy_gamma_decay_mask, dim=1)

        return neglogp, entropy

    def _sample_batch(self, n):
        task = Program.task
        prior = self.prior
        batch_size = n

        initial_obs = task.reset_task(prior)
        obs = np.broadcast_to(initial_obs, (batch_size, len(initial_obs))).astype(
            np.float32
        )
        obs = self.state_manager.process_state(obs)

        initial_prior = prior.initial_prior().astype(np.float32)
        prior_np = np.broadcast_to(initial_prior, (batch_size, self.n_choices)).astype(
            np.float32
        )

        finished = np.zeros(batch_size, dtype=bool)
        actions_list = []
        obs_list = []
        priors_list = []

        hidden = self.rnn.init_hidden(batch_size, device=self.device)

        for t in range(self.max_length):
            input_t = self.state_manager.get_tensor_input(obs).to(self.device)
            logits_t, hidden_next = self.rnn.step(
                input_t, hidden, mask=~torch.as_tensor(finished, device=self.device)
            )
            logits_t = self.output_layer(logits_t)
            if self.action_prob_lowerbound != 0.0:
                logits_t = self.apply_action_prob_lowerbound(logits_t)
            logits_t = logits_t + torch.as_tensor(prior_np, device=self.device)

            action = torch.distributions.Categorical(logits=logits_t).sample()
            action_np = action.cpu().numpy()

            actions_list.append(action_np)
            obs_list.append(obs)
            priors_list.append(prior_np)

            actions = np.stack(actions_list, axis=1)
            next_obs, next_prior, next_finished = task.get_next_obs(
                actions, obs, finished
            )

            next_obs = next_obs.astype(np.float32)
            next_obs = self.state_manager.process_state(next_obs)
            finished = np.logical_or(next_finished, t + 1 >= self.max_length)

            obs = next_obs
            prior_np = next_prior.astype(np.float32)
            hidden = hidden_next

            if finished.all():
                break

        actions = np.stack(actions_list, axis=1)
        obs = np.stack(obs_list, axis=2)
        priors = np.stack(priors_list, axis=1)
        return actions, obs, priors

    def sample(self, n):
        if self.sample_novel_batch:
            return self.sample_novel(n)
        return self._sample_batch(n)

    def sample_novel(self, n):
        n_novel = 0
        old_a, old_o, old_p = [], [], []
        new_a, new_o, new_p = [], [], []
        n_attempts = 0
        while n_novel < n and n_attempts < self.max_attempts_at_novel_batch:
            actions, obs, priors = self._sample_batch(n)
            n_attempts += 1
            new_indices = []
            old_indices = []
            for idx, a in enumerate(actions):
                tokens = _finish_tokens(a)
                key = tokens.tostring()
                if key not in Program.cache.keys() and n_novel < n:
                    new_indices.append(idx)
                    n_novel += 1
                if key in Program.cache.keys():
                    old_indices.append(idx)
            new_a.append(np.take(actions, new_indices, axis=0))
            new_o.append(np.take(obs, new_indices, axis=0))
            new_p.append(np.take(priors, new_indices, axis=0))
            old_a.append(np.take(actions, old_indices, axis=0))
            old_o.append(np.take(obs, old_indices, axis=0))
            old_p.append(np.take(priors, old_indices, axis=0))

        n_remaining = n - n_novel

        for tup, name in zip(
            [(old_a, new_a), (old_o, new_o), (old_p, new_p)], ["action", "obs", "prior"]
        ):
            dim_length = 1 if name in ["action", "prior"] else 2
            max_length = np.max(
                [list_batch.shape[dim_length] for list_batch in tup[0] + tup[1]]
            )
            for list_batch in tup:
                for idx, batch in enumerate(list_batch):
                    n_pad = max_length - batch.shape[dim_length]
                    if name == "action":
                        width = ((0, 0), (0, n_pad))
                        vals = ((0, 0), (0, 0))
                    elif name == "obs":
                        width = ((0, 0), (0, 0), (0, n_pad))
                        vals = ((0, 0), (0, 0), (0, 0))
                    else:
                        width = ((0, 0), (0, n_pad), (0, 0))
                        vals = ((0, 0), (0, 0), (0, 0))
                    list_batch[idx] = np.pad(
                        batch, pad_width=width, mode="constant", constant_values=vals
                    )

        old_a = np.concatenate(old_a) if old_a else np.empty((0, 0), dtype=np.int32)
        old_o = (
            np.concatenate(old_o) if old_o else np.empty((0, 0, 0), dtype=np.float32)
        )
        old_p = (
            np.concatenate(old_p) if old_p else np.empty((0, 0, 0), dtype=np.float32)
        )
        new_a = (
            np.concatenate(new_a + [old_a[:n_remaining]])
            if new_a
            else old_a[:n_remaining]
        )
        new_o = (
            np.concatenate(new_o + [old_o[:n_remaining]])
            if new_o
            else old_o[:n_remaining]
        )
        new_p = (
            np.concatenate(new_p + [old_p[:n_remaining]])
            if new_p
            else old_p[:n_remaining]
        )

        self.extended_batch = np.array(
            [old_a.shape[0], old_a, old_o, old_p], dtype=object
        )
        self.valid_extended_batch = True

        return new_a, new_o, new_p

    def compute_probs(self, memory_batch, log=False):
        with torch.no_grad():
            neglogp, _ = self.make_neglogp_and_entropy(memory_batch, None)
            probs = (
                (-neglogp).cpu().numpy() if log else torch.exp(-neglogp).cpu().numpy()
            )
        return probs

    def apply_action_prob_lowerbound(self, logits):
        probs = F.softmax(logits, dim=-1)
        probs_bounded = (
            1 - self.action_prob_lowerbound
        ) * probs + self.action_prob_lowerbound / float(self.n_choices)
        return torch.log(probs_bounded)
