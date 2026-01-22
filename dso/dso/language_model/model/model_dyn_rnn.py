"""Model architecture of default (saved) LanguageModel in PyTorch."""

import torch
from torch import nn
import torch.nn.functional as F


class LanguageModel(nn.Module):
    def __init__(
        self, vocabulary_size, embedding_size, num_layers, num_hidden, mode="train"
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.mode = mode

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.rnn = nn.RNN(
            embedding_size, num_hidden, num_layers=num_layers, batch_first=True
        )
        self.output_layer = nn.Linear(num_hidden, vocabulary_size)

    def forward(self, x, keep_prob=1.0, initial_state=None):
        # x: (batch, seq_len)
        if self.mode == "train":
            lm_input = x[:, :-2]
            seq_len = x[:, -1].long()
        else:
            lm_input = x
            seq_len = torch.sum(lm_input != 0, dim=1).long()

        emb = self.embedding(lm_input)
        if keep_prob < 1.0:
            emb = F.dropout(emb, p=1 - keep_prob, training=self.training)

        rnn_outputs, last_state = self.rnn(emb, initial_state)
        rnn_outputs = F.dropout(rnn_outputs, p=1 - keep_prob, training=self.training)
        logits = self.output_layer(rnn_outputs)
        return logits, last_state, seq_len

    def sequence_loss(self, logits, targets, seq_len):
        # logits: (batch, time, vocab), targets: (batch, time)
        batch, time_steps, vocab = logits.shape
        logits_flat = logits.reshape(batch * time_steps, vocab)
        targets_flat = targets.reshape(batch * time_steps)
        loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = loss.reshape(batch, time_steps)
        mask = (
            torch.arange(time_steps, device=seq_len.device)[None, :] < seq_len[:, None]
        ).float()
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss
