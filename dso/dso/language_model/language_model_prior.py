"""Language model to get prior."""

import os
import pickle

import numpy as np
import torch

from .model.model_dyn_rnn import LanguageModel


class LanguageModelPrior(object):
    """
    Language model to get prior for DSO, given token.
    History of tokens of a sequence is held as a state of language model.
    Usage: LanguageModelPrior.get_lm_prior(token)
    """

    def __init__(
        self,
        dso_library,
        model_path="./language_model/model/saved_model",
        lib_path="./language_model/model/saved_model/word_dict.pkl",
        embedding_size=32,
        num_layers=1,
        num_hidden=256,
        prob_sharing=True,
        device="cpu",
    ):

        self.dso_n_input_var = len(dso_library.input_tokens)
        self.prob_sharing = prob_sharing
        self.device = torch.device(device)

        with open(lib_path, "rb") as f:
            self.lm_token2idx = pickle.load(f)
        self.dso2lm, self.lm2dso = self.set_lib_to_lib(dso_library)

        self.language_model = LanguageModel(
            len(self.lm_token2idx),
            embedding_size,
            num_layers,
            num_hidden,
            mode="predict",
        )
        self.language_model.to(self.device)
        self.language_model.eval()
        self.load_model(model_path)

        self.next_state = None

    def load_model(self, saved_language_model_path):
        if os.path.isdir(saved_language_model_path):
            candidate = os.path.join(saved_language_model_path, "language_model.pt")
            if not os.path.exists(candidate):
                candidate = os.path.join(saved_language_model_path, "model.pt")
        else:
            candidate = saved_language_model_path

        if not os.path.exists(candidate):
            raise FileNotFoundError(
                "Expected PyTorch language model checkpoint at '{}'. "
                "Convert the TensorFlow checkpoint to a .pt file.".format(candidate)
            )

        state = torch.load(candidate, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.language_model.load_state_dict(state)

    def set_lib_to_lib(self, dso_library):
        """Match token libraries of DSO and language model."""
        dso2lm = [self.lm_token2idx["TERMINAL"]] * self.dso_n_input_var
        dso2lm += [
            self.lm_token2idx[t.name.lower()]
            for t in dso_library.tokens
            if t.input_var is None
        ]
        lm2dso = {lm_idx: i for i, lm_idx in enumerate(dso2lm)}
        return dso2lm, lm2dso

    def get_lm_prior(self, next_input):
        """Return language model prior based on given current token."""
        next_input = np.array(self.dso2lm)[next_input]
        next_input = torch.as_tensor(next_input, dtype=torch.long, device=self.device)
        if next_input.dim() == 0:
            next_input = next_input.view(1, 1)
        else:
            next_input = next_input.view(-1, 1)

        batch_size = next_input.shape[0]
        if self.next_state is None or self.next_state.shape[1] != batch_size:
            self.next_state = torch.zeros(
                self.language_model.num_layers,
                batch_size,
                self.language_model.num_hidden,
                device=self.device,
            )

        with torch.no_grad():
            logits, self.next_state, _ = self.language_model(
                next_input, keep_prob=1.0, initial_state=self.next_state
            )

        self.next_state = self.next_state.detach()
        lm_logit = logits[:, -1, :]

        if self.prob_sharing is True:
            lm_logit[:, self.lm_token2idx["TERMINAL"]] = lm_logit[
                :, self.lm_token2idx["TERMINAL"]
            ] - np.log(self.dso_n_input_var)

        lm_prior = lm_logit[:, self.dso2lm]
        return lm_prior.cpu().numpy()
