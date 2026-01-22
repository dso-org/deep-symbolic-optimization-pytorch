from abc import ABC, abstractmethod
import weakref

import torch
from torch import nn
import torch.nn.functional as F

from dso.program import Program


class StateManager(nn.Module, ABC):
    """
    An interface for handling the torch.Tensor inputs to the Policy.
    """

    def __init__(self):
        super().__init__()

    def setup_manager(self, policy):
        """
        Function called inside the policy to perform the needed initializations.
        :param policy the policy class
        """
        # Store a weak proxy to avoid registering the policy as a submodule.
        object.__setattr__(self, "_policy_ref", weakref.proxy(policy))
        self.max_length = policy.max_length
        self.device = policy.device

    @abstractmethod
    def get_tensor_input(self, obs):
        """
        Convert an observation from a Task into a Tesnor input for the
        Policy, e.g. by performing one-hot encoding or embedding lookup.

        Parameters
        ----------
        obs : np.ndarray or torch.Tensor (dtype=float32)
            Observation coming from the Task.

        Returns
        --------
        input_ : torch.Tensor (dtype=float32)
            Tensor to be used as input to the Policy.
        """
        return

    def process_state(self, obs):
        """
        Entry point for adding information to the state tuple.
        If not overwritten, this functions does nothing
        """
        return obs


def make_state_manager(config):
    """
    Parameters
    ----------
    config : dict
        Parameters for this StateManager.

    Returns
    -------
    state_manager : StateManager
        The StateManager to be used by the policy.
    """
    manager_dict = {"hierarchical": HierarchicalStateManager}

    if config is None:
        config = {}

    # Use HierarchicalStateManager by default
    manager_type = config.pop("type", "hierarchical")

    manager_class = manager_dict[manager_type]
    state_manager = manager_class(**config)

    return state_manager


class HierarchicalStateManager(StateManager):
    """
    Class that uses the previous action, parent, sibling, and/or dangling as
    observations.
    """

    def __init__(
        self,
        observe_parent=True,
        observe_sibling=True,
        observe_action=False,
        observe_dangling=False,
        embedding=False,
        embedding_size=8,
    ):
        """
        Parameters
        ----------
        observe_parent : bool
            Observe the parent of the Token being selected?

        observe_sibling : bool
            Observe the sibling of the Token being selected?

        observe_action : bool
            Observe the previously selected Token?

        observe_dangling : bool
            Observe the number of dangling nodes?

        embedding : bool
            Use embeddings for categorical inputs?

        embedding_size : int
            Size of embeddings for each categorical input if embedding=True.
        """
        super().__init__()
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.observe_action = observe_action
        self.observe_dangling = observe_dangling
        self.library = Program.library

        # Parameter assertions/warnings
        assert (
            self.observe_action
            + self.observe_parent
            + self.observe_sibling
            + self.observe_dangling
            > 0
        ), "Must include at least one observation."

        self.embedding = embedding
        self.embedding_size = embedding_size

    def setup_manager(self, policy):
        super().setup_manager(policy)
        # Create embeddings if needed
        if self.embedding:
            if self.observe_action:
                self.action_embeddings = nn.Embedding(
                    self.library.n_action_inputs, self.embedding_size
                )
                nn.init.uniform_(self.action_embeddings.weight, a=-1.0, b=1.0)
            if self.observe_parent:
                self.parent_embeddings = nn.Embedding(
                    self.library.n_parent_inputs, self.embedding_size
                )
                nn.init.uniform_(self.parent_embeddings.weight, a=-1.0, b=1.0)
            if self.observe_sibling:
                self.sibling_embeddings = nn.Embedding(
                    self.library.n_sibling_inputs, self.embedding_size
                )
                nn.init.uniform_(self.sibling_embeddings.weight, a=-1.0, b=1.0)

    def _device(self):
        for param in self.parameters():
            return param.device
        if hasattr(self, "device"):
            return self.device
        if hasattr(self, "_policy_ref") and hasattr(self._policy_ref, "device"):
            return self._policy_ref.device
        return None

    def get_tensor_input(self, obs):
        observations = []
        if torch.is_tensor(obs):
            obs_t = obs
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device())
        if obs_t.dim() == 2:
            action = obs_t[:, 0]
            parent = obs_t[:, 1]
            sibling = obs_t[:, 2]
            dangling = obs_t[:, 3]
            extra_obs = obs_t[:, 4:] if obs_t.shape[1] > 4 else None
            sequence_mode = False
        elif obs_t.dim() == 3:
            action = obs_t[:, 0, :]
            parent = obs_t[:, 1, :]
            sibling = obs_t[:, 2, :]
            dangling = obs_t[:, 3, :]
            extra_obs = obs_t[:, 4:, :] if obs_t.shape[1] > 4 else None
            sequence_mode = True
        else:
            raise ValueError(
                "Expected obs with 2 or 3 dims, got shape {}.".format(
                    tuple(obs_t.shape)
                )
            )

        # Cast action, parent, sibling to int for embedding_lookup or one_hot
        action = action.long()
        parent = parent.long()
        sibling = sibling.long()

        # Action, parent, and sibling inputs are either one-hot or embeddings
        if self.observe_action:
            if self.embedding:
                x = self.action_embeddings(action)
            else:
                x = F.one_hot(action, num_classes=self.library.n_action_inputs).float()
            observations.append(x.float())
        if self.observe_parent:
            if self.embedding:
                x = self.parent_embeddings(parent)
            else:
                x = F.one_hot(parent, num_classes=self.library.n_parent_inputs).float()
            observations.append(x.float())
        if self.observe_sibling:
            if self.embedding:
                x = self.sibling_embeddings(sibling)
            else:
                x = F.one_hot(
                    sibling, num_classes=self.library.n_sibling_inputs
                ).float()
            observations.append(x.float())

        # Dangling input is just the value of dangling
        if self.observe_dangling:
            x = dangling.unsqueeze(-1)
            observations.append(x.float())

        input_ = torch.cat(observations, dim=-1)
        # possibly concatenates additional observations (e.g., bert embeddings)
        if extra_obs is not None and extra_obs.numel() > 0:
            if sequence_mode:
                extra_stack = torch.stack(
                    [extra_obs[:, i, :] for i in range(extra_obs.shape[1])], dim=-1
                )
            else:
                extra_stack = extra_obs
            input_ = torch.cat([input_, extra_stack.float()], dim=-1)
        return input_
