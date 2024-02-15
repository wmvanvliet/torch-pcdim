"""A predictive coding model of the N400.

This is the model proposed by:

Nour Eddine, Samer, Trevor Brothers, Lin Wang, Michael Spratling, and Gina R. Kuperberg.
"A Predictive Coding Model of the N400". bioRxiv, 11 April 2023.
https://doi.org/10.1101/2023.04.10.536279.
"""
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from torch_predcoding import InputLayer, MiddleLayer, OutputLayer
from weights_nour_eddine_2023 import get_lex_repr, get_orth_repr

# These constants control the sensitivity of the model. Currently set to the values
# given in Nour Eddine et al. (2023).
eps_1 = 0.01
eps_2 = 0.0001


class PCModel(nn.Module):
    """Predictive coding model.

    Parameters
    ----------
    weights : Weights
        A dataclass containing information about the weights, as obtained by the
        ``get_weights()`` function. Make sure to load them with ``pre_normalize=False``.
    batch_size : int
        Number of words we compute per batch.
    """

    def __init__(self, weights, batch_size=512):
        super().__init__()
        self.device = "cpu"

        self.weights = weights
        self.batch_size = batch_size

        # For convenience, make these available as attributes of the model.
        self.orth_units = weights.orth_units
        self.lex_units = weights.lex_units
        self.sem_units = weights.sem_units
        self.ctx_units = weights.ctx_units

        # Setup model
        self.layers = nn.Sequential(
            OrderedDict(
                orth=InputLayer(
                    n_units=len(self.orth_units),
                    batch_size=batch_size,
                ),
                lex=MiddleLayer(
                    n_units=len(self.lex_units),
                    n_in=len(self.orth_units),
                    batch_size=batch_size,
                    bu_weights=torch.as_tensor(self.weights.W_orth_lex).float().T,
                    td_weights=torch.as_tensor(self.weights.V_lex_orth).float().T,
                ),
                sem=MiddleLayer(
                    n_units=len(self.sem_units),
                    n_in=len(self.lex_units),
                    batch_size=batch_size,
                    bu_weights=torch.as_tensor(self.weights.W_lex_sem).float().T,
                    td_weights=torch.as_tensor(self.weights.V_sem_lex).float().T,
                ),
                ctx=OutputLayer(
                    n_in=len(self.sem_units),
                    n_units=len(self.ctx_units),
                    batch_size=batch_size,
                    bu_weights=torch.as_tensor(self.weights.W_sem_ctx).float().T,
                    td_weights=torch.as_tensor(self.weights.V_ctx_sem).float().T,
                ),
            )
        )

        # Normalize top-down weights
        self.layers.lex.td_weights.set_(
            self.layers.lex.td_weights * self.layers.lex.normalizer.T
        )
        self.layers.sem.td_weights.set_(
            self.layers.sem.td_weights * self.layers.sem.normalizer.T
        )
        self.layers.ctx.td_weights.set_(
            self.layers.ctx.td_weights * self.layers.ctx.normalizer.T
        )

        # Apply frequency scaling to the top-down weights
        self.layers.lex.td_weights.set_(
            (self.layers.lex.td_weights + torch.tensor(weights.freq[:, None])).float()
            * (self.layers.lex.td_weights > 0)
        )
        self.layers.sem.td_weights.set_(
            (self.layers.sem.td_weights + torch.tensor(weights.freq[None, :])).float()
            * (self.layers.sem.td_weights > 0)
        )
        self.layers.ctx.td_weights.set_(
            (self.layers.ctx.td_weights + torch.tensor(weights.freq[:, None])).float()
            * (self.layers.ctx.td_weights > 0)
        )

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        for layer in self.layers:
            layer.reset(batch_size)

    def __call__(
        self, clamp_orth=None, clamp_ctx=None, cloze_prob=1.0, train_weights=False
    ):
        """Run the simulation on the given input batch for a single step.

        This implementation follows Algorithm 1 presented in the supplementary
        information of Nour Eddine et al. (2023).

        After calling this, the ``state_*``, ``bu_err_*``, ``td_err_*``, ``bias_*``, and
        ``rec_*`` attributes will have been updated. These can be used to probe the new
        state of the model.

        Parameters
        ----------
        clamp_orth : list[str] | "zeros" | None
            A list of words (of length equal to the batch size) to clamp onto the first
            layers of the model.
            When ``"zero"``, the input will be all zeros.
            When ``None``, the input is left unclamped.
        clamp_ctx : list[str] | None
            The words to clamp the control (dummy) units to.
            When ``None``, the control units are left unclamped.
        cloze_prob : float
            The cloze probability for the words clamped onto the control units.
        train_weights : bool
            Whether to end the step by updating the weight matrices.
        """
        if clamp_orth is not None:
            if clamp_orth == "zeros":
                # Clamp zeros onto the orth units.
                state_orth = torch.zeros_like(self.layers.orth.state).float()
            else:
                # Clamp orthograhic input onto the orth units.
                state_orth = torch.tensor(
                    np.array([get_orth_repr(word) for word in clamp_orth])
                ).float()
            state_orth = state_orth.to(self.device)
            self.layers.orth.clamp(state_orth)
        else:
            self.layers.orth.release_clamp()

        if clamp_ctx is not None:
            state_ctx = torch.tensor(
                np.array(
                    [
                        get_lex_repr(word, self.lex_units, cloze_prob=cloze_prob)
                        for word in clamp_ctx
                    ]
                )
            ).float()
            state_ctx = state_ctx.to(self.device)
            self.layers.ctx.clamp(state_ctx)
        else:
            self.layers.ctx.release_clamp()

        with torch.no_grad():
            # Forward pass
            prederr = [self.layers[0]()]
            for layer in self.layers[1:-1]:
                prederr.append(layer(prederr[-1]))
            output = self.layers[-1](prederr[-1])

            # Backward pass
            rec = self.layers[-1].backward()
            for layer in self.layers[-2:0:-1]:
                rec = layer.backward(rec)
            self.layers[0].backward(rec)

            # Update weights
            if train_weights:
                self.layers.lex.train_weights(prederr[0])
                # for layer, err in zip(self.layers[1:], prederr):
                #     layer.train_weights(err)

        return output

    def get_lex_sem_prederr(self):
        """Get current average lexico-semantic prediction error across the batch.

        Returns
        -------
        lex_sem_prederr : float
            The total lexico-semantic prediction error.
        """
        lex_prederr = self.layers.lex.bu_err.detach().cpu().sum(axis=1)
        sem_prederr = self.layers.sem.bu_err.detach().cpu().sum(axis=1)
        return torch.mean(lex_prederr + sem_prederr).item()

    def to(self, device):
        """Move model to a given GPU (or CPU).

        Parameters
        ----------
        device : str | torch.device
            The device to move the model to.

        Returns
        -------
        self : PCModel
            Returns the model itself for convenience.
        """
        super().to(device)
        self.device = device
        return self

    def cuda(self):
        """Move model to first available GPU.

        Returns
        -------
        self : PCModel
            Returns the model itself for convenience.
        """
        return self.to("cuda")

    def cpu(self):
        """Move model to first available CPU.

        Returns
        -------
        self : PCModel
            Returns the model itself for convenience.
        """
        return self.to("cpu")
