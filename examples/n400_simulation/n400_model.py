from collections import OrderedDict

import numpy as np
import torch
from torch_pcdim.layers import InputLayer, MiddleLayer, OutputLayer
from torch_pcdim.models import PCModel

from weights import get_lex_repr, get_orth_repr


class N400Model(PCModel):
    """Predictive coding model simulating the N400.

    This is the model proposed by:

    Nour Eddine, Samer, Trevor Brothers, Lin Wang, Michael Spratling, and Gina R.
    Kuperberg.  "A Predictive Coding Model of the N400". bioRxiv, 11 April 2023.
    https://doi.org/10.1101/2023.04.10.536279.

    and replicates its results exactly.

    Parameters
    ----------
    weights : Weights
        A dataclass containing information about the weights, as obtained by the
        ``get_weights()`` function. Make sure to load them with ``pre_normalize=False``.
    batch_size : int
        Number of words we compute per batch.
    """

    def __init__(self, weights, batch_size=512):
        # These constants control the sensitivity of the model.
        self.eps_1 = 0.01
        self.eps_2 = 0.0001

        # Setup model.
        super().__init__(
            torch.nn.Sequential(
                OrderedDict(
                    orth=InputLayer(
                        n_units=len(weights.orth_units),
                        batch_size=batch_size,
                        eps_1=self.eps_1,
                        eps_2=self.eps_2,
                    ),
                    lex=MiddleLayer(
                        n_units=len(weights.lex_units),
                        n_in=len(weights.orth_units),
                        batch_size=batch_size,
                        bu_weights=torch.as_tensor(weights.W_orth_lex).float().T,
                        td_weights=torch.as_tensor(weights.V_lex_orth).float().T,
                        eps_1=self.eps_1,
                        eps_2=self.eps_2,
                    ),
                    sem=MiddleLayer(
                        n_units=len(weights.sem_units),
                        n_in=len(weights.lex_units),
                        batch_size=batch_size,
                        bu_weights=torch.as_tensor(weights.W_lex_sem).float().T,
                        td_weights=torch.as_tensor(weights.V_sem_lex).float().T,
                        eps_1=self.eps_1,
                        eps_2=self.eps_2,
                    ),
                    ctx=OutputLayer(
                        n_in=len(weights.sem_units),
                        n_units=len(weights.ctx_units),
                        batch_size=batch_size,
                        bu_weights=torch.as_tensor(weights.W_sem_ctx).float().T,
                        td_weights=torch.as_tensor(weights.V_ctx_sem).float().T,
                        eps_2=self.eps_2,
                    ),
                )
            )
        )

        # Normalize top-down weights.
        self.layers.lex.td_weights.set_(
            self.layers.lex.td_weights * self.layers.lex.normalizer.T
        )
        self.layers.sem.td_weights.set_(
            self.layers.sem.td_weights * self.layers.sem.normalizer.T
        )
        self.layers.ctx.td_weights.set_(
            self.layers.ctx.td_weights * self.layers.ctx.normalizer.T
        )

        # Apply frequency scaling to the top-down weights.
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

        # For convenience, make these available as attributes of the model.
        self.weights = weights
        self.orth_units = weights.orth_units
        self.lex_units = weights.lex_units
        self.sem_units = weights.sem_units
        self.ctx_units = weights.ctx_units

    def __call__(
        self, clamp_orth=None, clamp_ctx=None, cloze_prob=1.0, train_weights=False
    ):
        """Run the simulation on the given input batch for a single step.

        After calling this, the ``get_lex_sem_prederr`` method can be used to query the
        prediction errors within the model.

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
        # Clamp the desired kind of input.
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
            state_orth = None
            self.layers.orth.release_clamp()

        # Clamp the desired kind of output.
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

        # Run the model.
        output = self.forward()
        self.backward()

        # Optionally update weights.
        if train_weights:
            bu_err = self.layers.orth(state_orth)
            self.layers.lex.train_weights(bu_err, lr=1)

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
