from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from torch_predcoding import InputLayer, OutputLayer, PCLayer
from weights_nour_eddine_2023 import get_orth_repr

# These constants control the sensitivity of the model. Currently set to the values
# given in Nour Eddine et al. (2023).
eps_1 = 0.01
eps_2 = 0.0001


class PCModel(nn.Module):
    """Predictive coding model.

    Parameters
    ----------
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
                    n_out=len(self.lex_units),
                    batch_size=batch_size,
                    bu_weights=torch.as_tensor(self.weights.W_orth_lex).float(),
                ),
                lex=PCLayer(
                    n_units=len(self.lex_units),
                    n_in=len(self.orth_units),
                    n_out=len(self.sem_units),
                    batch_size=batch_size,
                    bu_weights=torch.as_tensor(self.weights.W_lex_sem).float(),
                    td_weights=torch.as_tensor(self.weights.V_lex_orth).float(),
                ),
                sem=PCLayer(
                    n_units=len(self.sem_units),
                    n_in=len(self.lex_units),
                    n_out=len(self.ctx_units),
                    batch_size=batch_size,
                    bu_weights=torch.as_tensor(self.weights.W_sem_ctx).float(),
                    td_weights=torch.as_tensor(self.weights.V_sem_lex).float(),
                ),
                ctx=OutputLayer(
                    n_in=len(self.sem_units),
                    n_units=len(self.ctx_units),
                    batch_size=batch_size,
                    td_weights=torch.as_tensor(self.weights.V_ctx_sem).float(),
                ),
            )
        )

    def __call__(self, input_batch=None):
        """Run the simulation on the given input batch for a single step.

        Parameters
        ----------
        input_batch : list
            List of words to feed through the model.

        Returns
        -------
        output : tensor
            The output of the model.
        """
        if input_batch is None:
            x = torch.zeros_like(self.layers.orth.state)
        else:
            x = torch.tensor(
                np.array([get_orth_repr(word) for word in input_batch])
            ).T.float()
        x = x.to(self.device)

        with torch.no_grad():
            # Forward pass
            prederr = self.layers[0](x)
            for layer in self.layers[1:-1]:
                prederr = layer(prederr)
            output = self.layers[-1](prederr)

            # Backward pass
            rec = self.layers[-1].backward()
            for layer in self.layers[-2:0:-1]:
                rec = layer.backward(rec)
            self.layers[0].backward(rec)

        return output

    def get_lex_sem_prederr(self):
        """Get current average lexico-semantic prediction error across the batch.

        Returns
        -------
        lex_sem_prederr : float
            The total lexico-semantic prediction error.
        """
        lex_prederr = self.layers.lex.bu_err.detach().cpu().sum(axis=0)
        sem_prederr = self.layers.sem.bu_err.detach().cpu().sum(axis=0)
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
