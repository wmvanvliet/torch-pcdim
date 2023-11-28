"""A predictive coding model of the N400.

This is the model proposed by:

Nour Eddine, Samer, Trevor Brothers, Lin Wang, Michael Spratling, and Gina R. Kuperberg.
"A Predictive Coding Model of the N400". bioRxiv, 11 April 2023.
https://doi.org/10.1101/2023.04.10.536279.
"""
import numpy as np

from weights_nour_eddine_2023 import get_orth_repr

# These constants control the sensitivity of the model. Currently set to the values
# given in Nour Eddine et al. (2023).
eps_1 = 0.01
eps_2 = 0.0001


class PCModel:
    """Predictive coding model as defined in Nour Eddine et al. (2023).

    After instantiating the model, you can run simulation steps by calling the model
    object with a list of words as input.

    Parameters
    ----------
    weights : Weights
        A dataclass containing information about the weights, as obtained by the
        ``get_weights()`` function.
    batch_size : int
        The batch size to use.

    Examples
    --------
    m = PCModel(batch_size=4)
    input_batch =["trod", "pant", "robe", "slew"]
    # Run the simulation for 5 steps
    for step in range(5):
        m(input_batch)
    # Probe the state of the lexical units in the model
    print(m.state_lex)
    """

    def __init__(self, weights, batch_size=512):
        """Read the model weights and initialize the model."""
        self.weights = weights
        self.batch_size = batch_size

        # For convenience, make these available as attributes of the model.
        self.orth_units = weights.orth_units
        self.lex_units = weights.lex_units
        self.sem_units = weights.sem_units
        self.ctx_units = weights.ctx_units

        # Setup initial values for the units.
        self.reset(batch_size)

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        self.state_orth = (1 / 26) * np.ones(
            (len(self.orth_units), batch_size), dtype="float"
        )
        self.state_lex = (1 / len(self.lex_units)) * np.ones(
            (len(self.lex_units), batch_size), dtype="float"
        )
        self.state_sem = (1 / len(self.sem_units)) * np.ones(
            (len(self.sem_units), batch_size), dtype="float"
        )
        self.state_ctx = (1 / len(self.lex_units)) * np.ones(
            (len(self.lex_units), batch_size), dtype="float"
        )
        self.rec_orth = (1 / 26) * np.ones(
            (len(self.orth_units), batch_size), dtype="float"
        )
        self.rec_lex = (1 / len(self.lex_units)) * np.ones(
            (len(self.lex_units), batch_size), dtype="float"
        )
        self.rec_sem = (1 / len(self.sem_units)) * np.ones(
            (len(self.sem_units), batch_size), dtype="float"
        )
        self.bias_orth = (1 / 26) * np.ones(
            (len(self.orth_units), batch_size), dtype="float"
        )
        self.bias_lex = np.zeros((len(self.lex_units), batch_size), dtype="float")
        self.bias_sem = np.zeros((len(self.sem_units), batch_size), dtype="float")

    def __call__(self, input_batch=None):
        """Run the simulation on the given input batch for a single step.

        This implementation follows Algorithm 1 presented in the supplementary
        information of Nour Eddine et al. (2023).

        After calling this, the ``state_*``, ``prederr_*``, ``bias_*``, and ``rec_*``
        attributes will have been updated. These can be used to probe the new state of
        the model.

        Parameters
        ----------
        input_batch : list[str]
            A list of words (of length equal to the batch size) to feed through the
            model.
        """
        if input_batch is None:
            self.state_orth = np.zeros_like(self.state_orth)
        else:
            self.state_orth = np.array([get_orth_repr(word) for word in input_batch]).T

        self.prederr_orth = self.state_orth / np.maximum(eps_1, self.rec_orth)
        self.bias_orth = self.rec_orth / np.maximum(eps_1, self.state_orth)

        self.state_lex = np.maximum(eps_2, self.state_lex) * (
            self.weights.W_orth_lex @ self.prederr_orth
            + self.weights.I_orth_lex @ self.bias_lex
        )
        self.prederr_lex = self.state_lex / np.maximum(eps_1, self.rec_lex)
        self.bias_lex = self.rec_lex / np.maximum(eps_1, self.state_lex)

        self.state_sem = np.maximum(eps_2, self.state_sem) * (
            self.weights.W_lex_sem @ self.prederr_lex
            + (self.weights.I_lex_sem @ self.bias_sem)
        )
        self.prederr_sem = self.state_sem / np.maximum(eps_1, self.rec_sem)
        self.bias_sem = self.rec_sem / np.maximum(eps_1, self.state_sem)

        self.state_ctx = np.maximum(eps_2, self.state_ctx) * (
            self.weights.W_sem_ctx @ self.prederr_sem
        )

        # Compute reconstructions
        self.rec_orth = self.weights.V_lex_orth @ self.state_lex
        self.rec_lex = self.weights.V_sem_lex @ self.state_sem
        self.rec_sem = self.weights.V_ctx_sem @ self.state_ctx

    def get_lex_sem_prederr(self):
        """Get the current average lexico-semantic prediction error across the batch.

        Returns
        -------
        lex_sem_prederr : float
            The total lexico-semantic prediction error.
        """
        return np.mean(self.prederr_lex.sum(axis=0) + self.prederr_sem.sum(axis=0))
