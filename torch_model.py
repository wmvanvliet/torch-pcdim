"""Full predictive coding models."""
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from torch_predcoding import InputLayer, MiddleLayer, OutputLayer
from weights_nour_eddine_2023 import get_lex_repr, get_orth_repr


class PCModel(nn.Module):
    """A full predictive coding model.

    This class wraps around a list of ``torch.nn.Module`` objects that represent the
    different layers of the model. During the ``forward()`` and ``backward()`` passed,
    all modules will be called in sequence, propagating prediction error and
    reconstructions.

    There should be at least two layers: the input and output layers (of types
    ``InputLayer`` and ``OutputLayer`` respectively. Layers are accessible though
    ``model.layers.{name}`` attributes. By default, the first layer is called ``input``,
    the final layer is called ``output`` and all other layers are called ``hidden{i}``,
    but these names can be overridden by providing a ``torch.nn.Sequential`` object.

    Parameters
    ----------
    layers : list of torch.nn.Module | torch.nn.Sequential
        The layers of the model. These should be predictive coding layers as defined in
        ``torch_predcoding.py``
    batch_size : int
        The batch size used during operation of the model.
    """

    def __init__(self, layers, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

        assert len(layers) >= 2, "The model must have at least 2 layers (input/output)"

        # give layers names
        if not isinstance(layers, nn.Sequential):
            assert all([isinstance(layer, nn.Module) for layer in layers])
            names = (
                ["input"]
                + [f"hidden{i}" for i in range(1, len(layers) - 1)]
                + ["output"]
            )
            layers = nn.Sequential(OrderedDict(zip(names, layers)))

        # make sure there are input/output layers
        assert isinstance(layers[0], InputLayer)
        assert isinstance(layers[-1], OutputLayer)

        self.layers = layers
        self.eval()  # we're not learning weights

    def clamp(self, input_data=None, output_data=None):
        """Clamp input/output units unto a given state.

        Parameters
        ----------
        input_data: tensor (batch_size, n_in) | None
            The data to be clamped to the input layer. If left at ``None``, do not clamp
            the input layer to anything.
        output_data: tensor (batch_size, n_out) | None
            The data to be clamped to the output layer. If left at ``None``, do not
            clamp the output layer to anything.
        """
        if input_data is not None:
            self.layers[0].clamp(input_data)
        if output_data is not None:
            self.layers[-1].clamp(output_data)

    def release_clamp(self):
        """Release any clamps on the input and output units."""
        self.layers[0].release_clamp()
        self.layers[-1].release_clamp()

    def forward(self, input_data=None):
        """Perform a forward pass throught the model.

        Parameters
        ----------
        input_data: tensor (batch_size, n_in) | None
            The data to be clamped to the input layer during the forward pass. If left
            at ``None``, do not clamp the input layer to anything.

        Returns
        -------
        output_data: tensor (batch_size, n_out)
            The state of the output units in the model.
        """
        with torch.no_grad():
            bu_err = self.layers[0](input_data)
            for layer in self.layers[1:-1]:
                bu_err = layer(bu_err)
            output_data = self.layers[-1](bu_err)
        return output_data

    def backward(self):
        """Perform a backward pass through the model.

        Returns
        -------
        reconstruction: tensor (batch_size, n_in)
            The reconstruction of the input units made by the upper layers.
        """
        with torch.no_grad():
            reconstruction = self.layers[-1].backward()
            for layer in self.layers[-2::-1]:
                reconstruction = layer.backward(reconstruction)
            return reconstruction

    def train_weights(self, input_data, lr=0.01):
        """Perform a training step, updating the weights.

        For training to work properly, make sure to have the desired target output
        clamped onto the output nodes.

        Parameters
        ----------
        input_data: tensor (batch_size, n_in)
            The data to be clamped to the input layer during the weight training pass.
            If left at ``None``, do not clamp the input layer to anything.
        output_data: tensor (batch_size, n_in)
            The data to be clamped to the output layer during the weight training pass.
            If left at ``None``, do not clamp the input layer to anything.
        lr : float
            The learning rate to use when updating weights.
        """
        with torch.no_grad():
            bu_errors = [self.layers[0](input_data)]
            for layer in self.layers[1:-1]:
                bu_errors.append(layer(bu_errors[-1]))
            for layer, bu_err in zip(self.layers[1:], bu_errors):
                layer.train_weights(bu_err, lr=lr)

    def reset(self, batch_size=None):
        """Reset the state of all units to small randomized values.

        Parameters
        ----------
        batch_size : int | None
            Optionally change the batch size used by the model.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        for layer in self.layers:
            layer.reset(batch_size)

    @property
    def device(self):
        """Current device the model is loaded on."""
        return next(self.parameters()).device


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
            nn.Sequential(
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
