"""Torch modules to perform predictive coding.

A model can be assembled by stacking an ``InputLayer``, as many ``PCLayer``s as needed
(can be zero) and finally an ``OutputLayer``.

These module define both a ``forward`` and ``backward`` method. First, the ``forward``
methods should be called in sequence, followed by calling all the ``backward`` methods
in reverse sequence.
"""
import torch
from torch import nn


class MiddleLayer(nn.Module):
    """A predictive-coding layer that is sandwiched between two other layers.

    This layer propagates errors onward, and back-propagates reconstructions.

    Parameters
    ----------
    n_units : int
        How many units in this layer.
    n_in : int
        How many units in the previous layer, i.e. the number of incoming connections.
    n_out : int
        How many units in the next layer, i.e. the number of outgoing connections.
    batch_size : int
        The number of inputs we compute per batch.
    td_weights : tensor (n_units, n_in) | None
        The weight matrix used to back-propagate the prediction to the previous layer.
        When not specified, a randomly initiated matrix will be used.
    bu_rate : float
        Minimum error (bottom-up or top-down) for a unit. Should be a small
        positive number.
    td_rate : float
        Minimum activation of a unit. Should be a small positive number.
    """

    def __init__(
        self,
        n_units,
        n_in,
        batch_size=1,
        td_weights=None,
        bu_rate=0.001,
        td_rate=0.0,
    ):
        super().__init__()
        self.n_units = n_units
        self.n_in = n_in
        self.batch_size = batch_size

        self.clamped = False  # see the clamp() method

        self.register_buffer("bu_rate", torch.as_tensor(bu_rate))
        self.register_buffer("td_rate", torch.as_tensor(td_rate))
        self.register_buffer("state", torch.zeros((self.batch_size, self.n_units)))
        self.register_buffer(
            "reconstruction", torch.zeros((self.batch_size, self.n_units))
        )
        self.register_buffer("td_err", torch.zeros((self.batch_size, self.n_units)))
        self.register_buffer("bu_err", torch.zeros((self.batch_size, self.n_units)))

        # Optionally initialize the weight matrices
        if td_weights is None:
            td_weights = torch.zeros((n_units, n_in))
        assert td_weights.shape == (n_units, n_in)
        self.register_parameter(
            "td_weights", nn.Parameter(td_weights, requires_grad=False)
        )

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        device = self.state.device
        self.state = torch.zeros((self.batch_size, self.n_units))
        self.reconstruction = torch.zeros((self.batch_size, self.n_units))
        self.td_err = torch.zeros((self.batch_size, self.n_units))
        self.bu_err = torch.zeros((self.batch_size, self.n_units))
        self.to(device)

    def forward(self, bu_err):
        """Update state, propagate prediction error forward.

        Parameters
        ----------
        bu_err : tensor (batch_size, n_in)
            The bottom-up error computed in the previous layer.

        Returns
        -------
        bu_err : tensor (batch_size, n_units)
            The bottom-up error that needs to propagate to the next layer.
        """
        if not self.clamped:
            self.state = (
                self.state
                + self.bu_rate * (bu_err @ self.td_weights.T)
                + self.td_rate * self.td_err
            )
        self.bu_err = self.state - self.reconstruction
        self.td_err = self.reconstruction - self.state
        return self.bu_err

    def backward(self, reconstruction):
        """Back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (bathc_size, n_units)
            The reconstruction of the state of the units in this layer that was computed
            and then back-propagated from the next layer.

        Returns
        -------
        reconstruction : tensor (batch_size, n_in)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        self.reconstruction = reconstruction
        return self.state @ self.td_weights

    def clamp(self, state):
        """Clamp the units to a predefined state.

        Parameters
        ----------
        state : tensor (batch_size, n_units)
            The clamped state of the units.
        """
        self.state = state
        self.clamped = True

    def release_clamp(self):
        """Release any clamped state from the units."""
        self.clamped = False

    def train_weights(self, bu_err):
        """Perform a training step, updating the weights.

        Parameters
        ----------
        bu_err : tensor (batch_size, n_in)
            The bottom-up error computed in the previous layer.
        """
        self.td_weights.set_(
            torch.maximum(self.td_rate, self.td_weights)
            * ((self.state.T @ bu_err) / self.state.sum(axis=0, keepdims=True).T)
        )


class InputLayer(nn.Module):
    """A predictive-coding layer that is at the bottom of the stack.

    This layer propagates errors onward, but does not compute reconstructions.

    Parameters
    ----------
    n_units : int
        How many units in this layer.
    batch_size : int
        The number of inputs we compute per batch.
    bu_rate : float
        Minimum error (bottom-up or top-down) for a unit. Should be a small
        positive number.
    td_rate : float
        Minimum activation of a unit. Should be a small positive number.
    """

    def __init__(
        self, n_units, batch_size=1, bu_weights=None, bu_rate=0.001, td_rate=0.0
    ):
        super().__init__()
        self.n_units = n_units
        self.batch_size = batch_size

        self.clamped = False  # see the clamp() method

        self.register_buffer("state", torch.zeros((self.batch_size, self.n_units)))
        self.register_buffer(
            "reconstruction",
            torch.zeros((self.batch_size, self.n_units)),
        )
        self.register_buffer("td_err", torch.zeros((self.batch_size, self.n_units)))
        self.register_buffer("bu_rate", torch.as_tensor(bu_rate))
        self.register_buffer("td_rate", torch.as_tensor(td_rate))

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        device = self.state.device
        self.state = torch.zeros((self.batch_size, self.n_units))
        self.reconstruction = torch.zeros((self.batch_size, self.n_units))
        self.td_err = torch.zeros((self.batch_size, self.n_units))
        self.to(device)

    def forward(self, x=None):
        """Update state, propagate prediction error forward.

        Parameters
        ----------
        x : tensor (batch_size, n_units) | None
            The input given to the model. This will be the new state of the
            units in this layer. Set this to ``None`` to indicate there is no input and,
            unless the units are clamped, the state of the units should be affected only
            by top-down error.

        Returns
        -------
        bu_err : tensor (batch_size, n_units)
            The bottom-up error that needs to propagate to the next layer.
        """
        if not self.clamped:
            if x is not None:
                self.state = x
            else:
                self.state = self.state + self.td_rate * self.td_err
        self.td_err = self.reconstruction - self.state
        return self.state - self.reconstruction

    def backward(self, reconstruction):
        """Take in a reconstruction for use in the next iteration.

        Parameters
        ----------
        reconstruction : tensor (batch_size, n_units)
            The reconstruction of the state of the units in this layer that was computed
            and then back-propagated from the next layer.
        """
        self.reconstruction = reconstruction

    def clamp(self, state):
        """Clamp the units to a predefined state.

        Parameters
        ----------
        state : tensor (batch_size, n_units)
            The clamped state of the units.
        """
        self.state = state
        self.clamped = True

    def release_clamp(self):
        """Release any clamped state from the units."""
        self.clamped = False


class OutputLayer(nn.Module):
    """A predictive-coding layer that is at the end of the stack.

    This layer back-propagates reconstructions, but does not propagate errors forward.

    Parameters
    ----------
    n_units : int
        How many units in this layer.
    n_in : int
        How many units in the previous layer, i.e. the number of incoming connections.
    batch_size : int
        The number of inputs we compute per batch.
    td_weights : tensor (n_units, n_in) | None
        The weight matrix used to back-propagate the prediction to the previous layer.
        When not specified, a randomly initiated matrix will be used.
    td_rate : float
        Minimum activation of a unit. Should be a small positive number.
    """

    def __init__(
        self,
        n_in,
        n_units,
        batch_size=1,
        td_weights=None,
        bu_rate=0.001,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_units = n_units
        self.batch_size = batch_size

        self.clamped = False  # see the clamp() method

        self.register_buffer("bu_rate", torch.as_tensor(bu_rate))
        self.register_buffer("state", torch.zeros((self.batch_size, self.n_units)))

        # Optionally initialize the weight matrices
        if td_weights is None:
            td_weights = torch.zeros((n_units, n_in))
        assert td_weights.shape == (n_units, n_in)
        self.register_parameter(
            "td_weights", nn.Parameter(td_weights, requires_grad=False)
        )

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        device = self.state.device
        self.state = torch.zeros((self.batch_size, self.n_units))
        self.to(device)

    def forward(self, bu_err):
        """Update state based on the bottom-up error propagated from the previous layer.

        Parameters
        ----------
        bu_err : tensor (batch_size, n_in)
            The bottom-up error computed in the previous layer.

        Returns
        -------
        state : tensor (batch_size, n_units)
            The new state of the units in this layer. This is the output of the model.
        """
        if not self.clamped:
            self.state = self.state + self.bu_rate * (bu_err @ self.td_weights.T)
        return self.state

    def backward(self):
        """Back-propagate the reconstruction.

        Returns
        -------
        reconstruction : tensor (n_in, batch_size)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        return self.state @ self.td_weights

    def clamp(self, state):
        """Clamp the units to a predefined state.

        Parameters
        ----------
        state : tensor (batch_size, n_units)
            The clamped state of the units.
        """
        self.state = state
        self.clamped = True

    def release_clamp(self):
        """Release any clamped state from the units."""
        self.clamped = False

    def train_weights(self, bu_err):
        """Perform a training step, updating the weights.

        Parameters
        ----------
        bu_err : tensor (batch_size, n_in)
            The bottom-up error computed in the previous layer.
        """
        self.td_weights.set_(
            torch.maximum(self.td_rate, self.td_weights)
            * ((self.state.T @ bu_err) / self.state.sum(axis=0, keepdims=True).T)
        )
