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
    bu_weights : tensor (n_out, n_units) | None
        The weight matrix used to propagate the error signal from the previous layer.
        When not specified, a randomly initiated matrix will be used.
    td_weights : tensor (n_in, n_units) | None
        The weight matrix used to back-propagate the prediction to the previous layer.
        When not specified, a randomly initiated matrix will be used.
    eps_1 : float
        Minimum error (bottom-up or top-down) for a unit. Should be a small
        positive number.
    eps_2 : float
        Minimum activation of a unit. Should be a small positive number.
    """

    def __init__(
        self,
        n_units,
        n_in,
        batch_size=1,
        bu_weights=None,
        td_weights=None,
        eps_1=0.01,
        eps_2=0.0001,
    ):
        super().__init__()
        self.n_units = n_units
        self.n_in = n_in
        self.batch_size = batch_size

        self.clamped = False  # see the clamp() method

        self.register_buffer("eps_1", torch.as_tensor(eps_1))
        self.register_buffer("eps_2", torch.as_tensor(eps_2))
        self.register_buffer(
            "state", (1 / self.n_units) * torch.ones((self.batch_size, self.n_units))
        )
        self.register_buffer(
            "reconstruction",
            (1 / self.n_units) * torch.ones((self.batch_size, self.n_units)),
        )
        self.register_buffer("td_err", torch.zeros((self.batch_size, self.n_units)))
        self.register_buffer("bu_err", torch.zeros((self.batch_size, self.n_units)))

        # Optionally initialize the weight matrices
        if bu_weights is None:
            bu_weights = torch.rand(n_in, n_units) * 0.1
        if td_weights is None:
            td_weights = torch.rand(n_units, n_in) * 0.1
        assert bu_weights.shape == (n_in, n_units)
        assert td_weights.shape == (n_units, n_in)
        self.register_parameter(
            "bu_weights", nn.Parameter(bu_weights, requires_grad=False)
        )
        self.register_parameter(
            "td_weights", nn.Parameter(td_weights, requires_grad=False)
        )

        normalizer = 1 / (torch.sum(bu_weights, dim=0, keepdim=True))
        self.register_buffer("normalizer", normalizer)
        self.register_buffer("bu_weights_normalized", bu_weights * normalizer)

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
        self.state = (1 / self.n_units) * torch.ones((self.batch_size, self.n_units))
        self.reconstruction = (1 / self.n_units) * torch.ones(
            (self.batch_size, self.n_units)
        )
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
            self.state = torch.maximum(self.eps_2, self.state) * (
                (bu_err @ self.bu_weights) # + (self.normalizer * self.td_err)
            )
        self.bu_err = self.state / torch.maximum(self.eps_1, self.reconstruction)
        self.td_err = self.reconstruction / torch.maximum(self.eps_1, self.state)
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
        # return self.state @ self.td_weights
        return self.state @ (self.td_weights / self.td_weights.max(axis=1, keepdim=True)[0])

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

    def train_weights(self, bu_err, lr=0.01):
        """Perform a training step, updating the weights.

        Parameters
        ----------
        bu_err : tensor (batch_size, n_in)
            The bottom-up error computed in the previous layer.
        lr : float
            The learning rate.
        """
        delta = self.state.T @ bu_err
        delta /= torch.maximum(self.eps_2, self.state.sum(axis=0, keepdims=True)).T
        delta = 1 + lr * (delta - 1)

        td_weights = torch.clamp(self.td_weights * delta, 0, 1)
        self.td_weights.set_(td_weights)
        self.bu_weights.set_(self.td_weights.T)

        self.normalizer = 1 / (torch.sum(self.bu_weights, dim=0, keepdim=True) + 1)
        self.bu_weights_normalized = self.bu_weights * self.normalizer

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"n_in={self.n_in}, n_units={self.n_units}"


class InputLayer(nn.Module):
    """A predictive-coding layer that is at the bottom of the stack.

    This layer propagates errors onward, but does not compute reconstructions.

    Parameters
    ----------
    n_units : int
        How many units in this layer.
    batch_size : int
        The number of inputs we compute per batch.
    eps_1 : float
        Minimum error (bottom-up or top-down) for a unit. Should be a small
        positive number.
    eps_2 : float
        Minimum activation of a unit. Should be a small positive number.
    """

    def __init__(
        self, n_units, batch_size=1, bu_weights=None, eps_1=0.01, eps_2=0.0001
    ):
        super().__init__()
        self.n_units = n_units
        self.batch_size = batch_size

        self.clamped = False  # see the clamp() method

        self.register_buffer(
            "state", (1 / self.n_units) * torch.ones((self.batch_size, self.n_units))
        )
        self.register_buffer(
            "reconstruction",
            (1 / self.n_units) * torch.ones((self.batch_size, self.n_units)),
        )
        self.register_buffer("td_err", torch.zeros((self.batch_size, self.n_units)))
        self.register_buffer("eps_1", torch.as_tensor(eps_1))
        self.register_buffer("eps_2", torch.as_tensor(eps_2))

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
        self.state = (1 / self.n_units) * torch.ones((self.batch_size, self.n_units))
        self.reconstruction = (1 / self.n_units) * torch.ones(
            (self.batch_size, self.n_units)
        )
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
                self.state = torch.maximum(self.eps_2, self.state) * self.td_err
        self.td_err = self.reconstruction / torch.maximum(self.eps_1, self.state)
        return self.state / torch.maximum(self.eps_1, self.reconstruction)

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

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"n_units={self.n_units}"


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
    bu_weights : tensor (n_out, n_units) | None
        The weight matrix used to propagate the error signal from the previous layer.
        When not specified, a randomly initiated matrix will be used.
    td_weights : tensor (n_in, n_units) | None
        The weight matrix used to back-propagate the prediction to the previous layer.
        When not specified, a randomly initiated matrix will be used.
    eps_2 : float
        Minimum activation of a unit. Should be a small positive number.
    """

    def __init__(
        self,
        n_in,
        n_units,
        batch_size=1,
        bu_weights=None,
        td_weights=None,
        eps_2=0.0001,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_units = n_units
        self.batch_size = batch_size

        self.clamped = False  # see the clamp() method

        self.register_buffer("eps_2", torch.as_tensor(eps_2))
        self.register_buffer(
            "state", (1 / self.n_units) * torch.ones((self.batch_size, self.n_units))
        )

        # Optionally initialize the weight matrices
        if bu_weights is None:
            bu_weights = torch.rand(n_in, n_units) * 0.1
        if td_weights is None:
            td_weights = torch.rand(n_units, n_in) * 0.1
        assert bu_weights.shape == (n_in, n_units)
        assert td_weights.shape == (n_units, n_in)
        self.register_parameter(
            "bu_weights", nn.Parameter(bu_weights, requires_grad=False)
        )
        self.register_parameter(
            "td_weights", nn.Parameter(td_weights, requires_grad=False)
        )

        normalizer = 1 / (torch.sum(bu_weights, dim=0, keepdim=True) + 1)
        self.register_buffer("normalizer", normalizer)
        self.register_buffer("bu_weights_normalized", bu_weights * normalizer)

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
        self.state = (1 / self.n_units) * torch.ones((self.batch_size, self.n_units))
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
            self.state = torch.maximum(self.eps_2, self.state) * (
                bu_err @ self.bu_weights
            )
        return self.state

    def backward(self, normalize=False):
        """Back-propagate the reconstruction.

        Returns
        -------
        reconstruction : tensor (n_in, batch_size)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        normalize : bool
            Whether the normalize the top-down weights before computing the
            reconstruction. This is done in Samer et al. 2023.
        """
        return self.state @ (self.td_weights / self.td_weights.max(axis=1, keepdim=True)[0])

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

    def train_weights(self, bu_err, lr=0.01):
        """Perform a training step, updating the weights.

        Parameters
        ----------
        bu_err : tensor (batch_size, n_in)
            The bottom-up error computed in the previous layer.
        lr : float
            The learning rate.
        """
        delta = self.state.T @ bu_err
        delta /= torch.maximum(self.eps_2, self.state.sum(axis=0, keepdims=True)).T
        delta = 1 + lr * (delta - 1)

        td_weights = torch.clamp(self.td_weights * delta, 0, 1)
        self.td_weights.set_(td_weights)
        self.bu_weights.set_(td_weights.T)

        self.normalizer = 1 / torch.sum(self.bu_weights, dim=0, keepdim=True)
        self.bu_weights_normalized = self.bu_weights * self.normalizer

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"n_in={self.n_in}, n_units={self.n_units}"
