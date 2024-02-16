"""Torch modules to perform predictive coding.

A model can be assembled by stacking an ``InputLayer``, as many ``PCLayer``s as needed
(can be zero) and finally an ``OutputLayer``.

These module define both a ``forward`` and ``backward`` method. First, the ``forward``
methods should be called in sequence, followed by calling all the ``backward`` methods
in reverse sequence.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    """A predictive-coding convolution layer.

    This layer propagates errors onward, and back-propagates reconstructions.

    Parameters
    ----------
    n_in_channels : int
        How many channels the data coming into the layer has.
    n_out_channels : int
        How many channels the data coming out of this layer has.
    kernel_size : int
        The size of the convolution kernel.
    in_width : int
        The width of the data coming into the layer. Height is assumed to be the same.
    batch_size : int
        The number of inputs we compute per batch.
    td_weights : tensor (kernel_size, kernel_size) | None
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
        n_in_channels,
        n_out_channels,
        kernel_size,
        in_width,
        batch_size,
        padding=0,
        stride=1,
        dilation=1,
        bu_weights=None,
        eps_1=0.01,
        eps_2=0.0001,
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.kernel_size = kernel_size
        self.in_width = in_width
        self.batch_size = batch_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.out_width = (
            math.floor((self.in_width + (2 * padding) - kernel_size) / stride) + 1
        )
        self.shape = (batch_size, n_out_channels, self.out_width, self.out_width)
        self.n_units = torch.prod(torch.tensor(self.shape[1:])).item()

        self.clamped = False  # see the clamp() method

        self.register_buffer("eps_1", torch.as_tensor(eps_1))
        self.register_buffer("eps_2", torch.as_tensor(eps_2))
        self.register_buffer("state", (1 / self.n_units) * torch.ones(self.shape))
        self.register_buffer(
            "reconstruction", (1 / self.n_units) * torch.ones(self.shape)
        )
        self.register_buffer("td_err", torch.zeros(self.shape))
        self.register_buffer("bu_err", torch.zeros(self.shape))

        # Optionally initialize the weight matrices
        if bu_weights is None:
            td_weights = (
                torch.rand(n_in_channels, n_out_channels, kernel_size, kernel_size)
                * 0.1
            )
            bu_weights = torch.rot90(td_weights.swapaxes(0, 1), 2, [2, 3])
        else:
            td_weights = torch.rot90(bu_weights.swapaxes(0, 1), 2, [2, 3])
        assert td_weights.shape == (
            n_in_channels,
            n_out_channels,
            kernel_size,
            kernel_size,
        )
        td_weights_flat = td_weights.view(n_in_channels, -1)
        bu_weights_flat = bu_weights.reshape(n_out_channels, -1)
        normalizer = 1 / (bu_weights_flat.sum(axis=1) + 1)
        self.register_parameter(
            "td_weights", nn.Parameter(td_weights, requires_grad=False)
        )
        self.register_buffer("td_weights_flat", td_weights_flat)
        self.register_buffer("bu_weights", bu_weights)
        self.register_buffer("bu_weights_flat", bu_weights_flat)
        self.register_buffer("normalizer", normalizer)

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        if batch_size is not None:
            self.batch_size = batch_size
            self.shape = (batch_size,) + self.shape[1:]
        device = self.state.device
        self.state = (1 / self.n_units) * torch.ones(self.shape)
        self.reconstruction = (1 / self.n_units) * torch.ones(self.shape)
        self.td_err = torch.zeros(self.shape)
        self.bu_err = torch.zeros(self.shape)
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
            update = F.conv2d(
                bu_err,
                self.bu_weights,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation,
            )
            update += self.td_err
            update *= self.normalizer[None, :, None, None]
            self.state = torch.maximum(self.eps_2, self.state) * update
        self.bu_err = self.state / torch.maximum(self.eps_1, self.reconstruction)
        self.td_err = self.reconstruction / torch.maximum(self.eps_1, self.state)
        return self.bu_err

    def backward(self, reconstruction, normalize=False):
        """Back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (bathc_size, n_units)
            The reconstruction of the state of the units in this layer that was computed
            and then back-propagated from the next layer.
        normalize : bool
            Whether to normalize the weights before computing the backwards
            reconstruction.

        Returns
        -------
        reconstruction : tensor (batch_size, n_in)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        self.reconstruction = reconstruction
        # if normalize:
        #     weights = self.normalizer * self.td_weights_flat
        # else:
        #     weights = self.td_weights_flat
        # state_flat = F.unfold(
        #     self.state,
        #     kernel_size=self.kernel_size,
        #     padding=self.padding,
        #     stride=self.stride,
        #     dilation=self.dilation,
        # )
        # reconstruction_flat = weights @ state_flat
        # reconstruction = reconstruction_flat.reshape(
        #     self.batch_size, self.n_in_channels, self.in_width, self.in_width
        # )
        if normalize:
            weights = self.normalizer * self.bu_weights
        else:
            weights = self.bu_weights
        reconstruction = F.conv_transpose2d(
            self.state,
            weights,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
        )
        return reconstruction

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
            The learning rate
        """
        state_flat = F.unfold(
            self.state,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
        )
        bu_err_flat = bu_err.view(self.batch_size, self.n_in_channels, -1) - 1
        # bu_err_flat = F.unfold(
        #     bu_err - 1,
        #     kernel_size=self.kernel_size,
        #     padding=self.padding,
        #     stride=self.stride,
        #     dilation=self.dilation,
        # )
        # state_flat = self.state.view(self.batch_size, self.n_out_channels, -1)
        delta_flat = (bu_err_flat @ state_flat.swapaxes(1, 2)).sum(axis=0)
        delta_flat /= torch.maximum(self.eps_2, state_flat.sum(axis=(0, 2)))[None, :]
        delta_flat = 1 + lr * delta_flat
        delta = delta_flat.view(
            self.n_in_channels, self.n_out_channels, self.kernel_size, self.kernel_size
        )
        # td_weights = torch.clamp(self.td_weights * delta, 0, 1)
        td_weights = self.td_weights * delta
        self.td_weights.set_(td_weights)
        self.td_weights_flat = td_weights.view(self.n_in_channels, -1)
        self.bu_weights = torch.rot90(td_weights.swapaxes(0, 1), 2, [2, 3])
        self.bu_weights_flat = self.bu_weights.reshape(self.n_out_channels, -1)
        self.normalizer = 1 / (self.bu_weights_flat.sum(axis=1) + 1)

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return (
            f"input_shape=({self.n_in_channels}, {self.in_width}, {self.in_width})"
            f"out_shape=({self.n_out_channels}, {self.out_width}, {self.out_width})"
        )


class AvgPoolLayer(nn.Module):
    """A predictive-coding layer that performs a max-pool operation.

    Parameters
    ----------
    kernel_size : int
        How large the patch is that should be max-pooled over.
    batch_size : int
        The number of inputs we compute per batch.
    """

    def __init__(self, kernel_size, batch_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.batch_size = batch_size

    def forward(self, bu_err):
        """Flatten and propagate prediction error forward.

        Parameters
        ----------
        bu_err : tensor (batch_size, channels, original_width, original_height)
            The bottom-up error computed in the previous layer.

        Returns
        -------
        bu_err : tensor (batch_size, channels, reduced_width, reduced_height)
            The bottom-up error that needs to propagate to the next layer.
        """
        bu_err = F.avg_pool2d(bu_err, self.kernel_size)
        return bu_err

    def backward(self, reconstruction):
        """Un-flatten and back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (batch_size, channels, reduced_width, reduced_height)
            The reconstruction back-propagated from the next layer.

        Returns
        -------
        reconstruction : tensor (batch_size, channels, original_width, original_height)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        return F.interpolate(reconstruction, scale_factor=self.kernel_size)

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        return

    def train_weights(self, bu_err, lr=0.01):
        """Perform a training step, updating the weights.

        Parameters
        ----------
        bu_err : tensor (batch_size, n_in)
            The bottom-up error computed in the previous layer.
        lr : float
            The learning rate
        """
        return

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"kernel_size={self.kernel_size}"


class FlattenLayer(nn.Module):
    """A predictive-coding layer that performs a flattening operation.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input.
    batch_size : int
        The number of inputs we compute per batch.
    """

    def __init__(self, input_shape, batch_size):
        super().__init__()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shape = (batch_size,) + input_shape

    def forward(self, bu_err):
        """Flatten and propagate prediction error forward.

        Parameters
        ----------
        bu_err : tensor (batch_size, input_shape)
            The bottom-up error computed in the previous layer.

        Returns
        -------
        bu_err : tensor (batch_size, -1)
            The bottom-up error that needs to propagate to the next layer.
        """
        return bu_err.view(self.batch_size, -1)

    def backward(self, reconstruction):
        """Un-flatten and back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (batch_size, -1)
            The reconstruction back-propagated from the next layer.

        Returns
        -------
        reconstruction : tensor (batch_size, input_shape)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        return reconstruction.view(self.shape)

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        if batch_size is not None:
            self.batch_size = batch_size
            self.shape = (batch_size,) + self.input_shape

    def train_weights(self, bu_err, lr=0.01):
        """Perform a training step, updating the weights.

        Parameters
        ----------
        bu_err : tensor (batch_size, n_in)
            The bottom-up error computed in the previous layer.
        lr : float
            The learning rate
        """
        return

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"input_shape={self.input_shape}"


class MiddleLayer(nn.Module):
    """A predictive-coding layer that is sandwiched between two other layers.

    This layer propagates errors onward, and back-propagates reconstructions.

    Parameters
    ----------
    n_in : int
        How many units in the previous layer, i.e. the number of incoming connections.
    n_units : int
        How many units in this layer.
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
        n_in,
        n_units,
        batch_size=1,
        bu_weights=None,
        td_weights=None,
        eps_1=0.01,
        eps_2=0.0001,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_units = n_units
        self.batch_size = batch_size
        self.shape = (batch_size, n_units)

        self.clamped = False  # see the clamp() method

        self.register_buffer("eps_1", torch.as_tensor(eps_1))
        self.register_buffer("eps_2", torch.as_tensor(eps_2))
        self.register_buffer("state", (1 / self.n_units) * torch.ones(self.shape))
        self.register_buffer(
            "reconstruction", (1 / self.n_units) * torch.ones(self.shape)
        )
        self.register_buffer("td_err", torch.zeros(self.shape))
        self.register_buffer("bu_err", torch.zeros(self.shape))

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
            self.shape = (batch_size,) + self.shape[1:]
        device = self.state.device
        self.state = (1 / self.n_units) * torch.ones(self.shape)
        self.reconstruction = (1 / self.n_units) * torch.ones(self.shape)
        self.td_err = torch.zeros(self.shape)
        self.bu_err = torch.zeros(self.shape)
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
                (bu_err @ self.bu_weights_normalized) + (self.normalizer * self.td_err)
            )
        self.bu_err = self.state / torch.maximum(self.eps_1, self.reconstruction)
        self.td_err = self.reconstruction / torch.maximum(self.eps_1, self.state)
        return self.bu_err

    def backward(self, reconstruction):
        """Back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (batch_size, n_units)
            The reconstruction of the state of the units in this layer that was computed
            and then back-propagated from the next layer.

        Returns
        -------
        reconstruction : tensor (batch_size, n_in)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        assert self.shape == reconstruction.shape
        self.reconstruction = reconstruction
        return self.state @ self.td_weights

    def clamp(self, state):
        """Clamp the units to a predefined state.

        Parameters
        ----------
        state : tensor (batch_size, n_units)
            The clamped state of the units.
        """
        assert self.shape == state.shape
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
        delta = self.state.T @ (bu_err - 1)
        delta /= torch.maximum(self.eps_2, self.state.sum(axis=0, keepdims=True)).T
        delta = 1 + lr * delta

        td_weights = torch.clamp(self.td_weights * delta, 0, 1)
        self.td_weights.set_(td_weights)
        self.bu_weights.set_(self.td_weights.T)

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
    n_units : int | tuple of int
        How many units in this layer. This can be a tuple to specify the amount of units
        in more than one dimension.
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
        self.batch_size = batch_size

        if isinstance(n_units, int):
            self.shape = (batch_size, n_units)
        elif isinstance(n_units, tuple):
            self.shape = (batch_size,) + n_units
        else:
            self.shape = (batch_size,) + tuple(n_units)
        self.n_units = torch.prod(torch.tensor(self.shape[1:])).item()

        self.clamped = False  # see the clamp() method

        self.register_buffer("state", (1 / self.n_units) * torch.ones(self.shape))
        self.register_buffer(
            "reconstruction", (1 / self.n_units) * torch.ones(self.shape)
        )
        self.register_buffer("td_err", torch.zeros(self.shape))
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
            self.shape = (batch_size,) + self.shape[1:]
        device = self.state.device
        self.state = (1 / self.n_units) * torch.ones(self.shape)
        self.reconstruction = (1 / self.n_units) * torch.ones(self.shape)
        self.td_err = torch.zeros(self.shape)
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
                assert self.shape == x.shape
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
        assert self.shape == reconstruction.shape
        self.reconstruction = reconstruction

    def clamp(self, state):
        """Clamp the units to a predefined state.

        Parameters
        ----------
        state : tensor (batch_size, n_units)
            The clamped state of the units.
        """
        assert self.shape == state.shape
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
        return f"shape={self.shape[1:]}"


class OutputLayer(nn.Module):
    """A predictive-coding layer that is at the end of the stack.

    This layer back-propagates reconstructions, but does not propagate errors forward.

    Parameters
    ----------
    n_in : int
        How many units in the previous layer, i.e. the number of incoming connections.
    n_units : int
        How many units in this layer.
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
        self.n_units = n_units
        self.n_in = n_in
        self.batch_size = batch_size
        self.shape = (batch_size, n_units)

        self.clamped = False  # see the clamp() method

        self.register_buffer("eps_2", torch.as_tensor(eps_2))
        self.register_buffer("state", (1 / self.n_units) * torch.ones(self.shape))

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
            self.shape = (batch_size,) + self.shape[1:]
        device = self.state.device
        self.state = (1 / self.n_units) * torch.ones(self.shape)
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
                bu_err @ self.bu_weights_normalized
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
        return self.state @ self.td_weights

    def clamp(self, state):
        """Clamp the units to a predefined state.

        Parameters
        ----------
        state : tensor (batch_size, n_units)
            The clamped state of the units.
        """
        assert self.shape == state.shape
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
        delta = self.state.T @ (bu_err - 1)
        delta /= torch.maximum(self.eps_2, self.state.sum(axis=0, keepdims=True)).T
        delta = 1 + lr * delta

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
