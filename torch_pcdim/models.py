"""Full predictive coding models."""
from collections import OrderedDict

import torch

from .layers import InputLayer, OutputLayer


class PCModel(torch.nn.Module):
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
        if not isinstance(layers, torch.nn.Sequential):
            assert all([isinstance(layer, torch.nn.Module) for layer in layers])
            names = (
                ["input"]
                + [f"hidden{i}" for i in range(1, len(layers) - 1)]
                + ["output"]
            )
            layers = torch.nn.Sequential(OrderedDict(zip(names, layers)))

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
