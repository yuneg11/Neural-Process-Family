from ..type import *

import abc

import numpy as np

import jax
from jax import numpy as jnp
from jax.scipy import stats
from flax import linen as nn


__all__ = [
    "Sequential",
    "MLP",
    "CNN",
    # "UNet",
]


def constant(value):
    def _constant(key, shape, dtype=jnp.float_):
        return jnp.full(shape, value, jax.dtypes.canonicalize_dtype(dtype))
    return _constant


class Sequential(nn.Module):
    layers: Sequence[nn.Module]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(nn.Module):
    hidden_features: Sequence[int]
    out_features: int
    use_bias: bool = True
    last_activation: bool = False
    kernel_init = lambda _, *args, **kwargs: nn.initializers.xavier_normal()(*args, **kwargs)  # FIXME: Temporary workaround
    bias_init   = lambda _, *args, **kwargs: constant(1e-3)(*args, **kwargs)                   # FIXME: Temporary workaround

    def _get_dense(self, features):
        return nn.Dense(
            features=features,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    @nn.compact
    def __call__(self, x):
        for features in self.hidden_features:
            x = self._get_dense(features)(x)
            x = nn.relu(x)

        x = self._get_dense(self.out_features)(x)
        if self.last_activation:
            x = nn.relu(x)

        return x


class CNN(nn.Module):
    dimension: int
    hidden_features: Sequence[int]
    out_features: int
    kernel_size: int = 5
    strides: int = 1
    use_bias: bool = True
    last_activation: bool = False
    kernel_init = lambda _, *args, **kwargs: nn.initializers.xavier_normal()(*args, **kwargs)  # FIXME: Temporary workaround
    bias_init   = lambda _, *args, **kwargs: constant(1e-3)(*args, **kwargs)                   # FIXME: Temporary workaround

    def _get_conv(self, features):
        return nn.Conv(
            features=features,
            kernel_size=(self.kernel_size,) * self.dimension,
            strides=self.strides,
            padding="SAME",
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    @nn.compact
    def __call__(self, x):
        if self.dimension not in (1, 2):
            raise ValueError(f"dimension must be 1 or 2, but got {self.dimension}")

        for features in self.hidden_features:
            x = self._get_conv(features)(x)
            x = nn.relu(x)

        x = self._get_conv(self.out_features)(x)
        if self.last_activation:
            x = nn.relu(x)

        return x


class UNet(nn.Module):
    pass



























class UNet(nn.Module):
    """Large convolutional architecture from 1d experiments in the paper.
    This is a 12-layer residual network with skip connections implemented by
    concatenation.
    Args:
        in_channels (int, optional): Number of channels on the input to
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8):
        super(UNet, self).__init__()
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = 16
        self.num_halving_layers = 6

        self.l1 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l2 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l3 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l4 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l5 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l6 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=8 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)

        self.l7 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l8 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l9 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=2 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l10 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=2 * self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l11 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l12 = nn.ConvTranspose1d(in_channels=2 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            init_layer_weights(layer)

    def forward(self, x):
        """Forward pass through the convolutional structure.
        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.
        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        h4 = self.activation(self.l4(h3))
        h5 = self.activation(self.l5(h4))
        h6 = self.activation(self.l6(h5))
        h7 = self.activation(self.l7(h6))

        h7 = pad_concat(h5, h7)
        h8 = self.activation(self.l8(h7))
        h8 = pad_concat(h4, h8)
        h9 = self.activation(self.l9(h8))
        h9 = pad_concat(h3, h9)
        h10 = self.activation(self.l10(h9))
        h10 = pad_concat(h2, h10)
        h11 = self.activation(self.l11(h10))
        h11 = pad_concat(h1, h11)
        h12 = self.activation(self.l12(h11))

        return pad_concat(x, h12)
