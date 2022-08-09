from ..typing import *

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


# class UNet(nn.Module):
#     pass
