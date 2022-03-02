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
    # "UNet",
    # "SimpleConvNet",
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

    @staticmethod
    def _get_dense(features, use_bias):
        return nn.Dense(
            features=features,
            use_bias=use_bias,
            kernel_init=nn.initializers.xavier_normal(),
            bias_init=constant(1e-3),
        )

    @nn.compact
    def __call__(self, x):
        for features in self.hidden_features:
            x = self._get_dense(features, self.use_bias)(x)
            x = nn.relu(x)

        x = self._get_dense(self.out_features, self.use_bias)(x)
        if self.last_activation:
            x = nn.relu(x)

        return x
