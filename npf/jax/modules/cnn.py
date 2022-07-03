from ..typing import *
import numpy as np

import jax
from jax import numpy as jnp
from jax.scipy import stats
from flax import linen as nn

class ResConvBlock(nn.Module):
    out_channels: int
    dims: int = 1
    kernel_size: int = 5
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        in_channels = x.shape[-1]
        padding = self.kernel_size // 2
        conv_depthwise = nn.Conv(in_channels, (self.kernel_size,)*self.dims,
                padding=padding, use_bias=self.use_bias,
                feature_group_count=in_channels)
        conv_pointwise = nn.Conv(self.out_channels, (1,)*self.dims, use_bias=self.use_bias)
        out = conv_depthwise(jax.nn.relu(x)) + x
        out = conv_pointwise(out)
        return out

class CNN(nn.Module):
    out_channels: Sequence[int]
    dims: int = 1
    kernel_size: int = 5
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        for chan in self.out_channels:
            x = ResConvBlock(chan,
                    dims=self.dims,
                    kernel_size=self.kernel_size,
                    use_bias=self.use_bias)(x)
        return x
