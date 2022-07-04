from ..typing import *

import math

from jax import numpy as jnp
from jax.scipy import stats
from flax import linen as nn

from .base import NPF
from .. import functional as F
from ..modules import (
    # UNet,
    CNN,
    Discretization1d,
    SetConv1dEncoder,
    SetConv1dDecoder,
)

__all__ = [
    "ConvCNPBase",
    "ConvCNP",
]

#! TODO: Change model to support on-the-grid(discretized) data.
class ConvCNPBase(NPF):
    """
    Base class of Convolutional Conditional Neural Process
    """

    discretizer:   Optional[nn.Module] = None
    encoder:       nn.Module = None
    cnn:           nn.Module = None
    decoder:        nn.Module = None
    min_sigma:     float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.encoder is None:
            raise ValueError("encoder is not specified")
        if self.cnn is None:
            raise ValueError("cnn is not specified")
        if self.decoder is None:
            raise ValueError("decoder is not specified")

    @nn.compact
    def __call__(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
    ) -> Tuple[Array[B, [T], Y], Array[B, [T], Y]]:

        # Discretize
        x_grid, mask_grid = self.discretizer(x_ctx, x_tar, mask_ctx, mask_tar)                      # [1, discrete, x_dim] (broadcastable to [batch, discrete, x_dim]), [discrete]

        # Encode
        h = self.encoder(x_grid, x_ctx, y_ctx, mask_ctx)                                            # [batch, discrete, y_dim + 1]

        # Convolution
        r = self.cnn(h)

        # Decode
        r = self.decoder(x_tar, x_grid, r, mask_grid)
        mu, sigma = jnp.split(nn.Dense(2*y_ctx.shape[-1])(r), 2, axis=-1)
        sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(sigma)

        mu    = F.masked_fill(mu,    mask_tar, non_mask_axis=-1)                                    # [batch, target, y_dim]
        sigma = F.masked_fill(sigma, mask_tar, non_mask_axis=-1)                                    # [batch, target, y_dim]
        return mu, sigma

    def log_likelihood(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
    ) -> Array:

        mu, sigma = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar)                                   # [batch, *target, y_dim] x 2

        log_prob = stats.norm.logpdf(y_tar, mu, sigma)                                              # [batch, *target, y_dim]
        ll = jnp.sum(log_prob, axis=-1)                                                             # [batch, *target]
        ll = F.masked_mean(ll, mask_tar)                                                            # (1)
        return ll                                                                                   # (1)

    def loss(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
    ) -> Array:

        return -self.log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)

#! TODO: Add 2d model
class ConvCNP:
    """
    Convolutional Conditional Neural Process
    """

    def __new__(cls,
        y_dim: int,
        x_min: float,
        x_max: float,
        cnn_dims: Optional[Sequence[int]] = None,
        cnn_xl: bool = False,
        points_per_unit: int = 64,
        x_margin: float = 0.1,
        r_dim: int = 64
    ):
        if cnn_xl:
            raise NotImplementedError("cnn_xl is not supported yet")
        Net = CNN
        cnn_dims = cnn_dims or (r_dim,)*4
        multiple = 1

        init_log_scale = math.log(2. / points_per_unit)

        discretizer = Discretization1d(
            minval=x_min,
            maxval=x_max,
            points_per_unit=points_per_unit,
            multiple=multiple,
            margin=x_margin,
        )

        return ConvCNPBase(
            discretizer   = discretizer,
            encoder       = SetConv1dEncoder(init_log_scale=init_log_scale),
            cnn = Net(dimension=1, hidden_features=cnn_dims, out_features=r_dim),
            decoder = SetConv1dDecoder(init_log_scale=init_log_scale)
        )
