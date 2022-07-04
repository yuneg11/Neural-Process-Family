from ..typing import *

import math

from jax import random
from jax import vmap
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
    "ConvNPBase",
    "ConvNP",
]

class ConvNPBase(NPF):
    """
    Base class of Convolutional Conditional Neural Process
    """

    z_dim: int
    discretizer:   Optional[nn.Module] = None
    encoder:       nn.Module = None
    cnn:           nn.Module = None
    cnn_post_z:     nn.Module = None
    decoder:        nn.Module = None
    min_sigma:     float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.encoder is None:
            raise ValueError("encoder is not specified")
        if self.cnn is None:
            raise ValueError("cnn is not specified")
        if self.cnn_post_z is None:
            raise ValueError("cnn_post_z is not specified")
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
        num_latents: int = 1,
    ) -> Tuple[Array[B, [T], Y], Array[B, [T], Y]]:

        # Discretize
        x_grid, mask_grid = self.discretizer(x_ctx, x_tar, mask_ctx, mask_tar)                      # [1, discrete, x_dim] (broadcastable to [batch, discrete, x_dim]), [discrete]

        # Encode
        h = self.encoder(x_grid, x_ctx, y_ctx, mask_ctx)                                            # [batch, discrete, y_dim + 1]
        # Convolution
        h = self.cnn(h)

        # Transform to qz parameters and sample z
        r = nn.Dense(2*self.z_dim)(h)
        # [batch_size, num_grids, z_dim]
        mu, sigma = jnp.split(nn.Dense(2*self.z_dim)(r), 2, axis=-1)
        sigma = 0.1 + 0.9*nn.sigmoid(sigma)
        rng = self.make_rng("sample")
        # [batch_size, num_latents, num_grids, z_dim]
        eps = random.normal(rng, shape=(mu.shape[0],num_latents)+mu.shape[1:])
        # [batch_size, num_latents, num_grids, z_dim]
        z = jnp.expand_dims(mu, 1) + jnp.expand_dims(sigma, 1) * eps
        # [batch_size, num_latents, num_grids, z_dim]
        # merge first two dims, [batch_size*num_latents, num_grids, z_dim]
        z = z.reshape((-1,) + z.shape[2:])
        z = self.cnn_post_z(z)
        # split first two dims, [batch_size, num_latents, num_grids, z_dim]
        z = z.reshape((-1, num_latents) + z.shape[1:])

        # Decode
        h = self.decoder(
            jnp.expand_dims(x_tar, 1),
            jnp.expand_dims(x_grid, 1),
            z,
            jnp.expand_dims(mask_grid, 1))

        mu, sigma = jnp.split(nn.Dense(2*y_ctx.shape[-1])(h), 2, axis=-1)
        sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(sigma)

        mask_tar = jnp.expand_dims(mask_tar, 1)
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
        num_latents: int = 1,
    ) -> Array:

        mu, sigma = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar,
                         num_latents=num_latents)

        # [batch, num_latents, *targets, y_dim]
        ll = stats.norm.logpdf(jnp.expand_dims(y_tar, 1), mu, sigma)
        # [batch, num_latents *targets]
        ll = jnp.sum(ll, axis=-1)
        # [batch, num_latents]
        ll = F.masked_sum(ll, jnp.expand_dims(mask_tar, 1), axis=-1, non_mask_axis=())
        # [batch]
        ll = F.logmeanexp(ll, axis=-1)
        # divide by num_tar to adjust scale
        ll = jnp.mean(ll / mask_tar.sum(-1))
        return ll

    def loss(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_latents:int = 1,
    ) -> Array:

        return -self.log_likelihood(
            x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, num_latents=num_latents
        )

class ConvNP:
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
        r_dim: int = 64,
        z_dim: int = 64
    ):
        if cnn_xl:
            raise NotImplementedError("cnn_xl is not supported yet")
        Net = CNN
        cnn_dims = (r_dim,)*2
        multiple = 1

        init_log_scale = math.log(2. / points_per_unit)

        discretizer = Discretization1d(
            minval=x_min,
            maxval=x_max,
            points_per_unit=points_per_unit,
            multiple=multiple,
            margin=x_margin,
        )

        return ConvNPBase(
                z_dim=z_dim,
                discretizer = discretizer,
                encoder  = SetConv1dEncoder(init_log_scale=init_log_scale),
                cnn = Net(dimension=1, hidden_features=cnn_dims, out_features=r_dim),
                cnn_post_z = Net(dimension=1, hidden_features=cnn_dims, out_features=r_dim),
                decoder = SetConv1dDecoder(init_log_scale=init_log_scale)
                )
