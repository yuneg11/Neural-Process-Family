from ..typing import *

from jax import numpy as jnp
from jax.scipy import stats
from flax import linen as nn

from .base import NPF
from .. import functional as F
from ..modules import (
    MLP,
)

__all__ = [
    "CNPBase",
    "CNP",
]


class CNPBase(NPF):
    """
    Base class of Conditional Neural Process
    """

    encoder:   nn.Module = None
    decoder:   nn.Module = None
    min_sigma: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.encoder is None:
            raise ValueError("encoder is not specified")
        if self.decoder is None:
            raise ValueError("decoder is not specified")

    def _encode(
        self,
        x:    Array[B, ([M],), P, X],
        y:    Array[B, ([M],), P, Y],
        mask: Array[B, P],
    ) -> Array[B, ([M],), P, R]:

        xy = jnp.concatenate((x, y), axis=-1)                                                       # [batch, (*model), point, x_dim + y_dim]
        xy, shape = F.flatten(xy, start=0, stop=-2, return_shape=True)                              # [batch x (*model), point, x_dim + y_dim]
        r_i = self.encoder(xy)                                                                      # [batch x (*model), point, r_dim]
        r_i = F.unflatten(r_i, shape, axis=0)                                                        # [batch, (*model), point, r_dim]
        return r_i                                                                                  # [batch, (*model), point, r_dim]

    def _aggregate(
        self,
        x_tar:    Array[B, ([M],), T, X],
        x_ctx:    Array[B, ([M],), C, X],
        r_i_ctx:  Array[B, ([M],), C, R],
        mask_ctx: Array[B, C],
    ) -> Array[B, ([M],), T, R]:

        r_ctx = F.masked_mean(r_i_ctx, mask_ctx, axis=-2, mask_axis=(0, -2), keepdims=True)         # [batch, (*model), 1,      r_dim]
        r_ctx = jnp.repeat(r_ctx, x_tar.shape[-2], axis=-2)                                         # [batch, (*model), target, r_dim]
        return r_ctx                                                                                # [batch, (*model), target, r_dim]

    def _decode(
        self,
        x_tar:    Array[B, ([M],), T, X],
        r_ctx:    Array[B, ([M],), T, R],
        mask_tar: Array[B, T],
    ) -> Tuple[Array[B, ([M],), T, Y], Array[B, ([M],), T, Y]]:

        query = jnp.concatenate((x_tar, r_ctx), axis=-1)                                            # [batch, (*model), target, x_dim + r_dim]

        query, shape = F.flatten(query, start=0, stop=-2, return_shape=True)                        # [batch x (*model), target, x_dim + y_dim]
        mu_log_sigma = self.decoder(query)                                                          # [batch x (*model), target, y_dim x 2]
        mu_log_sigma = F.unflatten(mu_log_sigma, shape, axis=0)                                     # [batch, (*model), target, y_dim x 2]

        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, (*model), target, y_dim] x 2
        sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(log_sigma)                      # [batch, (*model), target, y_dim]
        return mu, sigma                                                                            # [batch, (*model), target, y_dim] x 2

    @nn.compact
    def __call__(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
    ) -> Tuple[Array[B, [T], Y], Array[B, [T], Y]]:

        # Flatten
        shape_tar = x_tar.shape[1:-1]
        x_ctx    = F.flatten(x_ctx,    start=1, stop=-1)                                            # [batch, context, x_dim]
        y_ctx    = F.flatten(y_ctx,    start=1, stop=-1)                                            # [batch, context, y_dim]
        x_tar    = F.flatten(x_tar,    start=1, stop=-1)                                            # [batch, target,  x_dim]
        mask_ctx = F.flatten(mask_ctx, start=1)                                                     # [batch, context]
        mask_tar = F.flatten(mask_tar, start=1)                                                     # [batch, target]

        # Algorithm
        r_i_ctx = self._encode(x_ctx, y_ctx, mask_ctx)                                              # [batch, context, r_dim]
        r_ctx = self._aggregate(x_tar, x_ctx, r_i_ctx, mask_ctx)                                    # [batch, target,  r_dim]
        mu, sigma = self._decode(x_tar, r_ctx, mask_tar)                                            # [batch, target,  y_dim] x 2

        # Unflatten and mask
        mu    = F.masked_fill(mu,    mask_tar, fill_value=0.,   non_mask_axis=-1)                   # [batch, target, y_dim]
        sigma = F.masked_fill(sigma, mask_tar, fill_value=1e-6, non_mask_axis=-1)                   # [batch, target, y_dim]
        mu    = F.unflatten(mu,    shape_tar, axis=-2)                                              # [batch, *target, y_dim]
        sigma = F.unflatten(sigma, shape_tar, axis=-2)                                              # [batch, *target, y_dim]
        return mu, sigma                                                                            # [batch, *target, y_dim] x 2

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

        loss = -self.log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)                 # (1)
        return loss                                                                                 # (1)


class CNP:
    """
    Conditional Neural Process
    """

    def __new__(
        cls,
        y_dim: int,
        r_dim: int = 128,
        encoder_dims: Sequence[int] = (128, 128, 128, 128, 128),
        decoder_dims: Sequence[int] = (128, 128, 128),
    ):
        return CNPBase(
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim),
            decoder = MLP(hidden_features=decoder_dims, out_features=(y_dim * 2)),
        )
