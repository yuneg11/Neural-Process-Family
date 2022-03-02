from ..type import *

import numpy as np

from jax import numpy as jnp
from flax import linen as nn

from .base import ConditionalNPF

from ..modules import (
    MLP,
)


__all__ = [
    "CNPBase",
    "CNP",
]


class CNPBase(ConditionalNPF):
    """
    Base class of Conditional Neural Process

    Args:
        encoder : [batch, ctx, x_dim + y_dim]
               -> [batch, ctx, r_dim]
        decoder : [batch, tar, x_dim + r_dim]
               -> [batch, tar, y_dim * 2]
    """

    encoder: nn.Module
    decoder: nn.Module

    def _aggregate(self,
        r_i_ctx:  Float[B, C, R],
        x_ctx:    Float[B, C, X],
        x_tar:    Float[B, T, X],
        mask_ctx: Float[B, C],
    ) -> Float[B, T, R]:

        mask_ctx = jnp.expand_dims(mask_ctx, axis=-1)                           # [batch, ctx, 1]
        r_i_ctx = jnp.where(mask_ctx, r_i_ctx, 0.)                              # [batch, ctx, r_dim]
        r_ctx = jnp.sum(r_i_ctx,  axis=1, keepdims=True) \
              / jnp.sum(mask_ctx, axis=1, keepdims=True)                        # [batch, 1, r_dim]
        r_ctx = r_ctx.repeat(x_tar.shape[1], axis=1)                            # [batch, tar, r_dim]
        return r_ctx

    @nn.compact
    def __call__(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
    ) -> Tuple[Float[B, T, Y], Float[B, T, Y]]:

        # Encode
        ctx = jnp.concatenate((x_ctx, y_ctx), axis=-1)                          # [batch, ctx, x_dim + y_dim]
        r_i_ctx = self.encoder(ctx)                                             # [batch, ctx, r_dim]

        # Aggregate
        r_ctx = self._aggregate(r_i_ctx, x_ctx, x_tar, mask_ctx)                # [batch, tar, r_dim]

        # Decode
        query = jnp.concatenate((x_tar, r_ctx), axis=-1)                        # [batch, tar, x_dim + r_dim]
        mu_log_sigma = self.decoder(query)                                      # [batch, tar, y_dim x 2]

        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                     # [batch, tar, y_dim] x 2
        sigma = 0.1 + 0.9 * nn.softplus(log_sigma)

        mask_tar = jnp.expand_dims(mask_tar, axis=-1)                           # [batch, ctx, 1]
        mu    = jnp.where(mask_tar, mu, 0.)
        sigma = jnp.where(mask_tar, sigma, 0.)

        return mu, sigma


class CNP:
    """
    Conditional Neural Process
    """

    def __new__(cls,
        y_dim: int,
        r_dim: int = 128,
        encoder_dims: Sequence[int] = (128, 128, 128, 128, 128),
        decoder_dims: Sequence[int] = (128, 128, 128),
    ):
        return CNPBase(
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim),
            decoder = MLP(hidden_features=decoder_dims, out_features=(y_dim * 2)),
        )
