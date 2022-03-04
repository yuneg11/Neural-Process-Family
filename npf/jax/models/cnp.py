from ..type import *

from jax import numpy as jnp
from flax import linen as nn

from .base import ConditionalNPF
from .. import functional as F
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
        encoder: [..., context, x_dim + y_dim] -> [..., context, r_dim]
        decoder: [...,  target, x_dim + r_dim] -> [...,  target, y_dim * 2]
    """

    encoder: nn.Module = None
    decoder: nn.Module = None

    def __post_init__(self):
        super().__post_init__()
        if self.encoder is None:
            raise ValueError("encoder is not specified")
        if self.decoder is None:
            raise ValueError("decoder is not specified")

    def _encode(self,
        x:    Array[..., P, X],
        y:    Array[..., P, Y],
        mask: Array[P],
    ) -> Array[..., P, R]:

        xy = jnp.concatenate((x, y), axis=-1)                                   # [..., point, x_dim + y_dim]
        r_i = self.encoder(xy)                                                  # [..., point, r_dim]
        return r_i

    def _aggregate(self,
        r_i_ctx:  Array[..., C, R],
        x_ctx:    Array[..., C, X],
        x_tar:    Array[..., T, X],
        mask_ctx: Array[C],
    ) -> Array[..., T, R]:

        r_ctx = F.masked_mean(r_i_ctx, mask=mask_ctx, axis=-2, keepdims=True)   # [..., 1, r_dim]
        r_ctx = r_ctx.repeat(x_tar.shape[-2], axis=-2)                          # [..., target, r_dim]
        return r_ctx

    def _decode(self,
        query: Array[..., T, X + R],
        mask_tar: Array[T],
    ) -> Tuple[Array[..., T, Y], Array[..., T, Y]]:

        mu_log_sigma = self.decoder(query)                                      # [..., target, y_dim x 2]
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                     # [..., target, y_dim] x 2
        sigma = 0.1 + 0.9 * nn.softplus(log_sigma)

        mu    = F.apply_mask(mu,    mask_tar, axis=-2)                          # [..., target, y_dim]
        sigma = F.apply_mask(sigma, mask_tar, axis=-2)                          # [..., target, y_dim]
        return mu, sigma

    @nn.compact
    def __call__(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[C],
        mask_tar: Array[T],
    ) -> Tuple[Array[B, T, Y], Array[B, T, Y]]:

        r_i_ctx = self._encode(x_ctx, y_ctx, mask_ctx)                          # [batch, context, r_dim]
        r_ctx = self._aggregate(r_i_ctx, x_ctx, x_tar, mask_ctx)                # [batch, target, r_dim]
        query = jnp.concatenate((x_tar, r_ctx), axis=-1)                        # [batch, target, x_dim + r_dim]
        mu, sigma = self._decode(query, mask_tar)                               # [batch, target, y_dim] x 2
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
