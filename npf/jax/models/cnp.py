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
        ctx:  Array[B, ..., P, V],
        mask: Array[B, P],
    ) -> Array[B, ..., P, R]:

        r_i = self.encoder(ctx)                                                                     # [batch, ..., point, r_dim]
        return r_i

    def _aggregate(self,
        r_i_ctx:  Array[B, ..., C, R],
        x_ctx:    Array[B, ..., C, X],
        x_tar:    Array[B, ..., T, X],
        mask_ctx: Array[B, C],
    ) -> Array[B, ..., T, R]:

        r_ctx = F.masked_mean(r_i_ctx, mask_ctx, axis=-2, mask_axis=(0, -2), keepdims=True)         # [batch, ..., context, r_dim]
        r_ctx = r_ctx.repeat(x_tar.shape[-2], axis=-2)                                              # [batch, ..., target,  r_dim]
        return r_ctx

    def _decode(self,
        query: Array[B, ..., T, X + R],
        mask_tar: Array[B, T],
    ) -> Tuple[Array[B, ..., T, Y], Array[B, ..., T, Y]]:

        mu_log_sigma = self.decoder(query)                                                          # [batch, ..., target, y_dim x 2]
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, ..., target, y_dim] x 2
        sigma = 0.1 + 0.9 * nn.softplus(log_sigma)

        mu    = F.masked_fill(mu,    mask_tar, mask_axis=(0, -2))                                    # [batch, ..., target, y_dim]
        sigma = F.masked_fill(sigma, mask_tar, mask_axis=(0, -2))                                    # [batch, ..., target, y_dim]
        return mu, sigma

    @nn.compact
    def __call__(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
    ) -> Tuple[Array[B, [T], Y], Array[B, [T], Y]]:

        _x_ctx,    _ = F.flatten(x_ctx,    start=1, stop=-1)                                        # [batch, context, x_dim]
        _y_ctx,    _ = F.flatten(y_ctx,    start=1, stop=-1)                                        # [batch, context, y_dim]
        _x_tar, meta = F.flatten(x_tar,    start=1, stop=-1)                                        # [batch, target,  x_dim]
        _mask_ctx, _ = F.flatten(mask_ctx, start=1)                                                 # [batch, context]
        _mask_tar, _ = F.flatten(mask_tar, start=1)                                                 # [batch, target]

        # Encode
        ctx = jnp.concatenate((_x_ctx, _y_ctx), axis=-1)                                            # [batch, context, x_dim + y_dim]
        r_i_ctx = self._encode(ctx, _mask_ctx)                                                      # [batch, context, r_dim]
        r_ctx = self._aggregate(r_i_ctx, _x_ctx, _x_tar, _mask_ctx)                                 # [batch, target,  r_dim]

        # Decode
        query = jnp.concatenate((_x_tar, r_ctx), axis=-1)                                           # [batch, target, x_dim + r_dim]
        mu, sigma = self._decode(query, _mask_tar)                                                  # [batch,  target, y_dim] x 2
        mu    = F.unflatten(mu,    meta, axis=-2)                                                   # [batch, *target, y_dim]
        sigma = F.unflatten(sigma, meta, axis=-2)                                                   # [batch, *target, y_dim]

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
