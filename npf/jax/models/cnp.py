from ..typing import *

from jax import numpy as jnp
from flax import linen as nn

from .base import NPF
from .. import functional as F
from ..data import NPData
from ..utils import npf_io, MultivariateNormalDiag
from ..modules import MLP

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

        xy = jnp.concatenate((x, y), axis=-1)                                                       # [batch,  (*model), point, x_dim + y_dim]
        xy, shape = F.flatten(xy, start=0, stop=-2, return_shape=True)                              # [batch x (*model), point, x_dim + y_dim]
        r_i = self.encoder(xy)                                                                      # [batch x (*model), point, r_dim]
        r_i = F.unflatten(r_i, shape, axis=0)                                                       # [batch,  (*model), point, r_dim]
        return r_i                                                                                  # [batch,  (*model), point, r_dim]

    def _aggregate(
        self,
        x:        Array[B, ([M],), P, X],
        x_ctx:    Array[B, ([M],), C, X],
        r_i_ctx:  Array[B, ([M],), C, R],
        mask_ctx: Array[B, C],
    ) -> Array[B, ([M],), P, R]:

        r_ctx = F.masked_mean(r_i_ctx, mask_ctx, axis=-2, mask_axis=(0, -2), keepdims=True)         # [batch, (*model), 1,     r_dim]
        r_ctx = jnp.repeat(r_ctx, x.shape[-2], axis=-2)                                             # [batch, (*model), point, r_dim]
        return r_ctx                                                                                # [batch, (*model), point, r_dim]

    def _decode(
        self,
        query: Array[B, ([M],), P, X + R],
        mask:  Array[B, P],
    ) -> Tuple[Array[B, ([M],), P, Y], Array[B, ([M],), P, Y]]:

        query, shape = F.flatten(query, start=0, stop=-2, return_shape=True)                        # [batch x (*model), point, x_dim + r_dim]
        mu_log_sigma = self.decoder(query)                                                          # [batch x (*model), point, y_dim x 2]
        mu_log_sigma = F.unflatten(mu_log_sigma, shape, axis=0)                                     # [batch,  (*model), point, y_dim x 2]

        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, (*model), point, y_dim] x 2
        sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(log_sigma)                      # [batch, (*model), point, y_dim]
        return mu, sigma                                                                            # [batch, (*model), point, y_dim] x 2

    @nn.compact
    @npf_io(flatten=True)
    def __call__(
        self,
        data: NPData,
    ) -> Tuple[Array[B, P, Y], Array[B, P, Y]]:

        r_i_ctx = self._encode(data.x_ctx, data.y_ctx, data.mask_ctx)                               # [batch, context, r_dim]
        r_ctx = self._aggregate(data.x_tar, data.x_ctx, r_i_ctx, data.mask_ctx)                     # [batch, point,   r_dim]
        query = jnp.concatenate((data.x, r_ctx), axis=-1)                                           # [batch, point,   x_dim + r_dim]
        mu, sigma = self._decode(query, data.mask)                                                  # [batch, point,   y_dim] x 2

        # Mask
        mu    = F.masked_fill(mu,    data.mask, fill_value=0., non_mask_axis=-1)                    # [batch, point, y_dim]
        sigma = F.masked_fill(sigma, data.mask, fill_value=0., non_mask_axis=-1)                    # [batch, point, y_dim]
        return mu, sigma                                                                            # [batch, point, y_dim] x 2

    @npf_io(flatten_input=True)
    def log_likelihood(
        self,
        data: NPData,
        *,
        split_set: bool = False,
    ) -> Array:

        mu, sigma = self(data, skip_io=True)                                                        # [batch, point, y_dim] x 2

        log_prob = MultivariateNormalDiag(mu, sigma).log_prob(data.y)                               # [batch, point]
        ll = F.masked_mean(log_prob, data.mask, axis=-1)                                            # [batch]
        ll = jnp.mean(ll)                                                                           # (1)

        if split_set:
            ll_ctx = F.masked_mean(log_prob, data.mask_ctx, axis=-1)                                # [batch]
            ll_tar = F.masked_mean(log_prob, data.mask_tar, axis=-1)                                # [batch]
            ll_ctx = jnp.mean(ll_ctx)                                                               # (1)
            ll_tar = jnp.mean(ll_tar)                                                               # (1)

            return ll, ll_ctx, ll_tar                                                               # (1) x 3
        else:
            return ll                                                                               # (1)

    @npf_io(flatten_input=True)
    def loss(
        self,
        data: NPData,
    ) -> Array:

        loss = -self.log_likelihood(data, skip_io=True)                                             # (1)
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
        min_sigma: float = 0.1,
    ):
        return CNPBase(
            encoder=MLP(hidden_features=encoder_dims, out_features=r_dim),
            decoder=MLP(hidden_features=decoder_dims, out_features=(y_dim * 2)),
            min_sigma=min_sigma,
        )
