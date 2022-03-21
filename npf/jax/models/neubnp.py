from ..type import *

import math

import numpy as np

import jax
from jax import numpy as jnp
from jax import random
from flax import linen as nn

from .cnp import CNPBase
from .attncnp import AttnCNPBase

from .. import functional as F
from ..modules import (
    MLP,
    MultiheadAttention,
    MultiheadSelfAttention,
)


__all__ = [
    "NeuBNPBase",
    "NeuBNP",
]


class NeuBNPBase(CNPBase):
    def _predict(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_samples: int = 1,
    ):

        _x_ctx,    _ = F.flatten(x_ctx,    start=1, stop=-1)                                        # [batch, context, x_dim]
        _y_ctx,    _ = F.flatten(y_ctx,    start=1, stop=-1)                                        # [batch, context, y_dim]
        _x_tar, meta = F.flatten(x_tar,    start=1, stop=-1)                                        # [batch, target,  x_dim]
        _mask_ctx, _ = F.flatten(mask_ctx, start=1)                                                 # [batch, context]
        _mask_tar, _ = F.flatten(mask_tar, start=1)                                                 # [batch, target]

        num_batch = x_ctx.shape[0]

        s_x_ctx = F.repeat_axis(_x_ctx, num_samples, axis=1)                                        # [batch, sample, context, x_dim]
        s_y_ctx = F.repeat_axis(_y_ctx, num_samples, axis=1)                                        # [batch, sample, context, y_dim]
        s_x_tar = F.repeat_axis(_x_tar, num_samples, axis=1)                                        # [batch, sample, target,  x_dim]

        # Bootstrap weights
        key = self.make_rng("sample")
        alpha = jnp.expand_dims(_mask_ctx, axis=1)                                                  # [batch, 1,      context]
        w_ctx = random.dirichlet(key, alpha=alpha, shape=(num_batch, num_samples))                  # [batch, sample, context]
        w_ctx = jnp.expand_dims(w_ctx, axis=-1)                                                     # [batch, sample, context, 1]
        w_ctx = w_ctx * jnp.expand_dims(jnp.sum(_mask_ctx, axis=-1), axis=(1, -2, -1))              # [batch, sample, context, 1]

        # Encode
        ctx = jnp.concatenate((s_x_ctx, s_y_ctx, w_ctx), axis=-1)                                   # [batch, sample, context, x_dim + y_dim + 1]
        r_i_ctx = self._encode(ctx, _mask_ctx)                                                      # [batch, sample, context, r_dim]
        r_i_ctx = r_i_ctx * w_ctx                                                                   # [batch, sample, context, r_dim]
        r_ctx = self._aggregate(r_i_ctx, s_x_ctx, s_x_tar, _mask_ctx)                               # [batch, sample, target,  r_dim]

        # Decode
        query = jnp.concatenate((s_x_tar, r_ctx), axis=-1)                                          # [batch, sample, target, x_dim + r_dim]
        mu, sigma = self._decode(query, _mask_tar)                                                  # [batch, sample, target, y_dim] x 2
        mu    = F.unflatten(mu,    meta, axis=-2)                                                   # [batch, sample, *target, y_dim]
        sigma = F.unflatten(sigma, meta, axis=-2)                                                   # [batch, sample, *target, y_dim]

        return mu, sigma, w_ctx

    @nn.compact
    def __call__(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_samples: int = 1,
    ) -> Tuple[Array[B, T, Y], Array[B, T, Y]]:

        mu, sigma, _ = self._predict(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_samples)          # [batch, sample, *target, y_dim] x 2
        return mu, sigma

    def log_likelihood(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_samples: int = 1,
    ) -> Array:
        """
        Calculate log-likelihood.
        """

        mu, sigma = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_samples)                      # [batch, sample, *target, y_dim] x 2, [batch, target, r_dim]

        s_y_tar = jnp.expand_dims(y_tar, axis=1)                                                    # [batch, 1,      *target, y_dim]
        log_likelihood = self._log_likelihood(s_y_tar, mu, sigma)                                   # [batch, sample, *target]
        log_likelihood = F.logmeanexp(log_likelihood, axis=1)                                       # [batch, *target]
        log_likelihood = F.masked_mean(log_likelihood, mask_tar)                                    # [1]

        return log_likelihood

    def loss(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_samples: int = 1,
    ) -> Array:
        """
        Calculate log-likelihood.
        """

        # assert np.array_equal(x_ctx, x_tar) and np.array_equal(y_ctx, y_tar) and mask_ctx.shape == mask_tar.shape, \
        #     "Currently, only support context and target from the same array."

        mu, sigma, w_ctx = self._predict(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_samples)      # [batch, target, y_dim] x 2, [batch, target, r_dim]

        s_y_tar = jnp.expand_dims(y_tar, axis=-3)                                                   # [..., 1, target, y_dim]
        ll = self._log_likelihood(s_y_tar, mu, sigma)                                               # [batch, sample, target]

        mask_ex_tar = (mask_tar & (~mask_ctx))
        w_ctx = w_ctx[..., 0]                                                                       # [batch, samples, context]

        ll_tar = F.logmeanexp(ll,         axis=-2)                                                  # [batch, point]
        ll_ctx = F.logmeanexp(ll * w_ctx, axis=-2)                                                  # [batch, point]

        ll_tar = F.masked_sum(ll_tar, mask_ex_tar, axis=-1)                                         # [batch]
        ll_ctx = F.masked_sum(ll_ctx, mask_ctx,    axis=-1)                                         # [batch]
        ll = (ll_tar + ll_ctx) / jnp.sum(mask_tar)                                                  # [batch]
        loss = -jnp.mean(ll)                                                                        # [1]

        return loss


class AttnNeuBNP:
    pass


class NeuBNP:
    """
    Neural Bootstrapping Neural Process
    """

    def __new__(cls,
        y_dim: int,
        r_dim: int = 128,
        encoder_dims: Sequence[int] = (128, 128, 128, 128, 128),
        decoder_dims: Sequence[int] = (128, 128, 128),
    ):
        return NeuBNPBase(
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim),
            decoder = MLP(hidden_features=decoder_dims, out_features=(y_dim * 2)),
        )
