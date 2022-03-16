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
    @nn.compact
    def __call__(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_samples: int = 1,
        _return_aux: bool = False,
    ) -> Tuple[Array[B, T, Y], Array[B, T, Y]]:

        num_batch = x_ctx.shape[0]

        s_x_ctx = F.repeat_axis(x_ctx, num_samples, axis=-3)
        s_y_ctx = F.repeat_axis(y_ctx, num_samples, axis=-3)
        s_x_tar = F.repeat_axis(x_tar, num_samples, axis=-3)

        # Bootstrap weights
        key = self.make_rng("sample")
        alpha = jnp.expand_dims(mask_ctx, axis=(0, 1))                          # [1, 1, context]
        w_ctx = random.dirichlet(key, alpha=alpha, shape=(num_batch, num_samples)) # [batch, samples, context]
        w_ctx = jnp.expand_dims(w_ctx, axis=-1)                                 # [batch, samples, context, 1]
        w_ctx = w_ctx * jnp.sum(mask_ctx)

        # Encode
        ctx = jnp.concatenate((s_x_ctx, s_y_ctx, w_ctx), axis=-1)
        r_i_ctx = self._encode(ctx, mask_ctx)                                   # [batch, samples, context, r_dim]
        r_i_ctx = r_i_ctx * w_ctx                                               # [batch, samples, context, r_dim]
        r_ctx = self._aggregate(r_i_ctx, s_x_ctx, s_x_tar, mask_ctx)            # [batch, samples, r_dim]

        # Decode
        query = jnp.concatenate((s_x_tar, r_ctx), axis=-1)                      # [batch, samples, target, x_dim + r_dim]
        mu, sigma = self._decode(query, mask_tar)                               # [batch, samples, target, y_dim] x 2

        if _return_aux:
            return mu, sigma, w_ctx
        else:
            return mu, sigma

    def log_likelihood(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[B, C],
        mask_tar: Array[B, T],
        num_samples: int = 1,
    ) -> Array:
        """
        Calculate log-likelihood.
        """

        mu, sigma, w_ctx = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_samples, _return_aux=True)
                                                                                # [batch, target, y_dim] x 2, [batch, target, r_dim]

        s_y_tar = jnp.expand_dims(y_tar, axis=-3)                               # [..., 1, target, y_dim]
        ll = self._log_likelihood(s_y_tar, mu, sigma)                           # [batch, sample, target]

        mask_ex_tar = (mask_tar & (~mask_ctx))
        w_ctx = w_ctx[..., 0]                                                   # [batch, samples, context]

        ll_tar = jax.nn.logsumexp(ll,         axis=-2) - math.log(num_samples)  # [batch, point]
        ll_ctx = jax.nn.logsumexp(ll * w_ctx, axis=-2) - math.log(num_samples)  # [batch, point]

        ll_tar = F.masked_sum(ll_tar, mask_ex_tar, axis=-1)                     # [batch]
        ll_ctx = F.masked_sum(ll_ctx, mask_ctx,    axis=-1)                     # [batch]
        ll = (ll_tar + ll_ctx) / jnp.sum(mask_tar)                              # [batch]
        log_likelihood = jnp.mean(ll)                                           # [1]

        return log_likelihood

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
