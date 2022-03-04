from ..type import *

import math

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
    "BNPBase",
    "BANPBase",
    "BNP",
    "BANP",
]


def sample_with_replacement(
    key: random.KeyArray,
    *items: Sequence[Union[Array[B, P, X], Array[B, S, P, X]]],
    mask: Array[B, P],
    num_samples: Optional[int] = None,
):
    _x = items[0]
    idx = jnp.arange(_x.shape[-2])

    if num_samples is not None:
        shape = (*_x.shape[:-2], num_samples, _x.shape[-2])
        new_axis = True
    else:
        shape = _x.shape[:-1]
        new_axis = False

    sampled_idx = jnp.expand_dims(random.choice(key, idx, shape=shape, p=mask, replace=True), axis=-1) # [batch, samples, points, 1]
    sampled_items = []
    for item in items:
        if new_axis:
            item = jnp.expand_dims(item, axis=-3)
        sampled_item = jnp.take_along_axis(item, indices=sampled_idx, axis=-2)
        sampled_item = F.apply_mask(sampled_item, mask, axis=-2)
        sampled_items.append(sampled_item)

    if len(sampled_items) == 1:
        return sampled_items[0]
    else:
        return sampled_items


class BNPMeta:
    def __class_getitem__(cls, item: str):
        if item == "CNPBase":
            return CNPBase
        elif item == "AttnCNPBase":
            return AttnCNPBase
        else:
            raise ValueError(f"Unknown class: {item}")

    def _predict(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_samples: int = 1,
    ) -> Tuple[Array[B, S, T, Y], Array[B, S, T, Y], Array[B, T, R]]:

        # Bootstrapping
        key = self.make_rng("sample")
        b_x_ctx, b_y_ctx = sample_with_replacement(key, x_ctx, y_ctx, mask=mask_ctx, num_samples=num_samples)  # [batch, sample, context, x_dim], [batch, sample, context, y_dim]
        s_x_ctx = F.repeat_axis(x_ctx, num_samples, axis=-3)                    # [batch, sample, context, x_dim]

        b_r_i_ctx = self._encode(b_x_ctx, b_y_ctx, mask_ctx)                    # [batch, sample, context, r_dim]
        b_r_ctx = self._aggregate(b_r_i_ctx, b_x_ctx, s_x_ctx, mask_ctx)        # [batch, sample, context, r_dim]

        b_query = jnp.concatenate((b_x_ctx, b_r_ctx), axis=-1)                  # [batch, sample, context, x_dim + r_dim]
        b_mu, b_sigma = self._decode(b_query, mask_ctx)                         # [batch, sample, context, y_dim]

        # Residual
        key = self.make_rng("sample")
        s_y_ctx = F.repeat_axis(y_ctx, num_samples, axis=-3)                    # [batch, sample, context, y_dim]
        res = sample_with_replacement(key, (s_y_ctx - b_mu) / b_sigma, mask=mask_ctx)  # [batch, sample, context, y_dim]
        res -= F.masked_mean(res, mask_ctx, axis=-2, keepdims=True)             # [batch, sample, context, y_dim]

        res_x_ctx = s_x_ctx
        res_y_ctx = b_mu + b_sigma * res

        # Encode
        r_i_ctx = self._encode(x_ctx, y_ctx, mask_ctx)
        r_ctx = self._aggregate(r_i_ctx, x_ctx, x_tar, mask_ctx)                # [batch, target, r_dim]

        s_x_tar = F.repeat_axis(x_tar, num_samples, axis=-3)                    # [batch, sample,  target, y_dim]
        res_r_i_ctx = self._encode(res_x_ctx, res_y_ctx, mask_ctx)              # [batch, sample, context, r_dim]
        res_r_ctx = self._aggregate(res_r_i_ctx, res_x_ctx, s_x_tar, mask_ctx)  # [batch, sample,  target, r_dim]

        # Decode
        s_r_ctx = F.repeat_axis(r_ctx, num_samples, axis=-3)                    # [batch, sample, target, r_dim]
        b_query = jnp.concatenate((s_x_tar, s_r_ctx), axis=-1)
        r_query = nn.Dense(features=b_query.shape[-1])(res_r_ctx)
        query = b_query + r_query

        mu, sigma = self._decode(query, mask_tar)

        return mu, sigma, r_ctx

    @nn.compact
    def __call__(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_samples: int = 1,
    ) -> Tuple[Array[B, S, T, Y], Array[B, S, T, Y]]:

        mu, sigma, _ = self._predict(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_samples)
        return mu, sigma

    # Likelihood

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

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]
            num_samples: int

        Returns:
            log_likelihood: float
        """

        mu, sigma, r_ctx = self._predict(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_samples)  # [batch, target, y_dim] x 2, [batch, target, r_dim]
        base_query = jnp.concatenate((x_tar, r_ctx), axis=-1)                   # [batch, target, x_dim + r_dim]
        mu_base, sigma_base = self._decode(base_query, mask_tar)                # [batch, target, y_dim] x 2

        _y_tar = jnp.expand_dims(y_tar, axis=-3)                                # [..., 1, target, y_dim]
        ll = self._log_likelihood(_y_tar, mu, sigma)                            # [batch, sample, target]
        ll = jax.nn.logsumexp(ll, axis=-2) - math.log(num_samples)              # [batch, target]
        ll = F.masked_mean(ll, mask_tar, axis=-1)                               # [batch]
        ll = jnp.mean(ll)                                                       # [1]

        ll_base = self._log_likelihood(y_tar, mu_base, sigma_base)              # [batch, target]
        ll_base = F.masked_mean(ll_base, mask_tar, axis=-1)                     # [batch]
        ll_base = jnp.mean(ll_base)                                             # [1]

        log_likelihood = ll + ll_base
        return log_likelihood

    def loss(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_samples: int = 1,
    ) -> Array:
        """
        Calculate loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]
            num_samples: int

        Returns:
            loss: float
        """

        loss = -self.log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, num_samples=num_samples)
        return loss


class BNPBase(BNPMeta["CNPBase"]):
    f"""
    Base class of Bootstrapping Neural Process

    Args:
        encoder: [batch, ctx, x_dim + y_dim] -> [batch, ctx, r_dim]
        decoder: [batch, tar, x_dim + r_dim] -> [batch, tar, y_dim * 2]
    """


class BANPBase(BNPMeta["AttnCNPBase"]):
    f"""
    Base class of Bootstrapping Attentive Neural Process

    Args:
        encoder: [batch, ctx, x_dim + y_dim] -> [batch, ctx, r_dim]
        decoder: [batch, tar, x_dim + r_dim] -> [batch, tar, y_dim * 2]
    """



class BNP:
    """
    Bootstrapping Neural Process
    """

    def __new__(cls,
        y_dim: int,
        r_dim: int = 128,
        encoder_dims: Sequence[int] = (128, 128, 128, 128, 128),
        decoder_dims: Sequence[int] = (128, 128, 128),
    ):
        return BNPBase(
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim),
            decoder = MLP(hidden_features=decoder_dims, out_features=(y_dim * 2)),
        )


class BANP:
    """
    Bootstrapping Attentive Neural Process
    """

    def __new__(cls,
        y_dim: int,
        r_dim: int = 128,
        sa_heads: Optional[int] = 8,
        ca_heads: Optional[int] = 8,
        encoder_dims: Sequence[int] = (128, 128, 128, 128, 128),
        decoder_dims: Sequence[int] = (128, 128, 128),
    ):

        if sa_heads is not None:
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim, last_activation=True)
            self_attention = MultiheadSelfAttention(dim_out=r_dim, num_heads=sa_heads)
        else:
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim, last_activation=False)
            self_attention = None

        cross_attention = MultiheadAttention(dim_out=r_dim, num_heads=ca_heads)
        decoder = MLP(hidden_features=decoder_dims, out_features=(y_dim * 2))

        return BANPBase(
            encoder=encoder,
            self_attention=self_attention,
            cross_attention=cross_attention,
            decoder=decoder,
        )
