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
    "AttnBNPBase",
    "BNP",
    "AttnBNP",
]


# Temporary implementation of a.ndim == 2 and p.ndim == 2
def random_choice(key, a, shape: Sequence[int] = (), replace: bool = True, p = None, axis: int = 0):
    assert a.ndim == 2 and p is not None and p.ndim == 2, "only support a.ndim == 2 and p.ndim == 2"
    _shape = shape[1:] if shape else ()
    _axis = axis - 1 if axis > 0 else 1
    _vaxis = 0 if axis > 0 else 1
    _body = lambda _a, _p: random.choice(key, _a, shape=_shape, p=_p, replace=replace, axis=_axis)
    result = jax.vmap(_body, in_axes=(_vaxis, _vaxis), out_axes=_vaxis)(a, p)
    return result


def sample(
    key: random.KeyArray,
    *items: Sequence[Union[Array[B, P, X], Array[B, S, P, X]]],
    mask: Array[B, P],
    num_samples: Optional[int] = None,
):
    _x = items[0]
    idx = F.repeat_axis(jnp.arange(_x.shape[-2]), repeats=_x.shape[0], axis=0)

    if num_samples is not None:
        shape = (*_x.shape[:-1], num_samples)
        new_axis = True
    else:
        shape = (_x.shape[0], _x.shape[2], _x.shape[1])
        new_axis = False

    sampled_idx = random_choice(key, idx, shape=shape, p=mask, replace=True, axis=1)                # [batch, points, samples]
    sampled_idx = jnp.expand_dims(jnp.swapaxes(sampled_idx, -1, -2), axis=-1)                       # [batch, samples, points, 1]
    sampled_items = []

    for item in items:
        if new_axis:
            item = jnp.expand_dims(item, axis=-3)
        sampled_item = jnp.take_along_axis(item, indices=sampled_idx, axis=-2)
        sampled_item = F.masked_fill(sampled_item, mask, mask_axis=(0, -2))
        sampled_items.append(sampled_item)

    if len(sampled_items) == 1:
        return sampled_items[0]
    else:
        return sampled_items


class BNPMixin(nn.Module):

    @nn.compact
    def __call__(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_samples: int = 1,
        _return_aux: bool = False,
    ) -> Tuple[Array[B, S, [T], Y], Array[B, S, [T], Y]]:

        _x_ctx,    _ = F.flatten(x_ctx,    start=1, stop=-1)                                        # [batch, context, x_dim]
        _y_ctx,    _ = F.flatten(y_ctx,    start=1, stop=-1)                                        # [batch, context, y_dim]
        _x_tar, meta = F.flatten(x_tar,    start=1, stop=-1)                                        # [batch, target,  x_dim]
        _mask_ctx, _ = F.flatten(mask_ctx, start=1)                                                 # [batch, context]
        _mask_tar, _ = F.flatten(mask_tar, start=1)                                                 # [batch, target]

        # Bootstrapping
        key = self.make_rng("sample")
        b_x_ctx, b_y_ctx = sample(key, _x_ctx, _y_ctx, mask=_mask_ctx, num_samples=num_samples)     # [batch, sample, context, x_dim], [batch, sample, context, y_dim]
        s_x_ctx = F.repeat_axis(_x_ctx, num_samples, axis=1)                                        # [batch, sample, context, x_dim]

        b_ctx = jnp.concatenate((b_x_ctx, b_y_ctx), axis=-1)                                        # [batch, sample, context, x_dim + y_dim]
        b_r_i_ctx = self._encode(b_ctx, _mask_ctx)                                                  # [batch, sample, context, r_dim]
        b_r_ctx = self._aggregate(b_r_i_ctx, b_x_ctx, s_x_ctx, _mask_ctx)                           # [batch, sample, context, r_dim]

        b_query = jnp.concatenate((b_x_ctx, b_r_ctx), axis=-1)                                      # [batch, sample, context, x_dim + r_dim]
        b_mu, b_sigma = self._decode(b_query, _mask_ctx)                                            # [batch, sample, context, y_dim]

        # Residual
        key = self.make_rng("sample")
        s_y_ctx = F.repeat_axis(_y_ctx, num_samples, axis=1)                                        # [batch, sample, context, y_dim]
        res = sample(key, (s_y_ctx - b_mu) / b_sigma, mask=_mask_ctx)                               # [batch, sample, context, y_dim]
        res -= F.masked_mean(res, _mask_ctx, axis=-2, mask_axis=(0, -2), keepdims=True)             # [batch, sample, context, y_dim]

        res_x_ctx = s_x_ctx                                                                         # [batch, sample, context, x_dim]
        res_y_ctx = b_mu + b_sigma * res                                                            # [batch, sample, context, y_dim]

        # Encode
        _ctx = jnp.concatenate((_x_ctx, _y_ctx), axis=-1)                                           # [batch, context, x_dim + y_dim]
        r_i_ctx = self._encode(_ctx, _mask_ctx)                                                     # [batch, context, r_dim]
        r_ctx = self._aggregate(r_i_ctx, _x_ctx, _x_tar, _mask_ctx)                                 # [batch, target,  r_dim]

        s_x_tar = F.repeat_axis(_x_tar, num_samples, axis=1)                                        # [batch, sample, target,  y_dim]
        res_ctx = jnp.concatenate((res_x_ctx, res_y_ctx), axis=-1)                                  # [batch, sample, context, x_dim + y_dim]
        res_r_i_ctx = self._encode(res_ctx, _mask_ctx)                                              # [batch, sample, context, r_dim]
        res_r_ctx = self._aggregate(res_r_i_ctx, res_x_ctx, s_x_tar, _mask_ctx)                     # [batch, sample, target,  r_dim]

        # Decode
        s_r_ctx = F.repeat_axis(r_ctx, num_samples, axis=1)                                         # [batch, sample, target, r_dim]
        b_query = jnp.concatenate((s_x_tar, s_r_ctx), axis=-1)                                      # [batch, sample, target, x_dim + r_dim]
        r_query = nn.Dense(features=b_query.shape[-1])(res_r_ctx)                                   # [batch, sample, target, x_dim + r_dim]
        query = b_query + r_query                                                                   # [batch, sample, target, x_dim + r_dim]

        mu, sigma = self._decode(query, _mask_tar)                                                  # [batch, sample,  target, y_dim] x 2
        mu    = F.unflatten(mu,    meta, axis=-2)                                                   # [batch, sample, *target, y_dim]
        sigma = F.unflatten(sigma, meta, axis=-2)                                                   # [batch, sample, *target, y_dim]

        if _return_aux:
            return mu, sigma, r_ctx
        else:
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
        **kwargs,
    ) -> Array:
        """
        Calculate loss.
        """

        _x_tar,    _ = F.flatten(x_tar,    start=1, stop=-1)                                        # [batch, target, x_dim]
        _y_tar,    _ = F.flatten(y_tar,    start=1, stop=-1)                                        # [batch, target, y_dim]
        _mask_tar, _ = F.flatten(mask_tar, start=1)                                                 # [batch, target]

        mu, sigma, r_ctx = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_samples, _return_aux=True) # [batch, sample, *target, y_dim] x 2, [batch, target, r_dim]

        s_y_tar = jnp.expand_dims(y_tar, axis=1)                                                    # [batch, 1,      *target, y_dim]
        ll = self._log_likelihood(s_y_tar, mu, sigma)                                               # [batch, sample, *target]
        ll = F.logmeanexp(ll, axis=1)                                                               # [batch, *target]
        ll = F.masked_mean(ll, mask_tar)                                                            # [1]

        base_query = jnp.concatenate((_x_tar, r_ctx), axis=-1)                                      # [batch, target, x_dim + r_dim]
        mu_base, sigma_base = self._decode(base_query, _mask_tar)                                   # [batch, target, y_dim] x 2

        ll_base = self._log_likelihood(_y_tar, mu_base, sigma_base)                                 # [batch, target]
        ll_base = F.masked_mean(ll_base, _mask_tar)                                                 # [1]

        loss = -(ll + ll_base)
        return loss


class BNPBase(BNPMixin, CNPBase):
    f"""
    Base class of Bootstrapping Neural Process

    Args:
        encoder: [batch, ctx, x_dim + y_dim] -> [batch, ctx, r_dim]
        decoder: [batch, tar, x_dim + r_dim] -> [batch, tar, y_dim * 2]
    """


class AttnBNPBase(BNPMixin, AttnCNPBase):
    f"""
    Base class of Attentive Bootstrapping Neural Process

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


class AttnBNP:
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

        return AttnBNPBase(
            encoder=encoder,
            self_attention=self_attention,
            cross_attention=cross_attention,
            decoder=decoder,
        )
