from ..typing import *

import jax
from jax import numpy as jnp
from jax.scipy import stats
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


# TODO: Further refactor sampling functions


# Temporary implementation of a.ndim == 2 and p.ndim == 2
def random_choice(key, a, shape: Sequence[int] = (), replace: bool = True, p = None, axis: int = 0):
    assert a.ndim == 2 and p is not None and p.ndim == 2, "only support a.ndim == 2 and p.ndim == 2"
    _shape = shape[1:] if shape else ()
    _axis = axis - 1 if axis > 0 else 1
    vaxis = 0 if axis > 0 else 1
    body = lambda _a, _p: random.choice(key, _a, shape=_shape, p=_p, replace=replace, axis=_axis)
    result = jax.vmap(body, in_axes=(vaxis, vaxis), out_axes=vaxis)(a, p)
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
    """
    Mixins for Bootstrapping Neural Process
    """

    def _bootstrap(
        self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        mask_ctx: Array[B, C],
        num_samples: int = 1,
    ) -> Tuple[Array[B, S, C, Y], Array[B, S, C, Y]]:

        key = self.make_rng("sample")
        b_x_ctx, b_y_ctx = sample(key, x_ctx, y_ctx, mask=mask_ctx, num_samples=num_samples)        # [batch, sample, context, x_dim], [batch, sample, context, y_dim]
        s_x_ctx    = jnp.repeat(x_ctx,    num_samples, axis=0)                                      # [batch x sample, context, x_dim]
        s_mask_ctx = jnp.repeat(mask_ctx, num_samples, axis=0)                                      # [batch x sample, context]

        shape = b_x_ctx.shape[0:2]
        b_x_ctx = F.flatten(b_x_ctx, start=0, stop=2)                                               # [batch x sample, context, x_dim]
        b_y_ctx = F.flatten(b_y_ctx, start=0, stop=2)                                               # [batch x sample, context, y_dim]

        # TODO: Check the mask of b_x_ctx and b_y_ctx is s_mask_ctx
        b_r_i_ctx = self._encode(b_x_ctx, b_y_ctx, s_mask_ctx)                                      # [batch x sample, context, r_dim]
        b_r_ctx = self._aggregate(s_x_ctx, b_x_ctx, b_r_i_ctx, s_mask_ctx)                          # [batch x sample, context, r_dim]
        b_mu, b_sigma = self._decode(s_x_ctx, b_r_ctx, s_mask_ctx)                                  # [batch x sample, context, y_dim]

        b_mu    = F.unflatten(b_mu,    shape, axis=0)                                               # [batch, sample, context, y_dim]
        b_sigma = F.unflatten(b_sigma, shape, axis=0)                                               # [batch, sample, context, y_dim]
        return b_mu, b_sigma                                                                        # [batch, sample, context, y_dim] x 2

    def _residual_sample(
        self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        b_mu:     Array[B, S, C, Y],
        b_sigma:  Array[B, S, C, Y],
        mask_ctx: Array[B, C],
    ) -> Tuple[Array[B, S, C, X], Array[B, S, C, Y]]:

        key = self.make_rng("sample")
        s_x_ctx = F.repeat_axis(x_ctx, b_mu.shape[1], axis=1)                                       # [batch, sample, context, x_dim]
        s_y_ctx = F.repeat_axis(y_ctx, b_mu.shape[1], axis=1)                                       # [batch, sample, context, y_dim]

        res = sample(key, (s_y_ctx - b_mu) / b_sigma, mask=mask_ctx)                                # [batch, sample, context, y_dim]
        res -= F.masked_mean(res, mask_ctx, axis=-2, mask_axis=(0, -2), keepdims=True)              # [batch, sample, context, y_dim]

        res_x_ctx = s_x_ctx                                                                         # [batch, sample, context, x_dim]
        res_y_ctx = b_mu + b_sigma * res                                                            # [batch, sample, context, y_dim]
        return res_x_ctx, res_y_ctx                                                                 # [batch, sample, context, x_dim], [batch, sample, context, y_dim]

    def _adaptation_decode(
        self,
        x_tar:     Array[B, T, X],
        r_ctx:     Array[B, T, R],
        res_r_ctx: Array[B, S, T, R],
        mask_tar:  Array[B, T],
    ) -> Tuple[Array[B, S, T, Y], Array[B, S, T, Y]]:

        # TODO: Check adaptation layer is correctly implemented
        s_query = jnp.concatenate((x_tar, r_ctx), axis=-1)                                          # [batch, target, x_dim + r_dim]
        s_query = F.repeat_axis(s_query, res_r_ctx.shape[1], axis=1)                                # [batch, sample, target, x_dim + r_dim]
        r_query = nn.Dense(features=s_query.shape[-1])(res_r_ctx)                                   # [batch, sample, target, x_dim + r_dim]
        query = s_query + r_query                                                                   # [batch, sample, target, x_dim + r_dim]

        query = F.flatten(query, start=0, stop=2)                                                   # [batch x sample, target, x_dim + r_dim]
        mu_log_sigma = self.decoder(query)                                                          # [batch x sample, target, y_dim x 2]
        mu_log_sigma = F.unflatten(mu_log_sigma, res_r_ctx.shape[:2], axis=0)                       # [batch, sample, target, y_dim, 2]

        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, sample, target, y_dim] x 2
        sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(log_sigma)                      # [batch, sample, target, y_dim]
        return mu, sigma                                                                            # [batch, sample, target, y_dim] x 2

    @nn.compact
    def __call__(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        *,
        num_samples: int = 1,
        return_aux: bool = False,
    ) -> Union[
        Tuple[Array[B, S, [T], Y], Array[B, S, [T], Y]],
        Tuple[Array[B, S, [T], Y], Array[B, S, [T], Y], Array[B, T, R]],
    ]:

        # Flatten
        shape_tar = x_tar.shape[1:-1]
        x_ctx    = F.flatten(x_ctx,    start=1, stop=-1)                                            # [batch, context, x_dim]
        y_ctx    = F.flatten(y_ctx,    start=1, stop=-1)                                            # [batch, context, y_dim]
        x_tar    = F.flatten(x_tar,    start=1, stop=-1)                                            # [batch, target,  x_dim]
        mask_ctx = F.flatten(mask_ctx, start=1)                                                     # [batch, context]
        mask_tar = F.flatten(mask_tar, start=1)                                                     # [batch, target]

        # Algorithm
        b_mu, b_sigma = self._bootstrap(x_ctx, y_ctx, mask_ctx, num_samples)                        # [batch, sample, context, y_dim] x 2
        res_x_ctx, res_y_ctx = self._residual_sample(x_ctx, y_ctx, b_mu, b_sigma, mask_ctx)         # [batch, sample, context, x_dim], [batch, sample, context, y_dim]

        shape_res = res_x_ctx.shape[0:2]
        res_x_ctx = F.flatten(res_x_ctx, start=0, stop=2)                                           # [batch x sample, context, x_dim]
        res_y_ctx = F.flatten(res_y_ctx, start=0, stop=2)                                           # [batch x sample, context, x_dim]
        s_x_tar    = jnp.repeat(x_tar,    num_samples, axis=0)                                      # [batch x sample, target,  x_dim]
        s_mask_ctx = jnp.repeat(mask_ctx, num_samples, axis=0)                                      # [batch x sample, context]

        r_i_ctx = self._encode(x_ctx, y_ctx, mask_ctx)                                              # [batch, context, r_dim]
        r_ctx = self._aggregate(x_tar, x_ctx, r_i_ctx, mask_ctx)                                    # [batch, target,  r_dim]

        res_r_i_ctx = self._encode(res_x_ctx, res_y_ctx, s_mask_ctx)                                # [batch x sample, context, r_dim]
        res_r_ctx = self._aggregate(s_x_tar, res_x_ctx, res_r_i_ctx, s_mask_ctx)                    # [batch x sample, target,  r_dim]
        res_r_ctx = F.unflatten(res_r_ctx, shape_res, axis=0)                                       # [batch, sample, target, r_dim]

        mu, sigma = self._adaptation_decode(x_tar, r_ctx, res_r_ctx, mask_tar)                      # [batch, sample, target, y_dim] x 2

        # Unflatten and mask
        mu    = F.masked_fill(mu,    mask_tar, fill_value=0.,   non_mask_axis=(1, -1))              # [batch, sample, target, y_dim]
        sigma = F.masked_fill(sigma, mask_tar, fill_value=1e-6, non_mask_axis=(1, -1))              # [batch, sample, target, y_dim]
        mu    = F.unflatten(mu,    shape_tar, axis=-2)                                              # [batch, sample, *target, y_dim]
        sigma = F.unflatten(sigma, shape_tar, axis=-2)                                              # [batch, sample, *target, y_dim]

        if return_aux:
            return mu, sigma, r_ctx                                                                 # [batch, sample, *target, y_dim] x 2, [batch, target, r_dim]
        else:
            return mu, sigma                                                                        # [batch, sample, *target, y_dim] x 2

    def log_likelihood(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        *,
        num_samples: int = 1,
        as_mixture: bool = True,
        return_aux: bool = False,
    ) -> Union[
        Array,
        Tuple[Array, Array[B, T, R]],
    ]:

        mu, sigma, aux = \
            self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_samples=num_samples, return_aux=True) # [batch, sample, *target, y_dim] x 2, [batch, target, r_dim]

        s_y_tar = jnp.expand_dims(y_tar, axis=1)                                                    # [batch, 1,      *target, y_dim]
        log_prob = stats.norm.logpdf(s_y_tar, mu, sigma)                                            # [batch, sample, *target, y_dim]
        ll = jnp.sum(log_prob, axis=-1)                                                             # [batch, sample, *target]

        if as_mixture:
            ll = F.logmeanexp(ll, axis=1)                                                           # [batch, *target]
            ll = F.masked_mean(ll, mask_tar)                                                        # (1)
        else:
            axis = [-d for d in range(1, mask_tar.ndim)]
            ll = F.masked_mean(ll, mask_tar, axis=axis, non_mask_axis=1)                            # [batch, sample]
            ll = F.logmeanexp(ll, axis=1)                                                           # [batch]
            ll = jnp.mean(ll)                                                                       # (1)

        if return_aux:
            return ll, aux                                                                          # (1), [batch, target, r_dim]
        else:
            return ll                                                                               # (1)

    def loss(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        *,
        num_samples: int = 1,
        as_mixture: bool = True,
    ) -> Array:

        ll, r_ctx = self.log_likelihood(                                                            # (1), [batch, context, r_dim]
            x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar,
            num_samples=num_samples, as_mixture=as_mixture, return_aux=True,
        )

        x_tar    = F.flatten(x_tar,    start=1, stop=-1)                                            # [batch, target, x_dim]
        y_tar    = F.flatten(y_tar,    start=1, stop=-1)                                            # [batch, target, y_dim]
        mask_tar = F.flatten(mask_tar, start=1)                                                     # [batch, target]

        mu_base, sigma_base = self._decode(x_tar, r_ctx, mask_tar)                                  # [batch, target, y_dim] x 2

        log_prob_base = stats.norm.logpdf(y_tar, mu_base, sigma_base)                               # [batch, target, y_dim]
        ll_base = jnp.sum(log_prob_base, axis=-1)                                                   # [batch, target]
        ll_base = F.masked_mean(ll_base, mask_tar)                                                  # (1)

        loss = -(ll + ll_base)                                                                      # (1)
        return loss                                                                                 # (1)


class BNPBase(BNPMixin, CNPBase):
    """
    Base class of Bootstrapping Neural Process
    """


class AttnBNPBase(BNPMixin, AttnCNPBase):
    """
    Base class of Attentive Bootstrapping Neural Process
    """


class BNP:
    """
    Bootstrapping Neural Process
    """

    def __new__(
        cls,
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
    Attentive Bootstrapping Neural Process
    """

    def __new__(
        cls,
        y_dim: int,
        r_dim: int = 128,
        sa_heads: Optional[int] = 8,
        ca_heads: Optional[int] = 8,
        transform_qk_dims: Optional[Sequence[int]] = (128, 128, 128, 128, 128),
        encoder_dims: Sequence[int] = (128, 128, 128, 128, 128),
        decoder_dims: Sequence[int] = (128, 128, 128),
    ):

        if sa_heads is not None:
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim, last_activation=True)
            self_attention = MultiheadSelfAttention(dim_out=r_dim, num_heads=sa_heads)
        else:
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim, last_activation=False)
            self_attention = None

        if transform_qk_dims is not None:
            transform_qk = MLP(hidden_features=transform_qk_dims, out_features=r_dim, last_activation=False)
        else:
            transform_qk = None

        cross_attention = MultiheadAttention(dim_out=r_dim, num_heads=ca_heads)
        decoder = MLP(hidden_features=decoder_dims, out_features=(y_dim * 2))

        return AttnBNPBase(
            encoder=encoder,
            self_attention=self_attention,
            transform_qk=transform_qk,
            cross_attention=cross_attention,
            decoder=decoder,
        )
