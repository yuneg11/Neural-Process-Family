from ..type import *

import numpy as np

from jax import numpy as jnp
from jax import random
from flax import linen as nn

from .cnp import CNPBase

from ..modules import (
    MLP,
)


__all__ = [
    "BNP",
]


def sample_with_replacement(
    key: random.KeyArray,
    *items: Sequence[Union[Float[B, P, X], Float[B, S, P, X]]],
    mask: Float[B, P],
    num_samples: Optional[int] = None,
):
    _x = items[0]

    idx = jnp.arange(_x.shape[-2])
    p = mask[0, :].astype(jnp.bool_)

    if num_samples is not None:
        shape = (*_x.shape[:-2], num_samples, _x.shape[-2])
        new_axis = True
    else:
        shape = _x.shape[:-1]
        new_axis = False

    mask = jnp.expand_dims(mask, axis=(-3, -1))

    sampled_idx = jnp.expand_dims(random.choice(key, idx, shape=shape, p=p, replace=True), axis=-1) # [batch, samples, points, 1]
    sampled_items = []
    for item in items:
        if new_axis:
            item = jnp.expand_dims(item, axis=-3).repeat(num_samples, axis=-3)
        sampled_item = jnp.take_along_axis(item, indices=sampled_idx, axis=-2)
        sampled_item = jnp.where(mask, sampled_item, jnp.zeros_like(sampled_item))
        sampled_items.append(sampled_item)

    return sampled_items


class BNPBase(CNPBase):
    """
    Base class of Bootstrapping Neural Process

    Args:
        encoder : [batch, ctx, x_dim + y_dim]
               -> [batch, ctx, r_dim]
        decoder : [batch, tar, x_dim + r_dim]
               -> [batch, tar, y_dim * 2]
    """
    # fc_ctx: nn.Module

    # @nn.compact
    # def __call__(self,
    #     x_ctx:    Float[B, C, X],
    #     y_ctx:    Float[B, C, Y],
    #     x_tar:    Float[B, T, X],
    #     mask_ctx: Float[B, C],
    #     mask_tar: Float[B, T],
    #     num_samples: int = 1,
    # ) -> Tuple[Float[B, T, Y], Float[B, T, Y]]:

    #     key = self.make_rng("sample")
    #     bxc, byc = sample_with_replacement(key, x_ctx, y_ctx, mask=mask_ctx, num_samples=num_samples) # [batch, sample, context, x_dim], [batch, sample, context, y_dim]
    #     sxc, syc = jnp.repeat(jnp.expand_dims(x_ctx,axis=1), num_samples, 1), jnp.repeat(jnp.expand_dims(y_ctx,axis=1), num_samples, 1) # [batch, num_samples, ctx, x_dim], [batch, num_samples, ctx, y_dim]

    #     mask_bctx = jnp.repeat(mask_ctx, num_samples, 1)                    # [batch, num_samples, ctx]

    #     ctx = jnp.concatenate((bxc, byc), axis=-1)                          # [batch, num_samples, ctx, x_dim + y_dim]
    #     r_i_ctx = self.encoder(ctx)                                             # [batch, num_samples, ctx, r_dim]

    #     # Aggregate
    #     #############
    #     r_ctx = self._aggregate(r_i_ctx, bxc, sxc, mask_bctx)                # ??[batch, num_samples, tar, r_dim]
    #     #############

    #     query = jnp.concatenate((sxc, r_ctx), axis=-1)                        # [batch, num_samples, tar, x_dim + r_dim]
    #     py_res_mu_log_sigma = self.decoder(query)                             # [batch, num_samples, tar, y_dim x 2]

    #     py_res_mu, py_res_log_sigma = jnp.split(py_res_mu_log_sigma, 2, axis = -1)    # [batch, num_samples, tar, y_dim] x 2
    #     py_res_sigma = 0.1 + 0.9 * nn.softplus(py_res_log_sigma)

    #     ###############
    #     res = sample_with_replacement(key, (syc-py_res_mu)/(py_res_sigma),num_samples=None,mask=mask_bctx)
    #     ###############

    #     res = (res - jnp.mean(res, axis=-2, keepdims=True))

    #     bxc = sxc
    #     byc = py_res_mu + py_res_sigma * res

    #     ctx_base = jnp.concatenate((x_ctx, y_ctx), axis=-1)
    #     r_i_ctx_base = self.encoder(ctx_base)

    #     r_ctx_base = self._aggregate(r_i_ctx_base, x_ctx, x_tar, mask_ctx)

    #     sxt = jnp.repeat(jnp.expand_dims(x_tar,1), num_samples, 1)

    #     ctx_bs = jnp.concatenate((bxc, byc), axis=-1)
    #     r_i_ctx_bs = self.encoder(ctx_bs)

    #     r_ctx_bs = self._aggregate(r_i_ctx_bs, bxc, sxt, mask_bctx)

    #     query = jnp.concatenate((jnp.repeat(jnp.expand_dims(r_ctx_base,1), num_samples, 1), sxt, r_ctx_bs), axis=-1)


    @nn.compact
    def __call__(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
        num_samples: int = 1,
    ) -> Tuple[Float[B, T, Y], Float[B, T, Y]]:

        # Boot encode
        key = self.make_rng("sample")
        b_x_ctx, b_y_ctx = sample_with_replacement(key, x_ctx, y_ctx, mask=mask_ctx, num_samples=num_samples)  # [batch, sample, context, x_dim], [batch, sample, context, y_dim]
        b_ctx = jnp.concatenate((b_x_ctx, b_y_ctx), axis=-1)                    # [batch, sample, context, x_dim + y_dim]
        b_r_i_ctx = self.encoder(b_ctx)                                         # [batch, sample, context, r_dim]
        s_x_ctx = jnp.expand_dims(x_ctx, axis=-2).repeat(num_samples, axis=-2)  # [batch, sample, context, x_dim]
        b_r_ctx = self._aggregate(b_r_i_ctx, b_x_ctx, s_x_ctx, mask_ctx)        # [batch, sample, context, r_dim]

        # Boot decode
        b_query = jnp.concatenate((b_x_ctx, b_r_ctx), axis=-1)                  # [batch, sample, context, x_dim + r_dim]
        b_mu_log_sigma = self.decoder(b_query)                                  # [batch, sample, context, y_dim x 2]

        b_mu, b_log_sigma = jnp.split(b_mu_log_sigma, 2, axis=-1)               # [batch, sample, context, y_dim] x 2
        b_sigma = 0.1 + 0.9 * nn.softplus(b_log_sigma)

        _mask_ctx = jnp.expand_dims(mask_ctx, axis=(-3, -1))                    # [batch, 1, context, 1]
        # b_mu    = jnp.where(_mask_ctx, b_mu, 0.)
        # b_sigma = jnp.where(_mask_ctx, b_sigma, 0.)

        # Residual
        key = self.make_rng("sample")
        s_y_ctx = jnp.expand_dims(y_ctx, axis=-2).repeat(num_samples, axis=-2)  # [batch, sample, context, y_dim]
        res = sample_with_replacement(key, (s_y_ctx - b_mu) / b_sigma, mask=mask_ctx)  # [batch, sample, context, y_dim]
        res = res - (jnp.sum(res, axis=-2, keepdims=True) / jnp.sum(_mask_ctx, axis=-2, keepdims=True))

        b_y_ctx = b_mu + b_sigma * res

        # Encode
        ctx = jnp.concatenate((x_ctx, y_ctx), axis=-1)                          # [batch, context, x_dim + y_dim]
        r_i_ctx = self.encoder(ctx)                                             # [batch, context, r_dim]
        r_ctx = self._aggregate(r_i_ctx, x_ctx, x_tar, mask_ctx)                # [batch, target, r_dim]

        s_x_tar = jnp.expand_dims(x_tar, axis=-2).repeat(num_samples, axis=-2)  # [batch, sample, target, x_dim]
        b_ctx = jnp.concatenate((s_x_ctx, b_y_ctx), axis=-1)                    # [batch, sample, context, x_dim + y_dim]
        b_r_i_ctx = self.encoder(b_ctx)                                         # [batch, sample, context, r_dim]
        b_r_ctx = self._aggregate(r_i_ctx, s_x_ctx, s_x_tar, mask_ctx)          # [batch, sample, target, r_dim]

        # Decode
        s_r_ctx = jnp.expand_dims(r_ctx, axis=-2).repeat(num_samples, axis=-2)  # [batch, sample, target, r_dim]
        b_query = jnp.concatenate((b_x_ctx, b_r_ctx), axis=-1)
        r_query = nn.Dense(features=b_query.shape[-1])(s_r_ctx)
        query = b_query + r_query

        mu_log_sigma = self.decoder(query)                                  # [batch, sample, context, y_dim x 2]

        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                     # [batch, sample, context, y_dim] x 2
        sigma = 0.1 + 0.9 * nn.softplus(log_sigma)

        mask_tar = jnp.expand_dims(mask_tar, axis=-1)                           # [batch, context, 1]
        mu    = jnp.where(mask_tar, mu, 0.)
        sigma = jnp.where(mask_tar, sigma, 0.)
