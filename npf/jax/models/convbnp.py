from ..type import *

import math

import jax
from jax import numpy as jnp
from jax import random
from flax import linen as nn

from .convcnp import ConvCNPBase

from .. import functional as F
from ..modules import (
    # UNet,
    CNN,
    Discretization1d,
    SetConv1dEncoder,
    SetConv1dDecoder,
)


__all__ = [
    "ConvBNPBase",
    "ConvBNP",
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


class ConvBNPBase(ConvCNPBase):
    def convcnp(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[C],
        mask_tar: Array[T],
    ) -> Tuple[Array[B, T, Y], Array[B, T, Y]]:

        # Discretize
        x_grid, mask_grid = self.discretizer(x_ctx, x_tar, mask_ctx, mask_tar)  # [1, discrete, x_dim] (broadcastable to [batch, discrete, x_dim]), [discrete]

        # Encode
        h = self.encoder(x_grid, x_ctx, y_ctx, mask_ctx)                        # [batch, discrete, y_dim + 1]

        # Convolution
        mu_log_sigma_grid = self.cnn(h)                                         # [batch, discrete, y_dim x 2]
        mu_grid, log_sigma_grid = jnp.split(mu_log_sigma_grid, 2, axis=-1)      # [batch, discrete, y_dim] x 2
        sigma_grid = 0.1 + 0.9 * nn.softplus(log_sigma_grid)                                # [batch, discrete, y_dim]

        # Decode
        mu    = self.decoder(x_tar, x_grid, mu_grid,    mask_grid)              # [batch, target, y_dim]
        sigma = self.decoder(x_tar, x_grid, sigma_grid, mask_grid)              # [batch, target, y_dim]

        mu    = F.apply_mask(mu,    mask_tar, axis=-2)                          # [batch, target, y_dim]
        sigma = F.apply_mask(sigma, mask_tar, axis=-2)                          # [batch, target, y_dim]
        return mu, sigma

    @nn.compact
    def __call__(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_samples: int = 1,
        _return_aux: bool = False,
    ) -> Tuple[Array[B, S, T, Y], Array[B, S, T, Y]]:

        # Discretize
        x_grid, mask_grid = self.discretizer(x_ctx, x_tar, mask_ctx, mask_tar)  # [1, discrete, x_dim] (broadcastable to [batch, discrete, x_dim]), [discrete]

        # Bootstrapping
        key = self.make_rng("sample")
        b_x_ctx, b_y_ctx = sample_with_replacement(key, x_ctx, y_ctx, mask=mask_ctx, num_samples=num_samples)  # [batch, sample, context, x_dim], [batch, sample, context, y_dim]
        s_x_ctx = F.repeat_axis(x_ctx, num_samples, axis=-3)                    # [batch, sample, context, x_dim]
        # b_mu, b_sigma = super().__call__(b_x_ctx, b_y_ctx, x_ctx, mask_ctx, mask_ctx)

        ### TEMP ###
        num_batch, num_context, x_dim = x_ctx.shape
        y_dim = y_ctx.shape[-1]

        b_x_ctx = b_x_ctx.reshape((num_batch * num_samples, num_context, x_dim))
        b_y_ctx = b_y_ctx.reshape((num_batch * num_samples, num_context, y_dim))
        s_x_ctx = s_x_ctx.reshape((num_batch * num_samples, num_context, x_dim))
        b_mu, b_sigma = self.convcnp(b_x_ctx, b_y_ctx, s_x_ctx, mask_ctx, mask_ctx)
        b_mu    = b_mu.reshape((num_batch, num_samples, num_context, -1))
        b_sigma = b_sigma.reshape((num_batch, num_samples, num_context, -1))
        s_x_ctx = s_x_ctx.reshape((num_batch, num_samples, num_context, -1))

        # Residual
        key = self.make_rng("sample")
        s_y_ctx = F.repeat_axis(y_ctx, num_samples, axis=-3)                    # [batch, sample, context, y_dim]
        res = sample_with_replacement(key, (s_y_ctx - b_mu) / b_sigma, mask=mask_ctx)  # [batch, sample, context, y_dim]
        res -= F.masked_mean(res, mask_ctx, axis=-2, keepdims=True)             # [batch, sample, context, y_dim]

        res_x_ctx = s_x_ctx
        res_y_ctx = b_mu + b_sigma * res

        # Encode
        h_ctx = self.encoder(x_grid, x_ctx, y_ctx, mask_ctx)                    # [batch, discrete, y_dim + 1]

        s_x_grid = F.repeat_axis(x_grid, num_samples, axis=-3)                  # [batch, sample,  target, y_dim]
        res_h_ctx = self.encoder(s_x_grid, res_x_ctx, res_y_ctx, mask_ctx)      # [batch, discrete, y_dim + 1]

        # Decode
        s_h_ctx = F.repeat_axis(h_ctx, num_samples, axis=-3)                    # [batch, sample, target, r_dim]
        h = s_h_ctx + res_h_ctx            # TODO: Check: / 2 ?
        # density, value = jnp.split(h, (1,), axis=-1)                # [batch, target, 1], [batch, target, v_dim]
        # h = jnp.concatenate((density, value / (density + 1e-8)), axis=-1)   # [batch, target, v_dim + 1]

        num_batch, _, num_target, r_dim = h.shape
        h = h.reshape((num_batch * num_samples, num_target, r_dim))
        mu_log_sigma_grid = self.cnn(h)                                         # [batch, sample, discrete, y_dim x 2]
        mu_log_sigma_grid = mu_log_sigma_grid.reshape((num_batch, num_samples, num_target, -1))
        mu_grid, log_sigma_grid = jnp.split(mu_log_sigma_grid, 2, axis=-1)      # [batch, sample, discrete, y_dim] x 2
        sigma_grid = 0.1 + 0.9 * nn.softplus(log_sigma_grid)                                # [batch, sample, discrete, y_dim]

        # Decode
        mu    = self.decoder(x_tar, x_grid, mu_grid,    mask_grid)              # [batch, sample, target, y_dim]
        sigma = self.decoder(x_tar, x_grid, sigma_grid, mask_grid)              # [batch, sample, target, y_dim]

        mu    = F.apply_mask(mu,    mask_tar, axis=-2)                          # [batch, sample, target, y_dim]
        sigma = F.apply_mask(sigma, mask_tar, axis=-2)                          # [batch, sample, target, y_dim]

        if _return_aux:
            return mu, sigma, s_h_ctx, s_x_grid, mask_grid
        else:
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

        mu, sigma, s_h_ctx, s_x_grid, mask_grid = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_samples, _return_aux=True)
                                                                                # [batch, target, y_dim] x 2, [batch, target, r_dim]

        num_batch, _, num_target, r_dim = s_h_ctx.shape
        s_h_ctx = s_h_ctx.reshape((num_batch * num_samples, num_target, r_dim))
        mu_log_sigma_grid = self.cnn(s_h_ctx)                                         # [batch, sample, discrete, y_dim x 2]
        mu_log_sigma_grid = mu_log_sigma_grid.reshape((num_batch, num_samples, num_target, -1))
        mu_grid, log_sigma_grid = jnp.split(mu_log_sigma_grid, 2, axis=-1)      # [batch, sample, discrete, y_dim] x 2
        sigma_grid = 0.1 + 0.9 * nn.softplus(log_sigma_grid)                                # [batch, sample, discrete, y_dim]

        # Decode
        s_x_tar = F.repeat_axis(x_tar, num_samples, axis=-3)                    # [batch, sample, target, y_dim]
        mu_base    = self.decoder(s_x_tar, s_x_grid, mu_grid,    mask_grid)              # [batch, sample, target, y_dim]
        sigma_base = self.decoder(s_x_tar, s_x_grid, sigma_grid, mask_grid)              # [batch, sample, target, y_dim]

        _y_tar = jnp.expand_dims(y_tar, axis=-3)                                # [..., 1, target, y_dim]
        ll = self._log_likelihood(_y_tar, mu, sigma)                            # [batch, sample, target]
        ll = jax.nn.logsumexp(ll, axis=-2) - math.log(num_samples)              # [batch, target]
        ll = F.masked_mean(ll, mask_tar, axis=-1)                               # [batch]
        ll = jnp.mean(ll)                                                       # [1]

        ll_base = self._log_likelihood(_y_tar, mu_base, sigma_base)              # [batch, target]
        ll_base = F.masked_mean(ll_base, mask_tar, axis=-1)                     # [batch]
        ll_base = jnp.mean(ll_base)                                             # [1]

        log_likelihood = ll + ll_base
        return log_likelihood


#! TODO: Add 2d model
class ConvBNP:
    """
    Convolutional Bootstrapping Neural Process
    """

    def __new__(cls,
        y_dim: int,
        x_min: float,
        x_max: float,
        cnn_dims: Optional[Sequence[int]] = None,
        cnn_xl: bool = False,
        points_per_unit: int = 64,
        x_margin: float = 0.1,
    ):
        if cnn_xl:
            raise NotImplementedError("cnn_xl is not supported yet")
            Net = UNet
            cnn_dims = cnn_dims or (8, 16, 16, 32, 32, 64)
            multiple = 2 ** len(cnn_dims)  # num_halving_layers = len(cnn_dims)
        else:
            Net = CNN
            cnn_dims = cnn_dims or (16, 32, 16)
            multiple = 1

        init_log_scale = math.log(2. / points_per_unit)

        discretizer = Discretization1d(
            minval=x_min,
            maxval=x_max,
            points_per_unit=points_per_unit,
            multiple=multiple,
            margin=x_margin,
        )

        return ConvBNPBase(
            discretizer = discretizer,
            encoder     = SetConv1dEncoder(init_log_scale=init_log_scale),
            cnn         = Net(dimension=1, hidden_features=cnn_dims, out_features=(y_dim * 2)),
            decoder     = SetConv1dDecoder(init_log_scale=init_log_scale),
        )
