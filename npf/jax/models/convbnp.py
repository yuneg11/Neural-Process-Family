from ..typing import *

import math

from jax import numpy as jnp
from jax.scipy import stats
from flax import linen as nn

from .convcnp import ConvCNPBase
from .bnp import sample
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


#! TODO: Change model to support on-the-grid(discretized) data.
class ConvBNPBase(ConvCNPBase):
    """
    Base class of Convolutional Bootstrapping Neural Process
    """

    def _bootstrap(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        mask_ctx: Array[B, [C]],
        num_samples: int = 1,
    ) -> Tuple[Array[B, S, C, Y], Array[B, S, C, Y]]:

        key = self.make_rng("sample")
        b_x_ctx, b_y_ctx = sample(key, x_ctx, y_ctx, mask=mask_ctx, num_samples=num_samples)        # [batch, sample, context, x_dim], [batch, sample, context, y_dim]
        s_x_ctx    = jnp.repeat(x_ctx,    num_samples, axis=0)                                      # [batch x sample, context, x_dim]
        s_mask_ctx = jnp.repeat(mask_ctx, num_samples, axis=0)                                      # [batch x sample, context]

        shape = b_x_ctx.shape[0:2]
        b_x_ctx = F.flatten(b_x_ctx, start=0, stop=2)                                               # [batch x sample, context, x_dim]
        b_y_ctx = F.flatten(b_y_ctx, start=0, stop=2)                                               # [batch x sample, context, y_dim]

        # Discretize
        x_grid, mask_grid = self.discretizer(b_x_ctx, s_x_ctx, s_mask_ctx, s_mask_ctx)              # [1, grid, x_dim] (broadcastable to [batch, grid, x_dim]), [discrete]

        # Encode
        b_h = self.encoder(x_grid, b_x_ctx, b_y_ctx, s_mask_ctx)                                    # [batch x sample, grid, y_dim + 1]

        # Convolution
        b_r = self.cnn(b_h)                                                                         # [batch x sample, grid, r_dim]

        # Decode
        b_r = self.decoder(s_x_ctx, x_grid, b_r, mask_grid)                                         # [batch x sample, target, y_dim]
        b_mu_log_sigma = nn.Dense(2 * y_ctx.shape[-1])(b_r)                                         # [batch x sample, target, y_dim x 2]
        b_mu, b_log_sigma = jnp.split(b_mu_log_sigma, 2, axis=-1)                                   # [batch, sample, target, y_dim] x 2
        b_sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(b_log_sigma)                  # [batch, sample, target, y_dim]

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

    @nn.compact
    def __call__(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        *,
        num_samples: int = 1,
        return_aux: bool = False,
    ) -> Union[
        Tuple[Array[B, S, T, Y], Array[B, S, T, Y]],
        Tuple # Add type hints for return_aux
    ]:

        # Discretize
        x_grid, mask_grid = self.discretizer(x_ctx, x_tar, mask_ctx, mask_tar)                      # [1, grid, x_dim] (broadcastable to [batch, grid, x_dim]), [discrete]

        # Bootstrapping
        b_mu, b_sigma = self._bootstrap(x_ctx, y_ctx, mask_ctx, num_samples)                        # [batch, sample, context, y_dim] x 2
        res_x_ctx, res_y_ctx = self._residual_sample(x_ctx, y_ctx, b_mu, b_sigma, mask_ctx)         # [batch, sample, context, x_dim], [batch, sample, context, y_dim]

        # Encode
        h_ctx = self.encoder(x_grid, x_ctx, y_ctx, mask_ctx)                                        # [batch, grid, y_dim + 1]
        s_h_ctx = jnp.repeat(h_ctx, num_samples, axis=0)                                            # [batch x sample, grid, y_dim + 1]

        batch_sample_shape = res_x_ctx.shape[0:2]
        res_x_ctx = F.flatten(res_x_ctx, start=0, stop=2)                                           # [batch x sample, context, x_dim]
        res_y_ctx = F.flatten(res_y_ctx, start=0, stop=2)                                           # [batch x sample, context, y_dim]
        s_mask_ctx = jnp.repeat(mask_ctx, num_samples, axis=0)                                      # [batch x sample, context]

        res_h_ctx = self.encoder(x_grid, res_x_ctx, res_y_ctx, s_mask_ctx)                          # [batch x sample, grid, y_dim + 1]
        h = s_h_ctx + res_h_ctx                                                                     # [batch x sample, grid, y_dim + 1]

        # Convolution
        r = self.cnn(h)                                                                             # [batch x sample, grid, r_dim]

        # Decode
        s_x_tar = jnp.repeat(x_tar, num_samples, axis=0)                                            # [batch x sample, target, x_dim]
        r = self.decoder(s_x_tar, x_grid, r, mask_grid)                                             # [batch x sample, target, y_dim]
        linear = nn.Dense(2 * y_ctx.shape[-1])
        mu_log_sigma = linear(r)                                                                    # [batch x sample, target, y_dim x 2]
        mu_log_sigma = F.unflatten(mu_log_sigma, batch_sample_shape, axis=0)                        # [batch, sample, target, y_dim x 2]
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, sample, target, y_dim] x 2
        sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(log_sigma)                      # [batch, sample, target, y_dim]

        mu    = F.masked_fill(mu,    mask_tar, non_mask_axis=(1, -1))                               # [batch, sample, target, y_dim]
        sigma = F.masked_fill(sigma, mask_tar, non_mask_axis=(1, -1))                               # [batch, sample, target, y_dim]

        if return_aux:
            # Convolution
            r_base = self.cnn(h_ctx)                                                                # [batch, grid, r_dim]

            # Decode
            r_base = self.decoder(x_tar, x_grid, r_base, mask_grid)                                 # [batch, target, y_dim]
            mu_log_sigma_base = linear(r_base)                                                      # [batch, target, y_dim x 2]
            mu_base, log_sigma_base = jnp.split(mu_log_sigma_base, 2, axis=-1)                      # [batch, target, y_dim] x 2
            sigma_base = self.min_sigma + (1 - self.min_sigma) * nn.softplus(log_sigma_base)        # [batch, target, y_dim]

            return mu, sigma, (mu_base, sigma_base)                                                 # [batch, sample, target, y_dim] x 2, [FIXME something]
        else:
            return mu, sigma                                                                        # [batch, sample, target, y_dim] x 2

    def log_likelihood(
        self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[B, C],
        mask_tar: Array[B, T],
        *,
        num_samples: int = 1,
        as_mixture: bool = True,
        return_aux: bool = False,
    ) -> Union[
        Array,
        Tuple[Array, Array[B, T, R]],
    ]:

        outs = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_samples=num_samples, return_aux=return_aux) # [batch, sample, *target, y_dim] x 2, [FIXME something]

        if return_aux:
            mu, sigma, aux = outs
        else:
            mu, sigma = outs

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
            return ll, aux                                                                          # (1), [FIXME something]
        else:
            return ll                                                                               # (1)                                                                               # (1)

    def loss(
        self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[B, C],
        mask_tar: Array[B, T],
        *,
        num_samples: int = 1,
        as_mixture: bool = True,
        return_aux: bool = False,
    ) -> Array:

        ll, (mu_base, sigma_base) = self.log_likelihood(                                            # (1), [FIXME something]
            x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar,
            num_samples=num_samples, as_mixture=as_mixture, return_aux=True,
        )

        log_prob_base = stats.norm.logpdf(y_tar, mu_base, sigma_base)                               # [batch, target, y_dim]
        ll_base = jnp.sum(log_prob_base, axis=-1)                                                   # [batch, target]
        ll_base = F.masked_mean(ll_base, mask_tar)                                                  # (1)

        loss = -(ll + ll_base)                                                                      # (1)

        if return_aux:
            return loss, dict(ll=ll, ll_base=ll_base)
        else:
            return loss


#! TODO: Add 2d model
class ConvBNP:
    """
    Convolutional Bootstrapping Neural Process
    """

    def __new__(cls,
        y_dim: int,
        x_min: float,
        x_max: float,
        r_dim: int = 64,
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
            cnn_dims = cnn_dims or (r_dim,) * 4
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
            cnn         = Net(dimension=1, hidden_features=cnn_dims, out_features=r_dim),
            decoder     = SetConv1dDecoder(init_log_scale=init_log_scale),
        )
