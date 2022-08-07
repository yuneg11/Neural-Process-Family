from ..typing import *

import math

from jax import random
from jax import numpy as jnp
from jax.scipy import stats
from flax import linen as nn

from .base import NPF
from .. import functional as F
from ..data import NPData
from ..utils import npf_io, MultivariateNormalDiag
from ..modules import (
    # UNet,
    CNN,
    Discretization1d,
    SetConv1dEncoder,
    SetConv1dDecoder,
)


__all__ = [
    "ConvNPBase",
    "ConvNP",
]


# TODO: Change model to support on-the-grid(discretized) data.

# NOTE: Currently, `SetConv*` modules to not transforms the output dimension (input_dim = output_dim).
#       This is based on the original definition described in the paper.
#       However, the official implementation transforms the output dimension with pointwise transformation.
#       See (https://github.com/cambridge-mlg/convcnp/blob/3aa2d9c96ff42e55a3c0d8384d084459f19d00f5/convcnp/set_conv.py#L120).
#       In the future, we also need to change the implementation of `SetConv*` modules to match the official implementation.
#       This implementation uses some workaround to mimic some of the behavior of the official implementation.
#           (`mu_log_sigma = nn.Dense(features=(2 * data.y.shape[-1]))(y)`)


class ConvNPBase(NPF):
    """
    Base class of Convolutional Neural Process
    """

    discretizer:      Optional[nn.Module] = None
    encoder:          nn.Module = None
    determ_cnn:       nn.Module = None
    latent_cnn:       nn.Module = None
    decoder:          nn.Module = None
    loss_type:        str = "vi"
    min_sigma:        float = 0.1
    min_latent_sigma: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.encoder is None:
            raise ValueError("encoder is not specified")
        if self.determ_cnn is None:
            raise ValueError("determ_cnn is not specified")
        if self.latent_cnn is None:
            raise ValueError("latent_cnn is not specified")
        if self.decoder is None:
            raise ValueError("decoder is not specified")

    def _latent_dist(
        self,
        z_i:  Array[B, [G], Z * 2],
        mask: Array[B, [G]],
    ) -> Tuple[Array[B, 1, [G], Z], Array[B, 1, [G], Z]]:

        z_mu_log_sigma = jnp.expand_dims(z_i, axis=1)                                               # [batch, 1, *grid, z_dim x 2]
        z_mu, z_log_sigma = jnp.split(z_mu_log_sigma, 2, axis=-1)                                   # [batch, 1, *grid, z_dim] x 2
        z_sigma = self.min_latent_sigma + (1 - self.min_latent_sigma) * nn.sigmoid(z_log_sigma)     # [batch, 1, *grid, z_dim]
        return z_mu, z_sigma                                                                        # [batch, 1, *grid, z_dim] x 2

    def _latent_sample(
        self,
        z_mu:    Array[B, 1, [G], Z],
        z_sigma: Array[B, 1, [G], Z],
        num_latents: int = 1,
    ) -> Array[B, L, [G], Z]:

        rng = self.make_rng("sample")
        num_batches, other_shape = z_mu.shape[0], z_mu.shape[2:]
        eps = random.normal(rng, shape=(num_batches, num_latents, *other_shape))                    # [batch, latent, *grid, z_dim]

        z = z_mu + z_sigma * eps                                                                    # [batch, latent, *grid, z_dim]
        return z                                                                                    # [batch, latent, *grid, z_dim]

    @nn.compact
    @npf_io
    def __call__(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
        training: bool = False,
        return_aux: bool = False,
    ) -> Tuple[Array[B, [T], Y], Array[B, [T], Y]]:

        x_grid, mask_grid = self.discretizer(data.x_ctx, data.x, data.mask_ctx, data.mask)          # [1, grid, x_dim] (broadcastable to [batch, grid, x_dim]), [1, grid]

        if training:
            h = self.encoder(x_grid, data.x, data.y, data.mask)                                     # [batch, *grid, y_dim + 1]
            z_i = self.determ_cnn(h)                                                                # [batch, *grid, z_dim x 2]
            z_mu, z_sigma = self._latent_dist(z_i, mask_grid)                                       # [batch, 1, *grid, z_dim] x 2
        else:
            h_ctx = self.encoder(x_grid, data.x_ctx, data.y_ctx, data.mask_ctx)                     # [batch, *grid, y_dim + 1]
            z_i_ctx = self.determ_cnn(h_ctx)                                                        # [batch, *grid, z_dim x 2]
            z_mu, z_sigma = self._latent_dist(z_i_ctx, mask_grid)                                   # [batch, 1, *grid, z_dim] x 2

        z = self._latent_sample(z_mu, z_sigma, num_latents=num_latents)                             # [batch, latent, *grid, z_dim]

        f_z, shape = F.flatten(z, start=0, stop=2, return_shape=True)                               # [batch x latent, *grid, z_dim], shape
        f_y_grid = self.latent_cnn(f_z)                                                             # [batch x latent, *grid, y_dim]    # actually, y_dim is r_dim. See above NOTE.
        y_grid = F.unflatten(f_y_grid, shape, axis=0)                                               # [batch,  latent, *grid, y_dim]

        y = self.decoder(data.x, x_grid, y_grid, mask_grid)                                         # [batch, latent, *point, y_dim]

        mu_log_sigma = nn.Dense(features=(2 * data.y.shape[-1]))(y)                                 # [batch, latent, *point, y_dim x 2]
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, latent, *point, y_dim] x 2
        sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(log_sigma)                      # [batch, latent, *point, y_dim]

        mu    = F.masked_fill(mu,    data.mask, fill_value=0., non_mask_axis=(1, -1))               # [batch, latent, *point, y_dim]
        sigma = F.masked_fill(sigma, data.mask, fill_value=0., non_mask_axis=(1, -1))               # [batch, latent, *point, y_dim]

        if return_aux:
            h_ctx = self.encoder(x_grid, data.x_ctx, data.y_ctx, data.mask_ctx)                     # [batch, *grid, y_dim + 1]
            z_i_ctx = self.determ_cnn(h_ctx)                                                        # [batch, *grid, z_dim x 2]
            z_mu_ctx, z_sigma_ctx = self._latent_dist(z_i_ctx, mask_grid)                           # [batch, 1, *grid, z_dim] x 2

            return mu, sigma, (z, z_mu, z_sigma, z_mu_ctx, z_sigma_ctx, mask_grid)
        else:
            return mu, sigma

    @npf_io
    def log_likelihood(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
        joint: bool = False,
        split_set: bool = False,
    ) -> Array:

        mu, sigma = self(data, num_latents=num_latents, skip_io=True)                               # [batch, latent, *point, y_dim] x 2

        s_y = jnp.expand_dims(data.y, axis=1)                                                       # [batch, 1,      *point, y_dim]
        log_prob = MultivariateNormalDiag(mu, sigma).log_prob(s_y)                                  # [batch, latent, *point]
        axis = [-i for i in range(1, log_prob.ndim - 1)]

        if joint:
            ll = F.masked_sum(log_prob, data.mask, axis=axis, non_mask_axis=1)                      # [batch, latent]
            ll = F.logmeanexp(ll, axis=1) / jnp.sum(data.mask, axis=-1)                             # [batch]

            if split_set:
                ll_ctx = F.masked_sum(log_prob, data.mask_ctx, axis=axis, non_mask_axis=1)          # [batch, latent]
                ll_tar = F.masked_sum(log_prob, data.mask_tar, axis=axis, non_mask_axis=1)          # [batch, latent]
                ll_ctx = F.logmeanexp(ll_ctx, axis=1) / jnp.sum(data.mask_ctx, axis=-1)             # [batch]
                ll_tar = F.logmeanexp(ll_tar, axis=1) / jnp.sum(data.mask_tar, axis=-1)             # [batch]

        else:
            ll_all = F.logmeanexp(log_prob, axis=1)                                                 # [batch, *point]
            ll = F.masked_mean(ll_all, data.mask, axis=axis)                                        # [batch]

            if split_set:
                ll_ctx = F.masked_mean(ll_all, data.mask_ctx, axis=-1)                              # [batch]
                ll_tar = F.masked_mean(ll_all, data.mask_tar, axis=-1)                              # [batch]

        ll = jnp.mean(ll)                                                                           # (1)

        if split_set:
            ll_ctx = jnp.mean(ll_ctx)                                                               # (1)
            ll_tar = jnp.mean(ll_tar)                                                               # (1)

            return ll, ll_ctx, ll_tar                                                               # (1) x 3
        else:
            return ll                                                                               # (1)

    @npf_io
    def loss(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
        joint: bool = True,        # For `ml_loss`
        return_aux: bool = False,  # For `elbo_loss`
    ) -> Array:
        if self.loss_type == "vi" or self.loss_type == "iwae":
            return self.iwae_loss(data, num_latents=num_latents, skip_io=True)
        elif self.loss_type == "elbo":
            return self.elbo_loss(data, num_latents=num_latents, return_aux=return_aux, skip_io=True)
        elif self.loss_type == "ml":
            return self.ml_loss(data, num_latents=num_latents, joint=joint, skip_io=True)

    @npf_io
    def iwae_loss(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
    ) -> Array:

        mu, sigma, (z, z_mu, z_sigma, z_mu_ctx, z_sigma_ctx, mask_grid) = self(                     # [batch, latent, point, y_dim] x 2,
            data, num_latents=num_latents, training=True, return_aux=True, skip_io=True,            #     ([batch, latent, *grid, z_dim], [batch, 1, *grid, z_dim] x 4, [batch, *grid])
        )

        s_y = jnp.expand_dims(data.y, axis=1)                                                       # [batch, 1,      *point, y_dim]
        log_prob = MultivariateNormalDiag(mu, sigma).log_prob(s_y)                                  # [batch, latent, *point]
        axis = [-i for i in range(1, log_prob.ndim - 1)]

        ll = F.masked_sum(log_prob, data.mask, axis=axis, non_mask_axis=1)                          # [batch, latent]

        log_p = MultivariateNormalDiag(z_mu_ctx, z_sigma_ctx).log_prob(z)                           # [batch, latent, *grid]
        log_q = MultivariateNormalDiag(z_mu, z_sigma).log_prob(z)                                   # [batch, latent, *grid]
        log_p = F.masked_sum(log_p, mask_grid, axis=axis, non_mask_axis=1)                          # [batch, latent]
        log_q = F.masked_sum(log_q, mask_grid, axis=axis, non_mask_axis=1)                          # [batch, latent]

        loss = -F.logmeanexp(ll + log_p - log_q, axis=1) / jnp.sum(data.mask, axis=axis)            # [batch]
        loss = jnp.mean(loss)                                                                       # (1)

        return loss                                                                                 # (1)

    @npf_io
    def elbo_loss(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
        return_aux: bool = False,
    ) -> Array:

        mu, sigma, (_, z_mu, z_sigma, z_mu_ctx, z_sigma_ctx, mask_grid) = self(                     # [batch, latent, *point, y_dim] x 2,
            data, num_latents=num_latents, training=True, return_aux=True, skip_io=True,            #     (_, [batch, 1, *grid, z_dim] x 4, [batch, *grid])
        )

        s_y = jnp.expand_dims(data.y, axis=1)                                                       # [batch, 1,      *point, y_dim]
        log_prob = MultivariateNormalDiag(mu, sigma).log_prob(s_y)                                  # [batch, latent, *point]
        axis = [-i for i in range(1, log_prob.ndim - 1)]

        ll = F.masked_mean(log_prob, data.mask, axis=(1, *axis), non_mask_axis=1)                   # [batch]

        q_z = MultivariateNormalDiag(z_mu, z_sigma)                                                 # [batch, 1, *grid, z_dim]
        p_z = MultivariateNormalDiag(z_mu_ctx, z_sigma_ctx)                                         # [batch, 1, *grid, z_dim]

        kld = jnp.squeeze(q_z.kl_divergence(p_z), axis=1)                                           # [batch, *grid]
        kld = F.masked_sum(kld, mask_grid, axis=axis)                                               # [batch]

        loss = -ll + kld / jnp.sum(data.mask_ctx, axis=axis)                                        # [batch]
        loss = jnp.mean(loss)                                                                       # (1)

        if return_aux:
            ll = jnp.mean(ll)                                                                       # (1)
            kld = jnp.mean(kld)                                                                     # (1)
            return loss, dict(ll=ll, kld=kld)                                                       # (1), (aux)
        else:
            return loss                                                                             # (1)

    @npf_io
    def ml_loss(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
        joint: bool = True,
    ) -> Array:

        loss = -self.log_likelihood(data, num_latents=num_latents, joint=joint, skip_io=True)       # (1)
        return loss                                                                                 # (1)


class ConvNP:
    """
    Convolutional Neural Process
    """

    def __new__(cls,
        y_dim: int,
        x_min: float,
        x_max: float,
        z_dim: int = 64,
        r_dim: int = 64,
        cnn_dims: Optional[Sequence[int]] = None,
        cnn_xl: bool = False,
        on_the_grid: bool = False,
        points_per_unit: int = 64,
        loss_type: str = "vi",
        x_margin: float = 0.1,
        min_sigma: float = 0.1,
        min_latent_sigma: float = 0.1,
    ):

        assert on_the_grid is False, "on_the_grid is not supported yet"

        if cnn_xl:
            raise NotImplementedError("cnn_xl is not supported yet")
            Net = UNet
            cnn_dims = cnn_dims or (8, 16, 16, 32, 32, 64)
            multiple = 2 ** len(cnn_dims)  # num_halving_layers = len(cnn_dims)
        else:
            Net = CNN
            cnn_dims = (r_dim,) * 2
            multiple = 1

        init_log_scale = math.log(2. / points_per_unit)

        if on_the_grid:
            discretizer = None
            encoder = None
            decoder = None
        else:
            discretizer = Discretization1d(
                minval=x_min,
                maxval=x_max,
                points_per_unit=points_per_unit,
                multiple=multiple,
                margin=x_margin,
            )
            encoder = SetConv1dEncoder(init_log_scale=init_log_scale)   # TODO: Support dimension transformation
            decoder = SetConv1dDecoder(init_log_scale=init_log_scale)   # TODO: Support dimension transformation

        determ_cnn = Net(dimension=1, hidden_features=cnn_dims, out_features=(z_dim * 2))
        latent_cnn = Net(dimension=1, hidden_features=cnn_dims, out_features=r_dim)

        return ConvNPBase(
            discretizer=discretizer,
            encoder=encoder,
            determ_cnn=determ_cnn,
            latent_cnn=latent_cnn,
            decoder=decoder,
            loss_type=loss_type,
            min_sigma=min_sigma,
            min_latent_sigma=min_latent_sigma,
        )
