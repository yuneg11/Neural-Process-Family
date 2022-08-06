from ..typing import *

import math

from jax import random
from jax import numpy as jnp
from jax.scipy import stats
from flax import linen as nn

from .base import NPF
from .. import functional as F
from ..data import NPData
from ..utils import npf_io
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


def kl_divergence(mu_1, sigma_1, mu_2, sigma_2):
    kld = (
        jnp.log(sigma_2) - jnp.log(sigma_1)
        + (jnp.square(sigma_1) + jnp.square(mu_1 - mu_2)) / (2 * jnp.square(sigma_2))
        - 0.5
    )
    return kld


class ConvNPBase(NPF):
    """
    Base class of Convolutional Neural Process
    """

    z_dim:       int
    discretizer: Optional[nn.Module] = None
    encoder:     nn.Module = None
    cnn:         nn.Module = None
    cnn_post_z:  nn.Module = None
    decoder:     nn.Module = None
    min_sigma:   float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.encoder is None:
            raise ValueError("encoder is not specified")
        if self.cnn is None:
            raise ValueError("cnn is not specified")
        if self.cnn_post_z is None:
            raise ValueError("cnn_post_z is not specified")
        if self.decoder is None:
            raise ValueError("decoder is not specified")

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

        # Discretize
        x_grid, mask_grid = self.discretizer(data.x_ctx, data.x, data.mask_ctx, data.mask)          # [1, discrete, x_dim] (broadcastable to [batch, discrete, x_dim]), [discrete]

        # Encode
        h = self.encoder(x_grid, data.x_ctx, data.y_ctx, data.mask_ctx)                             # [batch, discrete, y_dim + 1]

        # Convolution
        h = self.cnn(h)

        # Transform to qz parameters and sample z
        r = nn.Dense(2 * self.z_dim)(h)
        z_mu_log_sigma = nn.Dense(2 * self.z_dim)(r)                                                # [batch, grid, z_dim x 2]
        z_mu_log_sigma = jnp.expand_dims(z_mu_log_sigma, axis=1)                                    # [batch, 1, grid, z_dim x 2]
        z_mu, z_log_sigma = jnp.split(z_mu_log_sigma, 2, axis=-1)                                   # [batch, 1, grid, z_dim] x 2
        z_sigma = self.min_sigma + (1 - self.min_sigma) * nn.sigmoid(z_log_sigma)                   # [batch, 1, grid, z_dim]

        rng = self.make_rng("sample")
        num_batches = z_mu.shape[0]
        eps = random.normal(rng, shape=(num_batches, num_latents) + z_mu.shape[1:])                 # [batch, latent, grid, z_dim]

        z_orig = z = z_mu + z_sigma * eps                                                           # [batch, latent, grid, z_dim]

        z, shape = F.flatten(z, start=0, stop=2, return_shape=True)                                 # [batch x latent, grid, z_dim]
        z = self.cnn_post_z(z)                                                                      # [batch x latent, grid, z_dim]
        z = F.unflatten(z, shape, axis=0)                                                           # [batch, latent, grid, z_dim]

        # Decode
        h = self.decoder(
            jnp.expand_dims(data.x, 1),
            jnp.expand_dims(x_grid, 1),
            z,
            jnp.expand_dims(mask_grid, 1),
        )                                                                                            # [batch, latent, target, y_dim]

        mu_log_sigma = nn.Dense(2 * data.y_ctx.shape[-1])(h)                                        # [batch, latent, target, y_dim x 2]
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, latent, target, y_dim] x 2
        sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(log_sigma)                      # [batch, latent, target, y_dim]

        mu    = F.masked_fill(mu,    data.mask, non_mask_axis=(1, -1))                               # [batch, latent, target, y_dim]
        sigma = F.masked_fill(sigma, data.mask, non_mask_axis=(1, -1))                               # [batch, latent, target, y_dim]

        if training and return_aux:

            return mu, sigma, (z_orig, z_mu, z_sigma, z_mu_ctx, z_sigma_ctx)
        else:
            return mu, sigma

    @npf_io
    def log_likelihood(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
    ) -> Array:

        mu, sigma = self(data, num_latents=num_latents, training=False, skip_io=True)               # [batch, latent, *point, y_dim] x 2

        s_y = jnp.expand_dims(data.y, axis=1)                                                       # [batch, 1,      *point, y_dim]
        log_prob = stats.norm.logpdf(s_y, mu, sigma)                                                # [batch, latent, *point, y_dim]
        ll = jnp.sum(log_prob, axis=-1)                                                             # [batch, latent, *point]

        axis = [-i for i in range(1, ll.ndim - 1)]
        ll = F.logmeanexp(ll, axis=1)                                                               # [batch, *point]
        ll = F.masked_mean(ll, data.mask, axis=axis)                                                # [batch]
        ll = jnp.mean(ll)                                                                           # (1)

        return ll

    @npf_io
    def loss(self, data: NPData, *, num_latents: int = 1, return_aux: bool = False) -> Array:
        if self.loss_type == "vi" or self.loss_type == "iwae":
            return self.iwae_loss(data, num_latents=num_latents, return_aux=return_aux, skip_io=True)
        elif self.loss_type == "elbo":
            return self.elbo_loss(data, num_latents=num_latents, return_aux=return_aux, skip_io=True)
        elif self.loss_type == "ml":
            return self.ml_loss(data, num_latents=num_latents, skip_io=True)

    @npf_io
    def iwae_loss(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
        return_aux: bool = False,
    ) -> Array:

        mu, sigma, (z, z_mu, z_sigma, z_mu_ctx, z_sigma_ctx) = self(                                # [batch, latent, point, y_dim] x 2, ([batch, latent, z_dim], [batch, 1, z_dim] x 2)
            data, num_latents=num_latents, training=True, return_aux=True, skip_io=True,
        )

        s_y = jnp.expand_dims(data.y, axis=1)                                                       # [batch, 1,      *point, y_dim]
        log_prob = stats.norm.logpdf(s_y, mu, sigma)                                                # [batch, latent, *point, y_dim]
        ll = jnp.sum(log_prob, axis=-1)                                                             # [batch, latent, *point]

        axis = [-i for i in range(1, ll.ndim - 1)]
        ll = F.masked_sum(ll, data.mask, axis=axis, non_mask_axis=1)                                # [batch, latent]

        log_p = stats.norm.logpdf(z, z_mu_ctx, z_sigma_ctx)                                         # [batch, latent, z_dim]
        log_p = jnp.sum(log_p, axis=-1)                                                             # [batch, latent]

        log_q = stats.norm.logpdf(z, z_mu, z_sigma)                                                 # [batch, latent, z_dim]
        log_q = jnp.sum(log_q, axis=-1)                                                             # [batch, latent]

        loss = - F.logmeanexp(ll + log_p - log_q, axis=1)                                           # [batch]
        loss = jnp.mean(loss)                                                                       # (1)

        return loss

    @npf_io
    def elbo_loss(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
        return_aux: bool = False,
    ) -> Array:

        mu, sigma, (_, z_mu, z_sigma, z_mu_ctx, z_sigma_ctx) = self(                                # [batch, latent, *point, y_dim] x 2, ([batch, latent, z_dim], [batch, 1, z_dim] x 2)
            data, num_latents=num_latents, training=True, return_aux=True, skip_io=True,
        )

        s_y = jnp.expand_dims(data.y, axis=1)                                                       # [batch, 1,      *point, y_dim]
        log_prob = stats.norm.logpdf(s_y, mu, sigma)                                                # [batch, latent, *point, y_dim]
        ll = jnp.sum(log_prob, axis=-1)                                                             # [batch, latent, *point]

        axis = [-i for i in range(1, ll.ndim - 1)]
        ll = F.masked_sum(ll, data.mask, axis=axis, non_mask_axis=1)                                # [batch, latent]
        ll = jnp.mean(ll)                                                                           # (1)

        kld = kl_divergence(z_mu, z_sigma, z_mu_ctx, z_sigma_ctx)                                   # [batch, 1, z_dim]
        kld = jnp.sum(kld, axis=(-2, -1))                                                           # [batch]
        kld = jnp.mean(kld)                                                                         # (1)

        loss = -ll + kld                                                                            # (1)

        if return_aux:
            return loss, dict(ll=ll, kld=kld)
        else:
            return loss

    @npf_io
    def ml_loss(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
    ) -> Array:

        mu, sigma = self(data, num_latents=num_latents, training=True, skip_io=True)                # [batch, latent, *point, y_dim] x 2

        s_y = jnp.expand_dims(data.y, axis=1)                                                       # [batch, 1,      *point, y_dim]
        log_prob = stats.norm.logpdf(s_y, mu, sigma)                                                # [batch, latent, *point, y_dim]
        ll = jnp.sum(log_prob, axis=-1)                                                             # [batch, latent, *point]

        axis = [-i for i in range(1, ll.ndim - 1)]
        ll = F.masked_sum(ll, data.mask, axis=axis, non_mask_axis=1)                                # [batch, latent]
        ll = F.logmeanexp(ll, axis=1)                                                               # [batch]
        ll = jnp.mean(ll)                                                                           # (1)

        loss = -ll                                                                                  # (1)

        return loss                                                                                 # (1)

class ConvNP:
    """
    Convolutional Neural Process
    """

    def __new__(cls,
        y_dim: int,
        x_min: float,
        x_max: float,
        cnn_dims: Optional[Sequence[int]] = None,
        cnn_xl: bool = False,
        points_per_unit: int = 64,
        x_margin: float = 0.1,
        r_dim: int = 64,
        z_dim: int = 64
    ):
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

        discretizer = Discretization1d(
            minval=x_min,
            maxval=x_max,
            points_per_unit=points_per_unit,
            multiple=multiple,
            margin=x_margin,
        )

        return ConvNPBase(
            z_dim       = z_dim,
            discretizer = discretizer,
            encoder     = SetConv1dEncoder(init_log_scale=init_log_scale),
            cnn         = Net(dimension=1, hidden_features=cnn_dims, out_features=r_dim),
            cnn_post_z  = Net(dimension=1, hidden_features=cnn_dims, out_features=r_dim),
            decoder     = SetConv1dDecoder(init_log_scale=init_log_scale)
        )
