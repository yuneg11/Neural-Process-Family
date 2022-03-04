from ..type import *

import math

from jax import random
from jax import numpy as jnp
from flax import linen as nn

from .base import LatentNPF
from .. import functional as F
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



#! TODO: Change model to support on-the-grid(discretized) data.
class ConvNPBase(LatentNPF):
    """
    Base class of Convolutional Neural Process

    Args:
        discretizer: [[batch, context, x_dim], [batch, target, x_dim]] -> [1, discrete, 1]
        encoder:     [[batch, discrete, x_dim], [batch, context, x_dim], [batch, context, y_dim]] -> [batch, y_dim + 1, discrete]
        determ_cnn:  [batch, y_dim + 1, discrete] -> [batch, z_dim x 2, discrete]
        latent_cnn:  [batch, latent, z_dim, discrete] -> [batch, latent, y_dim x 2, discrete]
        decoder:     [[batch, latent, target, x_dim], [batch, latent, discrete, x_dim], [batch, latent, y_dim, discrete]] -> [batch, latent, target, y_dim]
        loss_type:   "vi" or "ml"
    """

    discretizer: nn.Module = None
    encoder:     nn.Module = None
    determ_cnn:  nn.Module = None
    latent_cnn:  nn.Module = None
    decoder:     nn.Module = None
    loss_type:   str = "ml"

    def __post_init__(self):
        super().__post_init__()
        if self.determ_cnn is None:
            raise ValueError("determ_cnn is not specified")
        if self.latent_cnn is None:
            raise ValueError("latent_cnn is not specified")
        if self.decoder is None:
            raise ValueError("decoder is not specified")

    def _determ_conv(self,
        h: Array[B, D, Y + 1],
    ) -> Tuple[Array[B, D, Z], Array[B, D, Z]]:

        z_mu_log_sigma = self.determ_cnn(h)                                     # [batch, discrete, z_dim x 2]
        z_mu, z_log_sigma = jnp.split(z_mu_log_sigma, 2, axis=-1)               # [batch, discrete, z_dim] x 2
        z_sigma = 0.1 + 0.9 * nn.sigmoid(z_log_sigma)                           # [batch, discrete, z_dim]
        return z_mu, z_sigma

    def _predict(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_latents:  int = 1,
    ) -> Tuple[Array[B, L, T, Y], Array[B, L, T, Y], Array[B, D, Z], Array[B, D, Z], Array[1, D, X]]:

        # Discretize
        x_grid, mask_grid = self.discretizer(x_ctx, x_tar, mask_ctx, mask_tar)  # [1, discrete, x_dim] (broadcastable to [batch, discrete, x_dim]), [discrete]

        # Encode
        h_ctx = self.encoder(x_grid, x_ctx, y_ctx, mask_ctx)                    # [batch, discrete, y_dim + 1]

        # Deterministic convolution
        z_ctx_mu, z_ctx_sigma = self._determ_conv(h_ctx)                        # [batch, discrete, z_dim] x 2

        # Latent sample
        rng = self.make_rng("sample")
        num_batches, num_discretes, z_dim = z_ctx_mu.shape
        shape = (num_batches, num_latents, num_discretes, z_dim)
        z_samples = jnp.expand_dims(z_ctx_mu, axis=-3) \
                  + jnp.expand_dims(z_ctx_sigma, axis=-3) * random.normal(rng, shape=shape)         # [batch, latent, discrete, z_dim]
        z_samples = z_samples.reshape(num_batches * num_latents, num_discretes, z_dim)              # [batch x latent, discrete, z_dim]

        # Latent convolution
        mu_log_sigma_grid = self.latent_cnn(z_samples)                                              # [batch x latent, discrete, y_dim x 2]
        mu_log_sigma_grid = mu_log_sigma_grid.reshape(num_batches, num_latents, num_discretes, -1)  # [batch, latent, discrete, y_dim x 2]
        mu_grid, log_sigma_grid = jnp.split(mu_log_sigma_grid, 2, axis=-1)                          # [batch, latent, discrete, y_dim] x 2
        sigma_grid = nn.softplus(log_sigma_grid)                                                    # [batch, latent, discrete, y_dim]

        # Decode
        mu    = self.decoder(x_tar, x_grid, mu_grid,    mask_grid)              # [batch, latent, target, y_dim]
        sigma = self.decoder(x_tar, x_grid, sigma_grid, mask_grid)              # [batch, latent, target, y_dim]

        mu    = F.apply_mask(mu,    mask_tar, axis=-2)                          # [batch, latent, target, y_dim]
        sigma = F.apply_mask(sigma, mask_tar, axis=-2)                          # [batch, latent, target, y_dim]

        return mu, sigma, z_ctx_mu, z_ctx_sigma, x_grid

    @nn.compact
    def __call__(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_latents:  int = 1,
    ) -> Tuple[Array[B, L, T, Y], Array[B, L, T, Y]]:

        mu, sigma, _, _, _ = self._predict(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_latents)
        return mu, sigma

    def vi_loss(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_latents: int = 1,
    ) -> Array:

        mu, sigma, z_ctx_mu, z_ctx_sigma, x_grid = self._predict(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_latents)

        # Target latent distribution
        h_tar = self.encoder(x_grid, x_tar, y_tar, mask_tar)                    # [batch, discrete, y_dim + 1]
        z_tar_mu, z_tar_sigma = self._determ_conv(h_tar)                        # [batch, discrete, z_dim] x 2

        # Loss
        log_likelihood = self._log_likelihood(y_tar, mu, sigma)                 # [batch, latent, target]
        log_likelihood = F.masked_mean(log_likelihood, mask_tar, axis=-1)       # [batch, latent]
        log_likelihood = jnp.mean(log_likelihood)                               # [1]

        kl_divergence = self._kl_divergence(z_tar_mu, z_tar_sigma, z_ctx_mu, z_ctx_sigma)  # [batch, discrete, z_dim]
        kl_divergence = jnp.mean(kl_divergence)                                 # [1]

        loss = -log_likelihood + kl_divergence                                  # [1]

        return loss


#! TODO: Add 2d model
class ConvNP(ConvNPBase):
    """
    Convolutional Neural Process
    """

    def __new__(cls,
        y_dim: int,
        x_min: float,
        x_max: float,
        z_dim: int = 8,
        determ_cnn_dims: Optional[Sequence[int]] = None,
        latent_cnn_dims: Optional[Sequence[int]] = None,
        determ_cnn_xl: bool = False,
        latent_cnn_xl: bool = False,
        points_per_unit: int = 64,
        x_margin: float = 0.1,
        loss_type: str = "ml",
    ):
        if determ_cnn_xl:
            raise NotImplementedError("determ_cnn_xl is not supported yet")
            DetermNet = UNet
            determ_cnn_dims = determ_cnn_dims or (8, 16, 16, 32, 32, 64)
            determ_multiple = 2 ** len(determ_cnn_dims)  # num_halving_layers = len(cnn_dims)
        else:
            DetermNet = CNN
            determ_cnn_dims = determ_cnn_dims or (16, 32, 16)
            determ_multiple = 1

        if latent_cnn_xl:
            raise NotImplementedError("latent_cnn_xl is not supported yet")
            LatentNet = UNet
            latent_cnn_dims = latent_cnn_dims or (8, 16, 16, 32, 32, 64)
            latent_multiple = 2 ** len(latent_cnn_dims)  # num_halving_layers = len(cnn_dims)
        else:
            LatentNet = CNN
            latent_cnn_dims = latent_cnn_dims or (16, 32, 16)
            latent_multiple = 1


        init_log_scale = math.log(2. / points_per_unit)
        multiple = 2 ** max(determ_multiple, latent_multiple)

        discretizer = Discretization1d(
            minval=x_min,
            maxval=x_max,
            points_per_unit=points_per_unit,
            multiple=multiple,
            margin=x_margin,
        )

        return ConvNPBase(
            discretizer = discretizer,
            encoder     = SetConv1dEncoder(init_log_scale=init_log_scale),
            determ_cnn  = DetermNet(hidden_features=determ_cnn_dims, out_features=(z_dim * 2)),
            latent_cnn  = LatentNet(hidden_features=latent_cnn_dims, out_features=(y_dim * 2)),
            decoder     = SetConv1dDecoder(init_log_scale=init_log_scale),
            loss_type   = loss_type,
        )
