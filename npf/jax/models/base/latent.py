from ...type import *

import math
from abc import abstractmethod

import jax
from jax import numpy as jnp

from .univariate import UnivariateNPF
from ... import functional as F


__all__ = [
    "LatentNPF",
]


class LatentNPF(UnivariateNPF):
    """
    Base class for latent NPF models
    """

    loss_type: str = "vi"

    @property
    def is_latent_model(self) -> bool:
        return True

    def __post_init__(self):
        super().__post_init__()
        if self.loss_type not in ("vi", "ml"):
            raise ValueError(f"Invalid loss_type: {self.loss_type}. loss_type must be either 'vi' or 'ml'.")

    @staticmethod
    def _kl_divergence(
        z0_mu:    Array[...],
        z0_sigma: Array[...],
        z1_mu:    Array[...],
        z1_sigma: Array[...],
    ) -> Array[...]:
        """
        Calculate element-wise KL divergence between two Gaussian distributions.
        """

        kl_div = jnp.log(z1_sigma) - jnp.log(z0_sigma) \
               + (jnp.square(z0_sigma) + jnp.square(z0_mu - z1_mu)) / (2 * jnp.square(z1_sigma)) \
               - 0.5
        return kl_div

    def log_likelihood(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_latents: int = 1,
        **kwargs,
    ) -> Array:
        """
        """

        mu, sigma = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_latents, **kwargs)            # [batch, latent, *target, y_dim] x 2
        target_axis = [-d for d in range(1, mask_tar.ndim)]
        mask_axis = [0] + target_axis

        y_tar = jnp.expand_dims(y_tar, axis=1)                                                      # [batch, 1, *target, y_dim]
        log_likelihood = self._log_likelihood(y_tar, mu, sigma)                                     # [batch, latent, *target]
        log_likelihood = F.masked_mean(log_likelihood, mask_tar, axis=target_axis, mask_axis=mask_axis) # [batch, latent]
        log_likelihood = F.logmeanexp(log_likelihood, axis=1)                                       # [batch]
        log_likelihood = jnp.mean(log_likelihood)                                                   # [1]

        return log_likelihood

    def loss(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_latents: int = 1,
        **kwargs,
    ) -> Array:
        """
        Calculate loss.
        """
        if self.loss_type == "vi":
            return self.vi_loss(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, num_latents, **kwargs)
        elif self.loss_type == "ml":
            return self.ml_loss(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, num_latents, **kwargs)

    @abstractmethod
    def vi_loss(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_latents: int = 1,
        **kwargs,
    ) -> Array:
        """
        Calculate VI loss.
        """
        raise NotImplementedError

    def ml_loss(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_latents: int = 1,
        **kwargs,
    ) -> Array:
        """
        Calculate Maximum-Likelihood loss.
        """

        loss = -self.log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, num_latents, **kwargs)
        return loss
