from ..type import *

import math
from abc import abstractmethod

import numpy as np

import jax
from jax import numpy as jnp
from jax.scipy import stats
from flax import linen as nn


__all__ = [
    "NPF",
    "UnivariateNPF",
    "ConditionalNPF",
    "LatentNPF",
    "MultivariateNPF",
]


class NPF(nn.Module):
    """
    Base class for NPF models
    """

    @property
    def is_multivariate_model(self) -> bool:
        raise NotImplementedError

    @property
    def is_latent_model(self) -> bool:
        raise NotImplementedError

    @property
    def num_params(self) -> int:
        raise NotImplementedError


class UnivariateNPF(NPF):
    """
    Base class for univariate NPF models
    """

    @property
    def is_multivariate_model(self) -> bool:
        return False


class ConditionalNPF(UnivariateNPF):
    """
    Base class for conditional NPF models
    """

    @property
    def is_latent_model(self) -> bool:
        return False

    # Static method

    @staticmethod
    def _log_likelihood(
        y_tar:    Float[B, T, Y],
        mu:       Float[B, T, Y],
        sigma:    Float[B, T, Y],
        mask_tar: Float[B, T],
    ) -> Float[B, T]:
        """
        Calculate element-wise log-likelihood of Gaussian distribution.

        Args:
            y_tar:       Array[batch, target, y_dim]
            mu:          Array[batch, target, y_dim]
            sigma:       Array[batch, target, y_dim]
            mask_tar:    Array[batch, target]

        Returns:
            log_likelihood: Array[batch, target]
        """

        mask_tar = jnp.expand_dims(mask_tar, axis=-1)
        log_prob = stats.norm.logpdf(y_tar, loc=mu, scale=sigma)                # [batch, target, y_dim]
        log_prob = jnp.where(mask_tar, log_prob, 0.)
        log_likelihood = jnp.sum(log_prob, axis=-1)                             # [batch, target]
        return log_likelihood

    # Forward

    @abstractmethod
    def __call__(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
    ) -> Tuple[Float[B, T, Y], Float[B, T, Y]]:
        """
        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]

        Returns:
            mu:       Array[batch, target, y_dim]
            sigma:    Array[batch, target, y_dim]
        """

        raise NotImplementedError

    # Likelihood

    def log_likelihood(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        y_tar:    Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
    ) -> Float:
        """
        Calculate log-likelihood.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]

        Returns:
            log_likelihood: float
        """

        mu, sigma = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar)               # [batch, target, y_dim] x 2
        log_likelihood = self._log_likelihood(y_tar, mu, sigma, mask_tar)       # [batch, target]
        log_likelihood = jnp.sum(log_likelihood) / jnp.sum(mask_tar)            # [1]
        return log_likelihood

    # Loss

    def loss(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        y_tar:    Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
    ) -> Float:
        """
        Calculate loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]

        Returns:
            loss      float
        """

        loss = -self.log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        return loss


class LatentNPF(UnivariateNPF):
    """
    Base class for latent NPF models
    """

    @property
    def is_latent_model(self) -> bool:
        return True

    # Static methods

    @staticmethod
    def _log_likelihood(
        y_tar:    Float[B, T, Y],
        mu:       Float[B, L, T, Y],
        sigma:    Float[B, L, T, Y],
        mask_tar: Float[B, T],
    ) -> Float[B, L, T]:
        """
        Calculate element-wise log-likelihood of Gaussian distribution.

        Args:
            y_tar:          Array[batch, target, y_dim]
            mu:             Array[batch, latent, target, y_dim]
            sigma:          Array[batch, latent, target, y_dim]
            mask_tar:       Array[batch, target]

        Returns:
            log_likelihood: Array[batch, latent, target]
        """

        mask_tar = jnp.expand_dims(mask_tar, axis=-1)
        y_tar = jnp.expand_dims(y_tar, axis=1)                                  # [batch, 1, target, y_dim]
        mask_tar = jnp.expand_dims(mask_tar, axis=1)                            # [batch, 1, target, 1]
        log_prob = stats.norm.logpdf(y_tar, loc=mu, scale=sigma)                # [batch, latent, target, y_dim]
        log_prob = jnp.where(mask_tar, log_prob, 0.)
        log_likelihood = jnp.sum(log_prob, axis=-1)                             # [batch, latent, target]
        return log_likelihood

    @staticmethod
    def _kl_divergence(
        z0_mu:    Float,
        z0_sigma: Float,
        z1_mu:    Float,
        z1_sigma: Float,
    ) -> Float:  # TODO: Improve typing
        """
        Calculate element-wise KL divergence between two Gaussian distributions.

        Args:
            z0_mu:    Array[...]
            z0_sigma: Array[...]
            z1_mu:    Array[...]
            z1_sigma: Array[...]

        Returns:
            kl_divergence:   Array[...]
        """

        kl_div = jnp.log(z1_sigma) - jnp.log(z0_sigma) \
               + (jnp.square(z0_sigma) + jnp.square(z0_mu - z1_mu)) / (2 * jnp.square(z1_sigma)) \
               - 0.5
        return kl_div

    # Forward

    @abstractmethod
    def __call__(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
        num_latents: int = 1,
    ) -> Tuple[Float[B, L, T, Y], Float[B, L, T, Y]]:
        """
        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]
            num_latents:  int

        Returns:
            mu:       Array[batch, latent, target, y_dim]
            sigma:    Array[batch, latent, target, y_dim]
        """

    # Likelihood

    def log_likelihood(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        y_tar:    Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
        num_latents: int = 1,
    ) -> Float:
        """
        Calculate log-likelihood.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]
            num_latents: int

        Returns:
            log_likelihood: float
        """

        mu, sigma = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_latents)  # [batch, latent, target, y_dim] x 2

        log_likelihood = self._log_likelihood(y_tar, mu, sigma, mask_tar)       # [batch, latent, target]
        log_likelihood = jnp.sum(log_likelihood, axis=-1) \
                       / jnp.sum(mask_tar, axis=1, keepdims=True)               # [batch, latent]
        log_likelihood = jax.nn.logsumexp(log_likelihood, axis=-1) \
                       - math.log(num_latents)                                  # [batch]
        log_likelihood = jnp.mean(log_likelihood)                               # [1]

        return log_likelihood

    # Losses

    def loss(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        y_tar:    Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
        num_latents: int = 1,
        loss_type:   str = "vi",
    ) -> Float:
        """
        Calculate loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]
            num_latents: int
            loss_type:   str ("vi" or "ml")

        Returns:
            loss: float
        """
        if loss_type == "vi":
            return self.vi_loss(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, num_latents)
        elif loss_type == "ml":
            return self.ml_loss(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, num_latents)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Loss type should be either 'vi' or 'ml'.")


    @abstractmethod
    def vi_loss(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        y_tar:    Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar:  Float[B, T],
        num_latents: int = 1,
    ) -> Float:
        """
        Calculate VI loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[batch, context]
            mask_tar:  Array[batch,  target]
            num_latents: int

        Returns:
            loss: float
        """
        raise NotImplementedError

    def ml_loss(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        y_tar:    Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
        num_latents: int = 1,
    ) -> Float:
        """
        Calculate Maximum-Likelihood loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]
            num_latents: int

        Returns:
            loss: float
        """

        loss = -self.log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, num_latents)
        return loss


class MultivariateNPF(NPF):
    """
    Base class for multivariate NPF models
    """

    @property
    def is_multivariate_model(self) -> bool:
        raise True

    @property
    def is_latent_model(self) -> bool:
        raise False

    # Static methods

    @staticmethod
    def _univariate_log_likelihood(
        y_tar:    Float[B, T, Y],
        mu:       Float[B, T, Y],
        sigma:    Float[B, T, Y],
        mask_tar: Float[B, T],
    ) -> Float[B, T]:
        """
        Calculate element-wise log-likelihood of univariate Gaussian distribution.

        Args:
            y_tar:       Array[batch, target, y_dim]
            mu:          Array[batch, target, y_dim]
            sigma:       Array[batch, target, y_dim]
            mask_tar:    Array[batch, target]

        Returns:
            log_likelihood: Array[batch, target]
        """

        mask_tar = jnp.expand_dims(mask_tar, axis=-1)                           # [batch, target, 1]
        log_prob = stats.norm.logpdf(y_tar, loc=mu, scale=sigma)                # [batch, target, y_dim]
        log_prob = jnp.where(mask_tar, log_prob, 0.)
        log_likelihood = jnp.sum(log_prob, axis=-1)                             # [batch, target]
        return log_likelihood

    @staticmethod
    def _multivariate_log_likelihood(
        y_tar:    Float[B, T, Y],
        mu:       Float[B, Y, T],
        cov:      Float[B, Y, T, T],
        mask_tar: Float[B, T],
    ) -> Float:   # TODO: Fix type annotation "Float[B]"
        """
        Calculate batch-wise log-likelihood of multivariate Gaussian distribution.

        Args:
            y_tar:       Array[batch, target, y_dim]
            mu:          Array[batch, y_dim, target]
            sigma:       Array[batch, y_dim, target, target]
            mask_tar:    Array[batch, target]

        Returns:
            log_likelihood: Array[batch, target]
        """

#         num_target = y_tar.shape[1]

#         y_tar = y_tar.transpose(1, 2)                                     # [batch, y_dim, target]
#         distribution = MultivariateNormal(mu, cov)                              # MultivariateNormal[batch, y_dim, target, target]
#         log_prob = distribution.log_prob(y_tar)                              # [batch, y_dim]
#         log_likelihood = torch.sum(log_prob, dim=1) / num_target                # [batch]
#         return log_likelihood
        raise NotImplementedError

    # Forward

    @abstractmethod
    def __call__(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
        as_univariate: bool = False,
    ) -> Union[
        Tuple[Float[B, Y, T], Float[B, Y, T, T]],
        Tuple[Float[B, T, Y], Float[B, T, Y]],
    ]:
        """
        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]
            as_univariate: bool

        Returns:
            mu:       Array[batch, y_dim, target]          if as_univariate == False
                      Array[batch, target, y_dim]          if as_univariate == True
            cov:      Array[batch, y_dim, target, target]  if as_univariate == False
                      Array[batch, target, y_dim]          if as_univariate == True
        """
        raise NotImplementedError


    # Likelihoods

    def log_likelihood(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        y_tar:    Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
        as_univariate: bool = False,
    ) -> Float:
        """
        Calculate log-likelihood.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]
            as_univariate: bool

        Returns:
            log_likelihood: float
        """

        if as_univariate:
            return self.univariate_log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        else:
            return self.multivariate_log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)

    def univariate_log_likelihood(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        y_tar:    Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
    ) -> Float:
        """
        Calculate univariate log-likelihood.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]

        Returns:
            log_likelihood: float
        """

#         mu, sigma = self(x_ctx, y_ctx, x_tar, as_univariate=True)    # [batch, target, y_dim] * 2
#         log_likelihood = self._univariate_log_likelihood(y_tar, mu, sigma)   # [batch, target]
#         log_likelihood = torch.mean(log_likelihood)                             # [1]
#         return log_likelihood
        raise NotImplementedError

    def multivariate_log_likelihood(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        y_tar:    Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
    ) -> Float:
        """
        Calculate multivariate log-likelihood.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]

        Returns:
            log_likelihood: float
        """

#         mu, cov = self(x_ctx, y_ctx, x_tar, as_univariate=False)     # [batch, y_dim, target], [batch, y_dim, target, target]
#         log_likelihood = self._multivariate_log_likelihood(y_tar, mu, cov)   # [batch, target]
#         log_likelihood = torch.mean(log_likelihood)                             # [1]
#         return log_likelihood
        raise NotImplementedError

    # Losses

    def loss(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        y_tar:    Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
        as_univariate: bool = False,
    ) -> Float:
        """
        Calculate loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]
            as_univariate: bool

        Returns:
            loss: float
        """

        if as_univariate:
            return self.univariate_loss(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        else:
            return self.multivariate_loss(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)

    def univariate_loss(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        y_tar:    Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
    ) -> Float:
        """
        Calculate univariate loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]

        Returns:
            loss: float
        """

        loss = -self.univariate_log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        return loss

    def multivariate_loss(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        y_tar:    Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
    ) -> Float:
        """
        Calculate multivariate loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[batch, context]
            mask_tar: Array[batch,  target]

        Returns:
            loss: float
        """

        loss = -self.multivariate_log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        return loss
