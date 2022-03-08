from ..type import *

import math
from abc import abstractmethod

import jax
from jax import numpy as jnp
from jax.scipy import stats
from flax import linen as nn

from .. import functional as F


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
        y:     Array[..., T, Y],
        mu:    Array[..., T, Y],
        sigma: Array[..., T, Y],
    ) -> Array[..., T]:
        """
        Calculate element-wise log-likelihood of Gaussian distribution.

        Args:
            y_tar: Array[batch, target, y_dim]
            mu:    Array[batch, target, y_dim]
            sigma: Array[batch, target, y_dim]

        Returns:
            log_likelihood: Array[batch, target]
        """

        log_prob = stats.norm.logpdf(y, loc=mu, scale=sigma)                    # [..., target, y_dim]
        log_likelihood = jnp.sum(log_prob, axis=-1)                             # [..., target]
        return log_likelihood

    # Forward

    @abstractmethod
    def __call__(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        **kwargs,
    ) -> Tuple[Array[B, ..., T, Y], Array[B, ..., T, Y]]:
        """
        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]

        Returns:
            mu:       Array[batch, target, y_dim]
            sigma:    Array[batch, target, y_dim]
        """

        raise NotImplementedError

    # Likelihood

    def log_likelihood(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        **kwargs,
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

        Returns:
            log_likelihood: Array
        """

        mu, sigma = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, **kwargs)     # [..., target, y_dim] x 2
        _log_likelihood = self._log_likelihood(y_tar, mu, sigma)                # [..., target]
        _log_likelihood = F.masked_mean(_log_likelihood, mask_tar, axis=-1)     # [...]
        log_likelihood = jnp.mean(_log_likelihood)                              # [1]
        return log_likelihood

    # Loss

    def loss(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        **kwargs,
    ) -> Array:
        """
        Calculate loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]

        Returns:
            loss      float
        """

        loss = -self.log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, **kwargs)
        return loss


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
        if self.loss_type not in ["vi", "ml"]:
            raise ValueError(f"Invalid loss_type: {self.loss_type}. loss_type must be either 'vi' or 'ml'.")

    # Static methods

    @staticmethod
    def _log_likelihood(
        y:     Array[..., T, Y],
        mu:    Array[..., L, T, Y],
        sigma: Array[..., L, T, Y],
    ) -> Array[..., T]:
        """
        Calculate element-wise log-likelihood of Gaussian distribution.

        Args:
            y_tar: Array[batch, target, y_dim]
            mu:    Array[batch, latent, target, y_dim]
            sigma: Array[batch, latent, target, y_dim]

        Returns:
            log_likelihood: Array[batch, latent, target]
        """

        y = jnp.expand_dims(y, axis=-3)                                         # [..., 1, target, y_dim]
        log_prob = stats.norm.logpdf(y, loc=mu, scale=sigma)                    # [..., latent, target, y_dim]
        log_likelihood = jnp.sum(log_prob, axis=-1)                             # [..., latent, target]
        return log_likelihood

    @staticmethod
    def _kl_divergence(
        z0_mu:    Array[...],
        z0_sigma: Array[...],
        z1_mu:    Array[...],
        z1_sigma: Array[...],
    ) -> Array[...]:
        """
        Calculate element-wise KL divergence between two Gaussian distributions.

        Args:
            z0_mu:    Array[...]
            z0_sigma: Array[...]
            z1_mu:    Array[...]
            z1_sigma: Array[...]

        Returns:
            kl_divergence: Array[...]
        """

        kl_div = jnp.log(z1_sigma) - jnp.log(z0_sigma) \
               + (jnp.square(z0_sigma) + jnp.square(z0_mu - z1_mu)) / (2 * jnp.square(z1_sigma)) \
               - 0.5
        return kl_div

    # Forward

    @abstractmethod
    def __call__(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_latents: int = 1,
        **kwargs,
    ) -> Tuple[Array[B, L, T, Y], Array[B, L, T, Y]]:
        """
        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]
            num_latents: int

        Returns:
            mu:       Array[batch, latent, target, y_dim]
            sigma:    Array[batch, latent, target, y_dim]
        """

    # Likelihood

    def log_likelihood(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_latents: int = 1,
        **kwargs,
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
            num_latents: int

        Returns:
            log_likelihood: float
        """

        mu, sigma = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_latents, **kwargs)  # [batch, latent, target, y_dim] x 2

        log_likelihood = self._log_likelihood(y_tar, mu, sigma)                 # [batch, latent, target]
        log_likelihood = F.masked_mean(log_likelihood, mask_tar, axis=-1)       # [batch, latent]
        log_likelihood = jax.nn.logsumexp(log_likelihood, axis=-1) \
                       - math.log(num_latents)                                  # [batch]
        log_likelihood = jnp.mean(log_likelihood)                               # [1]

        return log_likelihood

    # Losses

    def loss(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_latents: int = 1,
        **kwargs,
    ) -> Array:
        """
        Calculate loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]
            num_latents: int

        Returns:
            loss: float
        """
        if self.loss_type == "vi":
            return self.vi_loss(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, num_latents, **kwargs)
        elif self.loss_type == "ml":
            return self.ml_loss(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, num_latents, **kwargs)

    @abstractmethod
    def vi_loss(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_latents: int = 1,
        **kwargs,
    ) -> Array:
        """
        Calculate VI loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]
            num_latents: int

        Returns:
            loss: float
        """
        raise NotImplementedError

    def ml_loss(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[C],
        mask_tar: Array[T],
        num_latents: int = 1,
        **kwargs,
    ) -> Array:
        """
        Calculate Maximum-Likelihood loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]
            num_latents: int

        Returns:
            loss: float
        """

        loss = -self.log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, num_latents, **kwargs)
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
        y_tar:    Array[B, T, Y],
        mu:       Array[B, T, Y],
        sigma:    Array[B, T, Y],
    ) -> Array[B, T]:
        """
        Calculate element-wise log-likelihood of univariate Gaussian distribution.

        Args:
            y_tar:    Array[batch, target, y_dim]
            mu:       Array[batch, target, y_dim]
            sigma:    Array[batch, target, y_dim]

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
        y_tar:    Array[B, T, Y],
        mu:       Array[B, Y, T],
        cov:      Array[B, Y, T, T],
        mask_tar: Array[B, T],
    ) -> Array:   # TODO: Fix type annotation "Array[B]"
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
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[B, C],
        mask_tar: Array[B, T],
        as_univariate: bool = False,
    ) -> Union[
        Tuple[Array[B, Y, T], Array[B, Y, T, T]],
        Tuple[Array[B, T, Y], Array[B, T, Y]],
    ]:
        """
        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]
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
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[B, C],
        mask_tar: Array[B, T],
        as_univariate: bool = False,
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
            as_univariate: bool

        Returns:
            log_likelihood: Array
        """

        if as_univariate:
            return self.univariate_log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        else:
            return self.multivariate_log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)

    def univariate_log_likelihood(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[B, C],
        mask_tar: Array[B, T],
    ) -> Array:
        """
        Calculate univariate log-likelihood.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]

        Returns:
            log_likelihood: Array
        """

#         mu, sigma = self(x_ctx, y_ctx, x_tar, as_univariate=True)    # [batch, target, y_dim] * 2
#         log_likelihood = self._univariate_log_likelihood(y_tar, mu, sigma)   # [batch, target]
#         log_likelihood = torch.mean(log_likelihood)                             # [1]
#         return log_likelihood
        raise NotImplementedError

    def multivariate_log_likelihood(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[B, C],
        mask_tar: Array[B, T],
    ) -> Array:
        """
        Calculate multivariate log-likelihood.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]

        Returns:
            log_likelihood: Array
        """

#         mu, cov = self(x_ctx, y_ctx, x_tar, as_univariate=False)     # [batch, y_dim, target], [batch, y_dim, target, target]
#         log_likelihood = self._multivariate_log_likelihood(y_tar, mu, cov)   # [batch, target]
#         log_likelihood = torch.mean(log_likelihood)                             # [1]
#         return log_likelihood
        raise NotImplementedError

    # Losses

    def loss(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[B, C],
        mask_tar: Array[B, T],
        as_univariate: bool = False,
    ) -> Array:
        """
        Calculate loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]
            as_univariate: bool

        Returns:
            loss: Array
        """

        if as_univariate:
            return self.univariate_loss(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        else:
            return self.multivariate_loss(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)

    def univariate_loss(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[B, C],
        mask_tar: Array[B, T],
    ) -> Array:
        """
        Calculate univariate loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]

        Returns:
            loss: Array
        """

        loss = -self.univariate_log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        return loss

    def multivariate_loss(self,
        x_ctx:    Array[B, C, X],
        y_ctx:    Array[B, C, Y],
        x_tar:    Array[B, T, X],
        y_tar:    Array[B, T, Y],
        mask_ctx: Array[B, C],
        mask_tar: Array[B, T],
    ) -> Array:
        """
        Calculate multivariate loss.

        Args:
            x_ctx:    Array[batch, context, x_dim]
            y_ctx:    Array[batch, context, y_dim]
            x_tar:    Array[batch,  target, x_dim]
            y_tar:    Array[batch,  target, y_dim]
            mask_ctx: Array[context]
            mask_tar: Array[target]

        Returns:
            loss: Array
        """

        loss = -self.multivariate_log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        return loss
