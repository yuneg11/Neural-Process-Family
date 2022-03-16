from ...type import *

from abc import abstractmethod

from jax import numpy as jnp
from jax.scipy import stats

from .base import NPF


__all__ = [
    "MultivariateNPF",
]


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

    @staticmethod
    def _univariate_log_likelihood(
        y_tar:    Array[B, T, Y],
        mu:       Array[B, T, Y],
        sigma:    Array[B, T, Y],
    ) -> Array[B, T]:
        """
        Calculate element-wise log-likelihood of univariate Gaussian distribution.
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
        """

#         num_target = y_tar.shape[1]

#         y_tar = y_tar.transpose(1, 2)                                     # [batch, y_dim, target]
#         distribution = MultivariateNormal(mu, cov)                              # MultivariateNormal[batch, y_dim, target, target]
#         log_prob = distribution.log_prob(y_tar)                              # [batch, y_dim]
#         log_likelihood = torch.sum(log_prob, dim=1) / num_target                # [batch]
#         return log_likelihood
        raise NotImplementedError

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
        """

#         mu, cov = self(x_ctx, y_ctx, x_tar, as_univariate=False)     # [batch, y_dim, target], [batch, y_dim, target, target]
#         log_likelihood = self._multivariate_log_likelihood(y_tar, mu, cov)   # [batch, target]
#         log_likelihood = torch.mean(log_likelihood)                             # [1]
#         return log_likelihood
        raise NotImplementedError

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
        """

        loss = -self.multivariate_log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        return loss
