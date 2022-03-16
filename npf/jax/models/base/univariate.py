from ...type import *

from abc import abstractmethod

from jax import numpy as jnp
from jax.scipy import stats

from .base import NPF


__all__ = [
    "UnivariateNPF",
]


class UnivariateNPF(NPF):
    """
    Base class for univariate NPF models
    """

    @property
    def is_multivariate_model(self) -> bool:
        return False

    @staticmethod
    def _log_likelihood(
        y:     Array[B, ..., Y],
        mu:    Array[B, ..., Y],
        sigma: Array[B, ..., Y],
    ) -> Array[B, ...]:
        """
        Calculate element-wise log-likelihood of Gaussian distribution.
        """

        log_prob = stats.norm.logpdf(y, loc=mu, scale=sigma)                    # [..., y_dim]
        log_likelihood = jnp.sum(log_prob, axis=-1)                             # [...]
        return log_likelihood

    @abstractmethod
    def __call__(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        **kwargs,
    ) -> Tuple[Array[B, ..., [T], Y], Array[B, ..., [T], Y]]:
        """
        """

        raise NotImplementedError
