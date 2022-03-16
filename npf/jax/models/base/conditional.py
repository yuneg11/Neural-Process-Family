from ...type import *

from .univariate import UnivariateNPF
from ... import functional as F


__all__ = [
    "ConditionalNPF",
]


class ConditionalNPF(UnivariateNPF):
    """
    Base class for conditional NPF models
    """

    @property
    def is_latent_model(self) -> bool:
        return False

    def log_likelihood(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        **kwargs,
    ) -> Array:
        """
        Calculate log-likelihood.
        """

        mask_axis = [0] + [-d for d in range(1, mask_tar.ndim)]
        mu, sigma = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, **kwargs)                         # [batch, ..., *target, y_dim] x 2
        _log_likelihood = self._log_likelihood(y_tar, mu, sigma)                                    # [batch, ..., *target]
        log_likelihood = F.masked_mean(_log_likelihood, mask_tar, mask_axis=mask_axis)              # [1]
        return log_likelihood

    def loss(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        **kwargs,
    ) -> Array:
        """
        Calculate loss.
        """

        loss = -self.log_likelihood(x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar, **kwargs)
        return loss
