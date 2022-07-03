from ..typing import *

from flax import linen as nn

__all__ = [
    "NPF",
]

class NPF(nn.Module):
    """
    Base class for NPFs
    """

    def __call__(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        **kwargs,
    ) -> Union[
        Tuple[Array[B, [T], Y], Array[B, [T], Y]],
        Tuple[Array[B, [T], Y], Array[B, [T], Y], Any],
        Tuple[Array[B, [M], [T], Y], Array[B, [M], [T], Y]],
        Tuple[Array[B, [M], [T], Y], Array[B, [M], [T], Y], Any],
    ]:
        """Forward pass

        Args:
            x_ctx    (Array[B, [C], X]): x_context
            y_ctx    (Array[B, [C], Y]): y_context
            x_tar    (Array[B, [T], X]): x_target
            mask_ctx (Array[B, [C]]): mask_context
            mask_tar (Array[B, [T]]): mask_target
            **kwargs: kwargs

        Returns:
            (mu_tar, sigma_tar) or (mu_tar, sigma_tar, aux)
        """
        raise NotImplementedError("NPFs must implement the '__call__' method")

    def log_likelihood(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        **kwargs,
    ) -> Array:
        """Log-Likelihood

        Args:
            x_ctx    (Array[B, [C], X]): x_context
            y_ctx    (Array[B, [C], Y]): y_context
            x_tar    (Array[B, [T], X]): x_target
            y_tar    (Array[B, [T], Y]): y_target
            mask_ctx (Array[B, [C]]): mask_context
            mask_tar (Array[B, [T]]): mask_target
            **kwargs: kwargs

        Returns:
            ll
        """
        raise NotImplementedError("NPFs must implement the 'log_likelihood' method")

    def loss(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        **kwargs,
    ) -> Array:
        """Loss

        Args:
            x_ctx    (Array[B, [C], X]): x_context
            y_ctx    (Array[B, [C], Y]): y_context
            x_tar    (Array[B, [T], X]): x_target
            y_tar    (Array[B, [T], Y]): y_target
            mask_ctx (Array[B, [C]]): mask_context
            mask_tar (Array[B, [T]]): mask_target
            **kwargs: kwargs

        Returns:
            loss
        """
        raise NotImplementedError("NPFs must implement the 'loss' method")
