from ..typing import *

from flax import linen as nn

from ..data import NPData
from ..utils import npf_io


__all__ = [
    "NPF",
]


class NPF(nn.Module):
    """
    Base class for NPFs
    """

    @npf_io
    def __call__(
        self,
        data: NPData,
        **kwargs,
    ) -> Union[
        Tuple[Array[B, [T], Y], Array[B, [T], Y]],
        Tuple[Array[B, [T], Y], Array[B, [T], Y], Any],
        Tuple[Array[B, [M], [T], Y], Array[B, [M], [T], Y]],
        Tuple[Array[B, [M], [T], Y], Array[B, [M], [T], Y], Any],
    ]:
        """Forward pass

        Args:
            data (NPData): NPData object
            **kwargs: kwargs

        Returns:
            (mu, sigma) or (mu, sigma, aux)
        """
        raise NotImplementedError("NPFs must implement the '__call__' method")

    @npf_io
    def log_likelihood(
        self,
        data: NPData,
        *,
        split_set: bool = False,
        **kwargs,
    ) -> Union[
        Array,
        Tuple[Array, Any],
        Tuple[Array, Array, Array],
        Tuple[Array, Array, Array, Any],
    ]:
        """Log-Likelihood

        Args:
            data (NPData): NPData object
            split_set (bool): If True, return log-likelihood of each set separately
            **kwargs: kwargs

        Returns:
            ll or (ll, aux) if split_set is False
            (ll, ll_ctx, ll_tar) or (ll, ll_ctx, ll_tar, aux) if split_set is True
        """
        raise NotImplementedError("NPFs must implement the 'log_likelihood' method")

    @npf_io
    def loss(
        self,
        data: NPData,
        **kwargs,
    ) -> Union[
        Array,
        Tuple[Array, Dict],
    ]:
        """Loss

        Args:
            data (NPData): NPData object
            **kwargs: kwargs

        Returns:
            loss or (loss, aux)
        """
        raise NotImplementedError("NPFs must implement the 'loss' method")
