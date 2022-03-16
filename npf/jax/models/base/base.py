from flax import linen as nn


__all__ = [
    "NPF",
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
