from ..type import *

import abc
import math

import torch
from torch import nn
from torch.distributions import (
    Normal,
    kl_divergence,
)


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
    def num_params(self) -> int:
        return sum([parameter.numel() for parameter in self.parameters()])


class UnivariateNPF(NPF):
    """
    Base class for univariate NPF models
    """

    @staticmethod
    def _log_likelihood(
        y_target: Union[TensorType[B, T, Y], TensorType[B, 1, T, Y]],
        mu:       Union[TensorType[B, T, Y], TensorType[B, L, T, Y]],
        sigma:    Union[TensorType[B, T, Y], TensorType[B, L, T, Y]],
    ) -> Union[TensorType[B, T], TensorType[B, L, T]]:

        distribution = Normal(mu, sigma)                                        # [batch, (latent,) target, y_dim]
        log_prob = distribution.log_prob(y_target)                              # [batch, (latent,) target, y_dim]
        log_likelihood = torch.sum(log_prob, dim=-1)                            # [batch, (latent,) target]
        return log_likelihood

    @staticmethod
    def _kl_divergence(
        z_data:    TensorType[...],
        z_context: TensorType[...],
    ) -> TensorType[...]:

        kl_div = kl_divergence(z_data, z_context)
        return kl_div


class ConditionalNPF(UnivariateNPF):
    """
    Base class for conditional NPF models
    """

    is_latent_model = False

    @abc.abstractmethod
    def forward(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
    ) -> Tuple[TensorType[B, T, Y], TensorType[B, T, Y]]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch, target, x_dim]

        Returns:
            mu:    Tensor[batch, target, y_dim]
            sigma: Tensor[batch, target, y_dim]
        """

    def log_likelihood(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
    ) -> TensorType[float]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch, target, x_dim]
            y_target:  Tensor[batch, target, y_dim]

        Returns:
            log_likelihood: Tensor[float]
        """

        mu, sigma = self(x_context, y_context, x_target)                        # [batch, target, y_dim] * 2
        log_likelihood = self._log_likelihood(y_target, mu, sigma)              # [batch, target]
        log_likelihood = torch.mean(log_likelihood)                             # [1]
        return log_likelihood

    def loss(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
    ) -> TensorType[float]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch, target, x_dim]
            y_target:  Tensor[batch, target, y_dim]

        Returns:
            loss: Tensor[float]
        """

        log_likelihood = self.log_likelihood(
            x_context, y_context,
            x_target, y_target
        )
        loss = -log_likelihood
        return loss


class LatentNPF(UnivariateNPF):
    """
    Base class for latent NPF models
    """

    is_latent_model = True

    def __init__(self,
        loss_type: str,
    ):
        super().__init__()

        self.loss_type = loss_type

        if loss_type == "vi":
            self.loss = self.vi_loss
        elif loss_type == "ml":
            self.loss = self.ml_loss
        else:
            raise ValueError(f"Invalid loss type: '{loss_type}'")

    @abc.abstractmethod
    def forward(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
        num_latents: int = 1,
    ) -> Tuple[TensorType[B, L, T, Y], TensorType[B, L, T, Y]]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch, target, x_dim]
            num_latents: int

        Returns:
            mu:    Tensor[batch, latent, target, y_dim]
            sigma: Tensor[batch, latent, target, y_dim]
        """

    def log_likelihood(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
        num_latents: int = 1,
    ) -> TensorType[float]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch, target, x_dim]
            y_target:  Tensor[batch, target, y_dim]
            num_latents: int

        Returns:
            log_likelihood: Tensor[float]
        """

        mu, sigma = self(x_context, y_context, x_target, num_latents)           # [batch, latent, target, y_dim] * 2
        y_target = y_target.unsqueeze(dim=1)                                    # [batch, 1, target, y_dim]

        log_likelihood = self._log_likelihood(y_target, mu, sigma)              # [batch, latent, target]
        log_likelihood = torch.mean(log_likelihood, dim=-1)                     # [batch, latent]
        log_likelihood = torch.logsumexp(log_likelihood, dim=-1) \
                       - math.log(num_latents)                                  # [batch]
        log_likelihood = torch.mean(log_likelihood)                             # [1]

        return log_likelihood

    def loss(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
        num_latents: int = 1,
    ) -> TensorType[float]:
        raise NotImplementedError  # Implementation is choosed at __init__

    @abc.abstractmethod
    def vi_loss(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
        num_latents: int = 1,
    ) -> TensorType[float]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch, target, x_dim]
            y_target:  Tensor[batch, target, y_dim]
            num_latents: int

        Returns:
            loss: Tensor[float]
        """

    def ml_loss(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
        num_latents: int = 1,
    ) -> TensorType[float]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch, target, x_dim]
            y_target:  Tensor[batch, target, y_dim]
            num_latents: int

        Returns:
            loss: Tensor[float]
        """

        log_likelihood = self.log_likelihood(
            x_context, y_context,
            x_target, y_target,
            num_latents
        )
        loss = -log_likelihood
        return loss


class MultivariateNPF(NPF):
    """
    Base class for multivariate NPF models
    """
