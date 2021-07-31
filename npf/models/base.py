from ..type import *

import abc
import math

import torch
from torch import nn
from torch.distributions import (
    Normal,
    MultivariateNormal,
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

    is_multivariate_model: bool
    is_latent_model: bool

    @property
    def num_params(self) -> int:
        return sum([parameter.numel() for parameter in self.parameters()])


class UnivariateNPF(NPF):
    """
    Base class for univariate NPF models
    """

    is_multivariate_model: bool = False
    is_latent_model: bool


class ConditionalNPF(UnivariateNPF):
    """
    Base class for conditional NPF models
    """

    is_latent_model: bool = False

    # Static method

    @staticmethod
    def _log_likelihood(
        y_target: TensorType[B, T, Y],
        mu:       TensorType[B, T, Y],
        sigma:    TensorType[B, T, Y],
    ) -> TensorType[B, T]:

        distribution = Normal(mu, sigma)                                        # [batch, target, y_dim]
        log_prob = distribution.log_prob(y_target)                              # [batch, target, y_dim]
        log_likelihood = torch.sum(log_prob, dim=-1)                            # [batch, target]
        return log_likelihood

    # Forward

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

    # Likelihood

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

    # Loss

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

        log_likelihood = \
            self.log_likelihood(x_context, y_context, x_target, y_target)
        loss = -log_likelihood
        return loss


class LatentNPF(UnivariateNPF):
    """
    Base class for latent NPF models
    """

    is_latent_model: bool = True

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

    # Static methods

    @staticmethod
    def _log_likelihood(
        y_target: TensorType[B, T, Y],
        mu:       TensorType[B, L, T, Y],
        sigma:    TensorType[B, L, T, Y],
    ) -> TensorType[B, L, T]:

        y_target = y_target[:, None, :, :]                                      # [batch, 1, target, y_dim]
        distribution = Normal(mu, sigma)                                        # [batch, latent, target, y_dim]
        log_prob = distribution.log_prob(y_target)                              # [batch, latent, target, y_dim]
        log_likelihood = torch.sum(log_prob, dim=-1)                            # [batch, latent, target]
        return log_likelihood

    @staticmethod
    def _kl_divergence(
        z_data:    TensorType[...],
        z_context: TensorType[...],
    ) -> TensorType[...]:

        kl_div = kl_divergence(z_data, z_context)
        return kl_div

    # Forward

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

    # Likelihood

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

        log_likelihood = self._log_likelihood(y_target, mu, sigma)              # [batch, latent, target]
        log_likelihood = torch.mean(log_likelihood, dim=-1)                     # [batch, latent]
        log_likelihood = torch.logsumexp(log_likelihood, dim=-1) \
                       - math.log(num_latents)                                  # [batch]
        log_likelihood = torch.mean(log_likelihood)                             # [1]

        return log_likelihood

    # Losses

    def loss(self,
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
        raise NotImplementedError  # The implementation is chosen from __init__

    @abc.abstractmethod
    def vi_loss(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
        num_latents: int = 1,
    ) -> TensorType[float]:
        pass

    def ml_loss(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
        num_latents: int = 1,
    ) -> TensorType[float]:

        log_likelihood = \
            self.log_likelihood(x_context, y_context, x_target, y_target, num_latents)
        loss = -log_likelihood
        return loss


class MultivariateNPF(NPF):
    """
    Base class for multivariate NPF models
    """

    is_multivariate_model: bool = True
    is_latent_model: bool = False

    def __init__(self,
        likelihood_type: str = "multivariate",
        loss_type: str = "multivariate",
    ):
        super().__init__()

        self.likelihood_type = (likelihood_type if likelihood_type else "multivariate")
        self.loss_type = (loss_type if loss_type else "multivariate")

        if likelihood_type == "multivariate":
            self.log_likelihood = self.multivariate_log_likelihood
        elif likelihood_type == "univariate":
            self.log_likelihood = self.univariate_log_likelihood
        else:
            raise ValueError(f"Invalid likelihood type: '{likelihood_type}'")

        if loss_type == "multivariate":
            self.loss = self.multivariate_loss
        elif loss_type == "univariate":
            self.loss = self.univariate_loss
        else:
            raise ValueError(f"Invalid loss type: '{loss_type}'")

    # Static methods

    @staticmethod
    def _univariate_log_likelihood(
        y_target: TensorType[B, T, Y],
        mu:       TensorType[B, T, Y],
        sigma:    TensorType[B, T, Y],
    ) -> TensorType[B, T]:

        distribution = Normal(mu, sigma)                                        # Normal[batch, target, y_dim]
        log_prob = distribution.log_prob(y_target)                              # [batch, target, y_dim]
        log_likelihood = torch.sum(log_prob, dim=-1)                            # [batch, target]
        return log_likelihood

    @staticmethod
    def _multivariate_log_likelihood(
        y_target: TensorType[B, T, Y],
        mu:       TensorType[B, Y, T],
        cov:      TensorType[B, Y, T, T],
    ) -> TensorType[B]:

        num_target = y_target.shape[1]

        y_target = y_target.transpose(1, 2)                                     # [batch, y_dim, target]
        distribution = MultivariateNormal(mu, cov)                              # MultivariateNormal[batch, y_dim, target, target]
        log_prob = distribution.log_prob(y_target)                              # [batch, y_dim]
        log_likelihood = torch.sum(log_prob, dim=1) / num_target                # [batch]
        return log_likelihood

    # Forward

    @abc.abstractmethod
    def forward(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
        as_univariate: bool = False,
    ) -> Union[
        Tuple[TensorType[B, Y, T], TensorType[B, Y, T, T]],
        Tuple[TensorType[B, T, Y], TensorType[B, T, Y]],
    ]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch, target, x_dim]
            as_univariate: bool

        Returns:
            mu:  Tensor[batch, y_dim, target]          if as_univariate == False
                 Tensor[batch, target, y_dim]          if as_univariate == True
            cov: Tensor[batch, y_dim, target, target]  if as_univariate == False
                 Tensor[batch, target, y_dim]          if as_univariate == True
        """

    # Likelihoods

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
        raise NotImplementedError  # The implementation is chosen from __init__

    def univariate_log_likelihood(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
    ) -> TensorType[float]:

        mu, sigma = self(x_context, y_context, x_target, as_univariate=True)    # [batch, target, y_dim] * 2
        log_likelihood = self._univariate_log_likelihood(y_target, mu, sigma)   # [batch, target]
        log_likelihood = torch.mean(log_likelihood)                             # [1]
        return log_likelihood

    def multivariate_log_likelihood(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
    ) -> TensorType[float]:

        mu, cov = self(x_context, y_context, x_target, as_univariate=False)     # [batch, y_dim, target], [batch, y_dim, target, target]
        log_likelihood = self._multivariate_log_likelihood(y_target, mu, cov)   # [batch, target]
        log_likelihood = torch.mean(log_likelihood)                             # [1]
        return log_likelihood

    # Losses

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
        raise NotImplementedError  # The implementation is chosen from __init__

    def univariate_loss(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
    ) -> TensorType[float]:

        log_likelihood = \
            self.univariate_log_likelihood(x_context, y_context, x_target, y_target)
        loss = -log_likelihood
        return loss

    def multivariate_loss(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
    ) -> TensorType[float]:

        log_likelihood = \
            self.multivariate_log_likelihood(x_context, y_context, x_target, y_target)
        loss = -log_likelihood
        return loss
