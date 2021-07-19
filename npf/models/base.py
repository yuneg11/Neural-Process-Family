from typing import Tuple
from torchtyping import TensorType

import abc

from torch import nn


__all__ = [
    "NPF",
    "ConditionalNPF",
    "LatentNPF",
]


class NPF(nn.Module):
    pass


class ConditionalNPF(NPF):
    @abc.abstractmethod
    def forward(
        self,
        x_context: TensorType["batch", "context", "x_dim"],
        y_context: TensorType["batch", "context", "y_dim"],
        x_target:  TensorType["batch", "target",  "x_dim"],
    ) -> Tuple[
        TensorType["batch", "target", "y_dim"],
        TensorType["batch", "target", "y_dim"]
    ]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch, target,  x_dim]

        Returns:
            mu:    Tensor[batch, target, y_dim]
            sigma: Tensor[batch, target, y_dim]
        """

    @abc.abstractmethod
    def log_likelihood(
        self,
        x_context: TensorType["batch", "context", "x_dim"],
        y_context: TensorType["batch", "context", "y_dim"],
        x_target:  TensorType["batch", "target",  "x_dim"],
        y_target:  TensorType["batch", "target",  "y_dim"],
    ) -> float:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch, target,  x_dim]
            y_target:  Tensor[batch, target,  y_dim]

        Returns:
            log_likelihood: float
        """

    @abc.abstractmethod
    def loss(
        self,
        x_context: TensorType["batch", "context", "x_dim"],
        y_context: TensorType["batch", "context", "y_dim"],
        x_target:  TensorType["batch", "target",  "x_dim"],
        y_target:  TensorType["batch", "target",  "y_dim"],
    ) -> float:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch, target,  x_dim]
            y_target:  Tensor[batch, target,  y_dim]

        Returns:
            log_likelihood: float
        """


class LatentNPF(NPF):
    @abc.abstractmethod
    def forward(
        self,
        x_context: TensorType["batch", "context", "x_dim"],
        y_context: TensorType["batch", "context", "y_dim"],
        x_target:  TensorType["batch", "target",  "x_dim"],
        num_latents: int = 1,
    ) -> Tuple[
        TensorType["batch", "latent", "target", "y_dim"],
        TensorType["batch", "latent", "target", "y_dim"]
    ]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch, target,  x_dim]
            num_latents: int

        Returns:
            mu:    Tensor[batch, target, y_dim]
            sigma: Tensor[batch, target, y_dim]
        """

    @abc.abstractmethod
    def log_likelihood(self, x_context, y_context, x_target, y_target, num_latents):
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, x_context, y_context, x_target, y_target, num_latents):
        raise NotImplementedError
