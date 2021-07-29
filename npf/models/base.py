from torchtyping import TensorType
from ..type import *

import abc

from torch import nn


__all__ = [
    "NPF",
    "ConditionalNPF",
    "LatentNPF",
]


class NPF(nn.Module):
    @property
    def num_params(self) -> int:
        return sum([parameter.numel() for parameter in self.parameters()])


class ConditionalNPF(NPF):
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
            x_target:  Tensor[batch,  target, x_dim]

        Returns:
            mu:    Tensor[batch, target, y_dim]
            sigma: Tensor[batch, target, y_dim]
        """

    @abc.abstractmethod
    def log_likelihood(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
    ) -> TensorType[float]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch,  target, x_dim]
            y_target:  Tensor[batch,  target, y_dim]

        Returns:
            log_likelihood: Tensor[float]
        """

    def loss(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
    ) -> TensorType[float]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch,  target, x_dim]
            y_target:  Tensor[batch,  target, y_dim]

        Returns:
            loss: Tensor[float]
        """

        log_likelihood = self.log_likelihood(x_context, y_context, x_target, y_target)
        loss = -log_likelihood
        return loss

    @property
    def is_latent_model(self):
        return False


class LatentNPF(NPF):
    def __init__(self,
        loss_type: str = "vi",
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
            x_target:  Tensor[batch,  target, x_dim]
            num_latents: int

        Returns:
            mu:    Tensor[batch, latent, target, y_dim]
            sigma: Tensor[batch, latent, target, y_dim]
        """

    @abc.abstractmethod
    def log_likelihood(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
        num_latents: int = 1,
    ) -> TensorType[float]:
        """
        Args:
            x_context: Tensor[batch, context, x_dim]
            y_context: Tensor[batch, context, y_dim]
            x_target:  Tensor[batch,  target, x_dim]
            y_target:  Tensor[batch,  target, y_dim]
            num_latents: int

        Returns:
            log_likelihood: Tensor[float]
        """

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
            x_target:  Tensor[batch,  target, x_dim]
            y_target:  Tensor[batch,  target, y_dim]
            num_latents: int

        Returns:
            log_likelihood: Tensor[float]
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
            x_target:  Tensor[batch,  target, x_dim]
            y_target:  Tensor[batch,  target, y_dim]
            num_latents: int

        Returns:
            loss: Tensor[float]
        """

        log_likelihood = self.log_likelihood(x_context, y_context, x_target, y_target, num_latents)
        loss = -log_likelihood
        return loss

    @property
    def is_latent_model(self):
        return True
