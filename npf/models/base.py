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
    def forward(self, x_context, y_context, x_target):
        """
        Args:
            x_context: [batch, context, x_dim]
            y_context: [batch, context, y_dim]
            x_target:  [batch, target,  x_dim]

        Returns:
            mu:    [batch, target, y_dim]
            sigma: [batch, target, y_dim]
        """

    @abc.abstractmethod
    def log_likelihood(self, x_context, y_context, x_target, y_target):
        """
        Args:
            x_context: [batch, context, x_dim]
            y_context: [batch, context, y_dim]
            x_target:  [batch, target,  x_dim]
            y_target:  [batch, target,  y_dim]

        Returns:
            log_likelihood: float
        """

    @abc.abstractmethod
    def loss(self, x_context, y_context, x_target, y_target):
        """
        Args:
            x_context: [batch, context, x_dim]
            y_context: [batch, context, y_dim]
            x_target:  [batch, target,  x_dim]
            y_target:  [batch, target,  y_dim]

        Returns:
            log_likelihood: float
        """


class LatentNPF(NPF):
    @abc.abstractmethod
    def forward(self, x_context, y_context, x_target, num_latents):
        raise NotImplementedError

    @abc.abstractmethod
    def log_likelihood(self, x_context, y_context, x_target, y_target, num_latents):
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, x_context, y_context, x_target, y_target, num_latents):
        raise NotImplementedError
