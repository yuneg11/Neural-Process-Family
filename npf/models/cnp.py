import torch
from torch.nn import functional as F

from .base import ConditionalNPF

from ..modules import (
    PointwiseMLP,
    LogLikelihood,
)


__all__ = ["CNPBase", "CNP"]


class CNPBase(ConditionalNPF):
    def __init__(self, encoder, decoder):
        """Conditional Neural Process model

        Args:
            encoder (PointwiseEncoder): [batch, context, x_dim + y_dim] -> [batch, context, r_dim]
            decoder (PointwiseDecoder): [batch, target, x_dim + r_dim] -> [batch, target, y_dim * 2]
            loss_fn ([type]): [description]
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.ll_fn = LogLikelihood()

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

        context = torch.cat((x_context, y_context), dim=-1)                     # [batch, context, x_dim + y_dim]
        pointwise_rep = self.encoder(context)                                   # [batch, context, r_dim]

        representation = torch.mean(pointwise_rep, dim=1, keepdim=True)         # [batch, 1, r_dim]
        representation = representation.repeat(1, x_target.shape[1], 1)         # [batch, target, r_dim]

        query = torch.cat((x_target, representation), dim=-1)                   # [batch, target, x_dim + r_dim]
        mu_log_sigma = self.decoder(query)                                      # [batch, target, y_dim * 2]

        y_dim = mu_log_sigma.shape[-1] // 2
        mu, log_sigma = torch.split(mu_log_sigma, (y_dim, y_dim), dim=-1)       # [batch, target, y_dim] * 2
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        return mu, sigma

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

        mu, sigma = self(x_context, y_context, x_target)
        log_likelihood = self.ll_fn(y_target, mu, sigma)

        return log_likelihood

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

        log_likelihood = self.log_likelihood(
            x_context, y_context, x_target, y_target,
        )
        loss = -log_likelihood

        return loss


class CNP(CNPBase):
    def __init__(
        self,
        x_dim,
        y_dim,
        r_dim,
        encoder_dims=[128, 128, 128, 128, 128, 128],
        decoder_dims=[128, 128, 128, 128],
    ):
        encoder = PointwiseMLP(
            in_features=(x_dim + y_dim),
            hidden_features=encoder_dims,
            out_features=r_dim,
        )

        decoder = PointwiseMLP(
            in_features=(x_dim + r_dim),
            hidden_features=decoder_dims,
            out_features=(y_dim * 2),
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
        )
