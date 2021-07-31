from ..type import *

import torch
from torch.nn import functional as F

from .base import ConditionalNPF

from ..modules import (
    MLP,
)


__all__ = ["CNPBase", "CNP"]


class CNPBase(ConditionalNPF):
    """
    Base class of Conditional Neural Process
    """

    def __init__(self,
        encoder,
        decoder,
    ):
        """
        Args:
            encoder : [batch, context, x_dim + y_dim]
                   -> [batch, context, r_dim]
            decoder : [batch, target, x_dim + r_dim]
                   -> [batch, target, y_dim * 2]
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def _aggregate(self,
        r_i_context: TensorType[B, C, R],
        x_context:   TensorType[B, C, X],
        x_target:    TensorType[B, T, X],
    ) -> TensorType[B, T, R]:

        r_context = torch.mean(r_i_context, dim=1, keepdim=True)                # [batch, 1, r_dim]
        r_context = r_context.repeat(1, x_target.shape[1], 1)                   # [batch, target, r_dim]
        return r_context

    def forward(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
    ) -> Tuple[TensorType[B, T, Y], TensorType[B, T, Y]]:

        # Encode
        context = torch.cat((x_context, y_context), dim=-1)                     # [batch, context, x_dim + y_dim]
        r_i_context = self.encoder(context)                                     # [batch, context, r_dim]

        # Aggregate
        r_context = self._aggregate(r_i_context, x_context, x_target)           # [batch, target, r_dim]

        # Decode
        query = torch.cat((x_target, r_context), dim=-1)                        # [batch, target, x_dim + r_dim]
        mu_log_sigma = self.decoder(query)                                      # [batch, target, y_dim * 2]

        y_dim = mu_log_sigma.shape[-1] // 2
        mu, log_sigma = torch.split(mu_log_sigma, (y_dim, y_dim), dim=-1)       # [batch, target, y_dim] * 2
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        return mu, sigma


class CNP(CNPBase):
    """
    Conditional Neural Process
    """

    def __init__(self,
        x_dim: int,
        y_dim: int,
        r_dim: int = 128,
        encoder_dims: List[int] = [128, 128, 128, 128, 128],
        decoder_dims: List[int] = [128, 128, 128],
    ):
        encoder = MLP(
            in_features=(x_dim + y_dim),
            hidden_features=encoder_dims,
            out_features=r_dim,
        )

        decoder = MLP(
            in_features=(x_dim + r_dim),
            hidden_features=decoder_dims,
            out_features=(y_dim * 2),
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
        )
