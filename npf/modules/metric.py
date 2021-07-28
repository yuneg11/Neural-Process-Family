from typing import Union
from torchtyping import TensorType
from ..type import *

import torch
from torch import nn
from torch.distributions import Normal


__all__ = [
    "LogLikelihood",
]


class LogLikelihood(nn.Module):
    @staticmethod
    def forward(
        y_target: TensorType[B, T, Y],
        mu:       Union[TensorType[B, T, Y], TensorType[B, L, T, Y]],
        sigma:    Union[TensorType[B, T, Y], TensorType[B, L, T, Y]],
    ) -> Union[TensorType[B, T], TensorType[B, L, T]]:

        if y_target.dim() == 3 and mu.dim() == sigma.dim() == 4:
            y_target = y_target.unsqueeze(dim=1)                                # [batch, 1, target, y_dim]

        distribution = Normal(mu, sigma)                                        # [batch, (latent,) target, y_dim]
        log_prob = distribution.log_prob(y_target)                              # [batch, (latent,) target, y_dim]
        log_likelihood = log_prob.sum(dim=-1)                                   # [batch, (latent,) target]

        return log_likelihood
