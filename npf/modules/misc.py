from torchtyping import TensorType
from ..type import *

from torch import nn


__all__ = [
    "Sample",
]


class Sample(nn.Module):
    @staticmethod
    def forward(
        dist: TensorType[B, 1, Z],
        num_latents: int = 1,
    ):
        samples = dist.rsample([num_latents]).transpose(1, 0)                   # [batch, latent, 1, z_dim]
        return samples
