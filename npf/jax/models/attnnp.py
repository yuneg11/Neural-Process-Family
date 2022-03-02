from ..type import *

import numpy as np

from jax import numpy as jnp
from flax import linen as nn

from .np import NPBase

from ..modules import (
    MLP,
    MultiheadAttention,
    MultiheadSelfAttention,
)


__all__ = [
    "AttnCNPBase",
    "AttnCNP",
]


class AttnNPBase(NPBase):
    """
    Base class of Attentive Neural Process

    Args:
        encoder         : [batch, context, x_dim + y_dim]
                        -> [batch, context, r_dim]
        cross_attention : [batch, context, r_dim]
                        -> [batch, target, r_dim]
        decoder         : [batch, target, x_dim + r_dim]
                        -> [batch, target, y_dim * 2]
    """
    cross_attention: nn.Module
    self_attention: Optional[nn.Module] = None

    def _aggregate(self,
        r_i_ctx:  Float[B, C, R],
        x_ctx:    Float[B, C, X],
        x_tar:    Float[B, T, X],
        mask_ctx: Float[B, C],
    ) -> Float[B, T, R]:

        if self.self_attention is not None:
            r_i_ctx = self.self_attention(r_i_ctx, mask=mask_ctx)
        r_ctx = self.cross_attention(x_tar, x_ctx, r_i_ctx, mask=mask_ctx)                          # [batch, target, r_dim]

        return r_ctx


class AttnNP(AttnNPBase):
    """
    Attentive Neural Process
    """

    def __new__(cls,
        y_dim: int,
        r_dim: int = 128,
        z_dim: int = 128,
        common_sa_heads: Optional[int] = None,
        latent_sa_heads: Optional[int] = None,
        determ_sa_heads: Optional[int] = None,
        determ_ca_heads: Optional[int] = 8,
        common_encoder_dims: Optional[Sequence[int]] = None,
        latent_encoder_dims: Optional[Sequence[int]] = [128, 128],
        determ_encoder_dims: Optional[Sequence[int]] = [128, 128, 128, 128, 128],
        decoder_dims: Sequence[int] = (128, 128, 128),
    ):

        if sa_heads is not None:
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim, last_activation=True)
            self_attention = MultiheadSelfAttention(dim_out=r_dim, num_heads=sa_heads)
        else:
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim, last_activation=False)
            self_attention = None

        cross_attention = MultiheadAttention(dim_out=r_dim, num_heads=ca_heads)
        decoder = MLP(hidden_features=decoder_dims, out_features=(y_dim * 2))

        return AttnNPBase(
            encoder=encoder,
            self_attention=self_attention,
            cross_attention=cross_attention,
            decoder=decoder,
        )
