from ..typing import *

from jax import numpy as jnp
from flax import linen as nn

from .cnp import CNPBase
from ..modules import (
    MLP,
    MultiheadAttention,
    MultiheadSelfAttention,
)

__all__ = [
    "AttnCNPBase",
    "AttnCNP",
]


class AttnCNPBase(CNPBase):
    """
    Base class of Attentive Conditional Neural Process
    """

    encoder:         nn.Module = None
    self_attention:  Optional[nn.Module] = None
    cross_attention: nn.Module = None
    decoder:         nn.Module = None
    min_sigma:       float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.cross_attention is None:
            raise ValueError("cross_attention is not specified")

    def _encode(
        self,
        x:    Array[B, P, X],
        y:    Array[B, P, Y],
        mask: Array[B, P],
    ) -> Array[B, P, R]:

        xy = jnp.concatenate((x, y), axis=-1)                                                       # [batch, point, x_dim + y_dim]
        r_i = self.encoder(xy)                                                                      # [batch, point, r_dim]
        if self.self_attention is not None:
            r_i = self.self_attention(r_i, mask=mask)                                               # [batch, point, r_dim]
        return r_i                                                                                  # [batch, point, r_dim]

    def _aggregate(
        self,
        x_tar:    Array[B, T, X],
        x_ctx:    Array[B, C, X],
        r_i_ctx:  Array[B, C, R],
        mask_ctx: Array[B, C],
    ) -> Array[B, T, R]:

        r_ctx = self.cross_attention(x_tar, x_ctx, r_i_ctx, mask=mask_ctx)                          # [batch, target, r_dim]
        return r_ctx                                                                                # [batch, target, r_dim]


class AttnCNP:
    """
    Attentive Conditional Neural Process
    """

    def __new__(
        cls,
        y_dim: int,
        r_dim: int = 128,
        sa_heads: Optional[int] = 8,
        ca_heads: Optional[int] = 8,
        encoder_dims: Sequence[int] = (128, 128, 128, 128, 128),
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

        return AttnCNPBase(
            encoder=encoder,
            self_attention=self_attention,
            cross_attention=cross_attention,
            decoder=decoder,
        )
