from ..type import *

from jax import numpy as jnp
from flax import linen as nn

from .cnp import CNPBase
from .. import functional as F
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

    Args:
        encoder:         [batch, context, x_dim + y_dim] -> [batch, context, r_dim]
        self_attention:  [batch, context, r_dim] -> [batch, context, r_dim]
        cross_attention: [batch, context, r_dim] -> [batch,  target, r_dim]
        decoder:         [batch, target, x_dim + r_dim] -> [batch, target, y_dim * 2]
    """

    encoder:         nn.Module = None
    self_attention:  nn.Module = None
    cross_attention: nn.Module = None
    decoder:         nn.Module = None

    def __post_init__(self):
        super().__post_init__()
        if self.cross_attention is None:
            raise ValueError("cross_attention is not specified")

    def _encode(self,
        ctx:  Array[..., P, V],
        mask: Array[P],
    ) -> Array[..., P, R]:

        r_i = self.encoder(ctx)                                                 # [..., point, r_dim]

        if self.self_attention is not None:
            attn_mask = F.repeat_axis(mask, axis=0, repeats=x.shape[0])         # [..., point]
            r_i = self.self_attention(r_i, mask=attn_mask)

        return r_i

    def _aggregate(self,
        r_i_ctx:  Array[..., C, R],
        x_ctx:    Array[..., C, X],
        x_tar:    Array[..., T, X],
        mask_ctx: Array[C],
    ) -> Array[..., T, R]:

        attn_mask_ctx = F.repeat_axis(mask_ctx, axis=0, repeats=x_tar.shape[0])
        r_ctx = self.cross_attention(x_tar, x_ctx, r_i_ctx, mask=attn_mask_ctx) # [batch, target, r_dim]
        return r_ctx


class AttnCNP(AttnCNPBase):
    """
    Attentive Conditional Neural Process
    """

    def __new__(cls,
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
