from ..typing import *

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
    """

    encoder:         nn.Module = None
    self_attention:  Optional[nn.Module] = None
    transform_qk:    Optional[nn.Module] = None
    cross_attention: nn.Module = None
    decoder:         nn.Module = None
    min_sigma:       float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.cross_attention is None:
            raise ValueError("cross_attention is not specified")

    def _encode(
        self,
        x:    Array[B, ([M],), P, X],
        y:    Array[B, ([M],), P, Y],
        mask: Array[B, P],
    ) -> Array[B, ([M],), P, R]:

        # TODO: Temporary fix before implementing more efficient attention module
        xy = jnp.concatenate((x, y), axis=-1)                                                       # [batch, (*model), point, x_dim + y_dim]
        # xy, shape = F.flatten(xy, start=0, stop=-2, return_shape=True)                              # [batch x (*model), point, x_dim + y_dim]
        r_i = self.encoder(xy)                                                                      # [batch x (*model), point, r_dim]
        if self.self_attention is not None:
            r_i = self.self_attention(r_i, mask=mask)                                               # [batch x (*model), point, r_dim]
        # r_i = F.unflatten(xy, shape, axis=0)                                                        # [batch, (*model), point, r_dim]
        return r_i                                                                                  # [batch, (*model), point, r_dim]

    def _aggregate(
        self,
        x_tar:    Array[B, ([M],), T, X],
        x_ctx:    Array[B, ([M],), C, X],
        r_i_ctx:  Array[B, ([M],), C, R],
        mask_ctx: Array[B, C],
    ) -> Array[B, ([M],), T, R]:

        # TODO: Temporary fix before implementing more efficient attention module
        # q_i, shape = F.flatten(x_tar,   start=0, stop=-2, return_shape=True)                        # [batch x (*model), target,  x_dim]
        # k_i        = F.flatten(x_ctx,   start=0, stop=-2)                                           # [batch x (*model), context, x_dim]
        # r_i_ctx    = F.flatten(r_i_ctx, start=0, stop=-2)                                           # [batch x (*model), context, r_dim]
        q_i, k_i = x_tar, x_ctx

        if self.transform_qk is not None:
            q_i, k_i = self.transform_qk(q_i), self.transform_qk(k_i)                               # [batch x (*model), target, qk_dim], [batch x (*model), context, qk_dim]

        r_ctx = self.cross_attention(q_i, k_i, r_i_ctx, mask=mask_ctx)                              # [batch x (*model), target, r_dim]
        # r_ctx = F.unflatten(r_ctx, shape, axis=0)                                                   # [batch, (*model), target, r_dim]
        return r_ctx                                                                                # [batch, (*model), target, r_dim]


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
        transform_qk_dims: Optional[Sequence[int]] = (128, 128, 128, 128, 128),
        encoder_dims: Sequence[int] = (128, 128, 128, 128, 128),
        decoder_dims: Sequence[int] = (128, 128, 128),
    ):

        if sa_heads is not None:
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim, last_activation=True)
            self_attention = MultiheadSelfAttention(dim_out=r_dim, num_heads=sa_heads)
        else:
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim, last_activation=False)
            self_attention = None

        if transform_qk_dims is not None:
            transform_qk = MLP(hidden_features=transform_qk_dims, out_features=r_dim, last_activation=False)
        else:
            transform_qk = None

        cross_attention = MultiheadAttention(dim_out=r_dim, num_heads=ca_heads)
        decoder = MLP(hidden_features=decoder_dims, out_features=(y_dim * 2))

        return AttnCNPBase(
            encoder=encoder,
            self_attention=self_attention,
            transform_qk=transform_qk,
            cross_attention=cross_attention,
            decoder=decoder,
        )
