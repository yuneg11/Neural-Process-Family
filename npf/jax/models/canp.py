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
    "CANPBase",
    "CANP",
]


class CANPBase(CNPBase):
    """
    Base class of Conditional Attentive Neural Process
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
        # r_i = F.unflatten(r_i, shape, axis=0)                                                        # [batch, (*model), point, r_dim]
        return r_i                                                                                  # [batch, (*model), point, r_dim]

    def _aggregate(
        self,
        x:        Array[B, ([M],), P, X],
        x_ctx:    Array[B, ([M],), C, X],
        r_i_ctx:  Array[B, ([M],), C, R],
        mask_ctx: Array[B, C],
    ) -> Array[B, ([M],), P, R]:

        # TODO: Temporary fix before implementing more efficient attention module
        # r_i_q, shape = F.flatten(x,       start=0, stop=-2, return_shape=True)                      # [batch x (*model), point,   x_dim]
        # r_i_k        = F.flatten(x_ctx,   start=0, stop=-2)                                         # [batch x (*model), context, x_dim]
        # r_i_ctx      = F.flatten(r_i_ctx, start=0, stop=-2)                                         # [batch x (*model), context, r_dim]
        r_i_q, r_i_k = x, x_ctx

        if self.transform_qk is not None:
            r_i_q, r_i_k = self.transform_qk(r_i_q), self.transform_qk(r_i_k)                       # [batch x (*model), point, qk_dim], [batch x (*model), context, qk_dim]

        r_ctx = self.cross_attention(r_i_q, r_i_k, r_i_ctx, mask=mask_ctx)                          # [batch x (*model), point, r_dim]
        # r_ctx = F.unflatten(r_ctx, shape, axis=0)                                                   # [batch, (*model), point, r_dim]
        return r_ctx                                                                                # [batch, (*model), point, r_dim]


class CANP:
    """
    Conditional Attentive Neural Process
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
        min_sigma: float = 0.1,
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

        return CANPBase(
            encoder=encoder,
            self_attention=self_attention,
            transform_qk=transform_qk,
            cross_attention=cross_attention,
            decoder=decoder,
            min_sigma=min_sigma,
        )
