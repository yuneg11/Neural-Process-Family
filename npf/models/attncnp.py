from typing import List, Optional
from torchtyping import TensorType
from ..type import *

from torch import nn

from .cnp import CNPBase

from ..modules import (
    MLP,
    MultiheadSelfAttention,
    MultiheadCrossAttention,
)


__all__ = ["AttnCNPBase", "AttnCNP"]


class AttnCNPBase(CNPBase):
    """Attentive Conditional Neural Process Base"""

    def __init__(self,
        encoder,
        cross_attention,
        decoder,
    ):
        """
        Args:
            encoder         : [batch, context, x_dim + y_dim]
                           -> [batch, context, r_dim]
            cross_attention : [batch, context, r_dim]
                           -> [batch,  target, r_dim]
            decoder         : [batch,  target, x_dim + r_dim]
                           -> [batch,  target, y_dim * 2]
        """
        super().__init__(
            encoder=encoder,
            decoder=decoder,
        )

        self.cross_attention = cross_attention

    def _aggregate(self,
        r_i_context: TensorType[B, C, R],
        x_context:   TensorType[B, C, X],
        x_target:    TensorType[B, T, X],
    ) -> TensorType[B, T, R]:

        r_context = self.cross_attention(x_target, x_context, r_i_context)      # [batch, target, r_dim]
        return r_context


class AttnCNP(AttnCNPBase):
    """Attentive Conditional Neural Process"""

    def __init__(self,
        x_dim: int,
        y_dim: int,
        r_dim: int = 128,
        sa_heads: Optional[int] = None,
        ca_heads: Optional[int] = 8,
        encoder_dims: List[int] = [128, 128, 128, 128, 128],
        decoder_dims: List[int] = [128, 128, 128],
    ):
        encoder_mlp = MLP(
            in_features=(x_dim + y_dim),
            hidden_features=encoder_dims,
            out_features=r_dim,
            last_activation=(sa_heads is not None),
        )

        if sa_heads is not None:
            self_attention = MultiheadSelfAttention(
                num_heads=sa_heads,
                input_dim=r_dim,
            )

            encoder = nn.Sequential(
                encoder_mlp,
                self_attention,
            )

        else:
            encoder = encoder_mlp

        cross_attention = MultiheadCrossAttention(
            num_heads=ca_heads,
            qk_dim=x_dim,
            v_dim=r_dim,
        )

        decoder = MLP(
            in_features=(x_dim + r_dim),
            hidden_features=decoder_dims,
            out_features=(y_dim * 2),
        )

        super().__init__(
            encoder=encoder,
            cross_attention=cross_attention,
            decoder=decoder,
        )
