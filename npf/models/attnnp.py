from logging import warning
from typing import List, Optional
from torchtyping import TensorType
from ..type import *

from torch import nn

from .np import NPBase

from ..modules import (
    MLP,
    MultiheadSelfAttention,
    MultiheadCrossAttention,
)


__all__ = ["AttnNPBase", "AttnNP"]


class AttnNPBase(NPBase):
    """Attentive Neural Process Base"""

    def __init__(self,
        latent_encoder,
        determ_encoder,
        cross_attention,
        decoder,
        loss_type: str = "vi",
    ):
        """
        Args:
            latent_encoder  : [batch, context, x_dim + y_dim]
                           -> [batch, context, z_dim * 2]
            determ_encoder  : [batch, context, x_dim + y_dim]
                           -> [batch, context, r_dim]
            cross_attention : [batch, context, r_dim]
                           -> [batch,  target, r_dim]
            decoder         : [batch,  latent, target, x_dim (+ r_dim) + z_dim]
                           -> [batch,  latent, target, y_dim * 2]
            loss_type       : str ("vi" or "ml")
        """
        super().__init__(
            latent_encoder=latent_encoder,
            determ_encoder=determ_encoder,
            decoder=decoder,
            loss_type=loss_type,
        )

        self.cross_attention = cross_attention

    def _determ_aggregate(self,
        r_i_context: TensorType[B, C, R],
        x_context:   TensorType[B, C, X],
        x_target:    TensorType[B, T, X],
    ) -> TensorType[B, T, R]:

        r_context = self.cross_attention(x_target, x_context, r_i_context)      # [batch, target, r_dim]
        return r_context


class AttnNP(AttnNPBase):
    """Attentive Neural Process"""

    def __init__(self,
        x_dim: int, y_dim: int,
        r_dim: int = 128, z_dim: int = 128,
        common_sa_heads: Optional[int] = None,
        latent_sa_heads: Optional[int] = None,
        determ_sa_heads: Optional[int] = None,
        determ_ca_heads: Optional[int] = 8,
        common_encoder_dims: Optional[List[int]] = None,
        latent_encoder_dims: Optional[List[int]] = [128, 128],
        determ_encoder_dims: Optional[List[int]] = [128, 128, 128, 128, 128],
        decoder_dims: List[int] = [128, 128, 128, 128],
        loss_type: str = "vi",
    ):

        if common_encoder_dims is not None:
            if r_dim != z_dim * 2:
                raise ValueError("Dimension mismatch: r_dim != z_dim * 2")

            latent_encoder_mlp = MLP(
                in_features=(x_dim + y_dim),
                hidden_features=common_encoder_dims,
                out_features=(z_dim * 2),
            )

            if common_sa_heads is not None:
                latent_self_attention = MultiheadSelfAttention(
                    num_heads=common_sa_heads,
                    input_dim=r_dim,
                )
                latent_encoder = nn.Sequential(
                    latent_encoder_mlp,
                    latent_self_attention,
                )
            else:
                latent_encoder = latent_encoder_mlp

            determ_encoder = latent_encoder
            decoder_input_dim = x_dim + r_dim + z_dim

        else:
            if latent_encoder_dims is None:
                raise ValueError("Invalid combination of encoders")

            latent_encoder_mlp = MLP(
                in_features=(x_dim + y_dim),
                hidden_features=latent_encoder_dims,
                out_features=(z_dim * 2),
            )

            if latent_sa_heads is not None:
                latent_self_attention = MultiheadSelfAttention(
                    num_heads=latent_sa_heads,
                    input_dim=(z_dim * 2),
                )
                latent_encoder = nn.Sequential(
                    latent_encoder_mlp,
                    latent_self_attention,
                )
            else:
                latent_encoder = latent_encoder_mlp

            if determ_encoder_dims is not None:
                determ_encoder_mlp = MLP(
                    in_features=(x_dim + y_dim),
                    hidden_features=determ_encoder_dims,
                    out_features=r_dim,
                )

                if determ_sa_heads is not None:
                    determ_self_attention = MultiheadSelfAttention(
                        num_heads=determ_sa_heads,
                        input_dim=r_dim,
                    )
                    determ_encoder = nn.Sequential(
                        determ_encoder_mlp,
                        determ_self_attention,
                    )
                else:
                    determ_encoder = determ_encoder_mlp

                decoder_input_dim = x_dim + r_dim + z_dim

            else:
                determ_encoder = None
                decoder_input_dim = x_dim + z_dim

        if determ_ca_heads is not None:
            if determ_encoder is None:
                warning.user("No deterministic encoder, but cross attention is provided that has no effect.")
            else:
                cross_attention = MultiheadCrossAttention(
                    num_heads=determ_ca_heads,
                    qk_dim=x_dim,
                    v_dim=r_dim,
                )
        else:
            if determ_encoder is None:
                cross_attention = None
            else:
                raise ValueError("cross attention for deterministic is required")

        decoder = MLP(
            in_features=decoder_input_dim,
            hidden_features=decoder_dims,
            out_features=(y_dim * 2),
        )

        super().__init__(
            latent_encoder=latent_encoder,
            determ_encoder=determ_encoder,
            cross_attention=cross_attention,
            decoder=decoder,
            loss_type=loss_type,
        )
