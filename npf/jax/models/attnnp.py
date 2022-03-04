from ..type import *

from jax import numpy as jnp
from flax import linen as nn

from .np import NPBase
from .. import functional as F
from ..modules import (
    MLP,
    MultiheadAttention,
    MultiheadSelfAttention,
)


__all__ = [
    "AttnNPBase",
    "AttnNP",
]


class AttnNPBase(NPBase):
    """
    Base class of Attentive Neural Process

    Args:
        latent_encoder:         [batch, context, x_dim + y_dim] -> [batch, context, z_dim x 2]
        latent_self_attention:  [batch, context, z_dim x 2] -> [batch, context, z_dim x 2]
        determ_encoder:         [batch, context, x_dim + y_dim] -> [batch, context, r_dim]
        determ_self_attention:  [batch, context, r_dim] -> [batch, context, r_dim]
        determ_cross_attention: [batch, context, r_dim] -> [batch, target, r_dim]
        decoder:                [batch, latent, target, x_dim (+ r_dim) + z_dim] -> [batch, latent, target, y_dim x 2]
    """

    latent_encoder:         nn.Module = None
    latent_self_attention:  nn.Module = None
    determ_encoder:         nn.Module = None
    determ_self_attention:  nn.Module = None
    determ_cross_attention: nn.Module = None
    decoder:                nn.Module = None

    def _determ_aggregate(self,
        r_i_ctx:  Array[B, C, R],
        x_ctx:    Array[B, C, R],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[C],
    ) -> Array[B, T, R]:

        attn_mask_ctx = F.repeat_axis(mask_ctx, axis=0, repeats=x_tar.shape[0])
        r_ctx = self.determ_cross_attention(x_tar, x_ctx, r_i_ctx, mask=attn_mask_ctx) # [batch, target, r_dim]
        return r_ctx

    def _encode(self,
        x:    Array[..., P, X],
        y:    Array[..., P, Y],
        mask: Array[P],
        latent_only: bool = False,
    ) -> Union[Array[B, C, Z], Tuple[Array[B, C, Z], Array[B, C, R]]]:

        xy = jnp.concatenate((x, y), axis=-1)                                   # [..., point, x_dim + y_dim]
        _z_i = self.latent_encoder(xy)

        attn_mask = F.repeat_axis(mask, axis=0, repeats=x.shape[0])             # [..., point]
        if self.latent_self_attention is not None:
            z_i = self.latent_self_attention(_z_i, mask=attn_mask)

        if latent_only:
            return z_i
        else:
            if self.determ_encoder is None:
                r_i = None
            elif self.determ_encoder is self.latent_encoder:
                if self.determ_self_attention is None:
                    r_i = _z_i
                elif self.determ_self_attention is self.determ_self_attention:
                    r_i = z_i
                else:
                    r_i = self.determ_self_attention(_z_i, mask=attn_mask)
            else:
                r_i = self.determ_encoder(xy)
                if self.determ_self_attention is not None:
                    r_i = self.determ_self_attention(r_i, mask=attn_mask)

            return z_i, r_i


class AttnNP(AttnNPBase):
    """
    Attentive Neural Process
    """

    def __new__(cls,
        y_dim: int,
        r_dim: int = 128,
        z_dim: int = 128,
        common_sa_heads: Optional[int] = None,
        latent_sa_heads: Optional[int] = 8,
        determ_sa_heads: Optional[int] = 8,
        determ_ca_heads: Optional[int] = 8,
        common_encoder_dims: Optional[Sequence[int]] = None,
        latent_encoder_dims: Optional[Sequence[int]] = (128, 128),
        determ_encoder_dims: Optional[Sequence[int]] = (128, 128, 128, 128, 128),
        decoder_dims: Sequence[int] = (128, 128, 128, 128),
        loss_type: str = "vi",
    ):

        if common_encoder_dims is not None:
            if r_dim != z_dim * 2:
                raise ValueError("Dimension mismatch: r_dim != z_dim x 2")

            latent_encoder = MLP(hidden_features=common_encoder_dims, out_features=(z_dim * 2), last_activation=(common_sa_heads is not None))
            latent_self_attention = MultiheadSelfAttention(dim_out=r_dim, num_heads=common_sa_heads) if common_sa_heads is not None else None

            determ_encoder = latent_encoder
            determ_self_attention = latent_self_attention
            determ_cross_attention = MultiheadAttention(dim_out=r_dim, num_heads=determ_ca_heads) if determ_ca_heads is not None else None

        else:
            if latent_encoder_dims is None:
                raise ValueError("Invalid combination of encoders")

            latent_encoder = MLP(hidden_features=latent_encoder_dims, out_features=(z_dim * 2), last_activation=(latent_sa_heads is not None))
            latent_self_attention = MultiheadSelfAttention(dim_out=(z_dim * 2), num_heads=latent_sa_heads) if latent_sa_heads is not None else None

            if determ_encoder_dims is not None:
                determ_encoder = MLP(hidden_features=determ_encoder_dims, out_features=r_dim, last_activation=(determ_sa_heads is not None))
                determ_self_attention = MultiheadSelfAttention(dim_out=r_dim, num_heads=determ_sa_heads) if determ_sa_heads is not None else None
                determ_cross_attention = MultiheadAttention(dim_out=r_dim, num_heads=determ_ca_heads) if determ_ca_heads is not None else None
            else:
                determ_encoder = None
                determ_self_attention = None
                determ_cross_attention = None

        decoder = MLP(hidden_features=decoder_dims, out_features=(y_dim * 2))

        return AttnNPBase(
            latent_encoder=latent_encoder,
            latent_self_attention=latent_self_attention,
            determ_encoder=determ_encoder,
            determ_self_attention=determ_self_attention,
            determ_cross_attention=determ_cross_attention,
            decoder=decoder,
            loss_type=loss_type,
        )
