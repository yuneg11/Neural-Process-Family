from ..typing import *

from jax import numpy as jnp
from flax import linen as nn

from .np import NPBase
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
    """

    latent_encoder:         nn.Module = None
    latent_self_attention:  Optional[nn.Module] = None
    determ_encoder:         Optional[nn.Module] = None
    determ_self_attention:  Optional[nn.Module] = None
    determ_cross_attention: Optional[nn.Module] = None
    decoder:                nn.Module = None
    loss_type:              str = "vi"
    min_sigma:              float = 0.1

    def _determ_aggregate(
        self,
        x_tar:    Array[B, T, X],
        x_ctx:    Array[B, C, X],
        r_i_ctx:  Array[B, C, R],
        mask_ctx: Array[B, C],
    ) -> Array[B, T, R]:

        r_ctx = self.determ_cross_attention(x_tar, x_ctx, r_i_ctx, mask=mask_ctx)                   # [batch, target, r_dim]
        return r_ctx

    def _encode(
        self,
        x:    Array[B, P, X],
        y:    Array[B, P, Y],
        mask: Array[B, P],
        latent_only: bool = False,
    ) -> Union[
        Tuple[Array[B, P, Z * 2], Array[B, P, R]],
        Array[B, P, Z * 2],
    ]:

        xy = jnp.concatenate((x, y), axis=-1)                                                       # [batch, point, x_dim + y_dim]
        _z_i = self.latent_encoder(xy)                                                              # [batch, point, z_dim x 2]

        if self.latent_self_attention is not None:
            z_i = self.latent_self_attention(_z_i, mask=mask)                                       # [batch, point, z_dim x 2]

        if latent_only:
            return z_i                                                                              # [batch, point, z_dim x 2]
        else:
            if self.determ_encoder is None:
                r_i = None                                                                          # None
            elif self.determ_encoder is self.latent_encoder:
                if self.determ_self_attention is None:
                    r_i = _z_i                                                                      # [batch, point, r_dim]
                elif self.determ_self_attention is self.latent_self_attention:
                    r_i = z_i                                                                       # [batch, point, r_dim]
                else:
                    r_i = self.determ_self_attention(_z_i, mask=mask)                               # [batch, point, r_dim]
            else:
                r_i = self.determ_encoder(xy)                                                       # [batch, point, r_dim]
                if self.determ_self_attention is not None:
                    r_i = self.determ_self_attention(r_i, mask=mask)                                # [batch, point, r_dim]
            return z_i, r_i                                                                         # [batch, point, z_dim x 2], ([batch, point, r_dim] | None)


class AttnNP:
    """
    Attentive Neural Process
    """

    def __new__(
        cls,
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
