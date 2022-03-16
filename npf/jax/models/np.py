from ..type import *

from jax import random
from jax import numpy as jnp
from flax import linen as nn

from .base import LatentNPF
from .. import functional as F
from ..modules import (
    MLP,
)


__all__ = [
    "NPBase",
    "NP",
]


class NPBase(LatentNPF):
    """
    Base class of Neural Process

    Args:
        latent_encoder: [batch, context, x_dim + y_dim] -> [batch, context, z_dim x 2]
        determ_encoder: [batch, context, x_dim + y_dim] -> [batch, context, r_dim]
        decoder:        [batch, latent, target, x_dim (+ r_dim) + z_dim] -> [batch, latent, target, y_dim x 2]
        loss_type:      "vi" or "ml"
    """

    latent_encoder: nn.Module = None
    determ_encoder: nn.Module = None
    decoder:        nn.Module = None
    loss_type:      str = "vi"

    def __post_init__(self):
        super().__post_init__()
        if self.latent_encoder is None:
            raise ValueError("latent_encoder is not specified")
        if self.decoder is None:
            raise ValueError("decoder is not specified")

    def _encode(self,
        x:    Array[B, P, X],
        y:    Array[B, P, Y],
        mask: Array[B, P],
        latent_only: bool = False,
    ) -> Union[Array[B, C, Z], Tuple[Array[B, C, Z], Array[B, C, R]]]:

        xy = jnp.concatenate((x, y), axis=-1)                                                       # [batch, point, x_dim + y_dim]
        z_i = self.latent_encoder(xy)

        if latent_only:
            return z_i
        else:
            if self.determ_encoder is None:
                r_i = None
            elif self.determ_encoder is self.latent_encoder:
                r_i = z_i
            else:
                r_i = self.determ_encoder(xy)
            return z_i, r_i

    # Aggregate and Distribute

    def _determ_aggregate(self,
        r_i_ctx:  Array[B, C, R],
        x_ctx:    Array[B, C, R],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[B, C],
    ) -> Array[B, T, R]:

        r_ctx = F.masked_mean(r_i_ctx, mask_ctx, axis=1, mask_axis=(0, 1), keepdims=True)                   # [batch, 1, r_dim]
        r_ctx = r_ctx.repeat(x_tar.shape[1], axis=1)                                                # [batch, target, r_dim]
        return r_ctx

    @staticmethod
    def _latent_dist(
        z_i:      Array[B, P, Z * 2],
        mask_ctx: Array[B, C],
    ) -> Tuple[Array[B, 1, Z], Array[B, 1, Z]]:

        mu_log_sigma = F.masked_mean(z_i, mask_ctx, axis=1, mask_axis=(0, 1), keepdims=True)
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, 1, z_dim] x 2
        sigma = 0.1 + 0.9 * nn.sigmoid(log_sigma)                                                   # [batch, 1, z_dim]
        return mu, sigma

    def _build_query(self,
        x_tar:     Array[B, T, X],
        z_samples: Array[B, L, 1, Z],
        r_ctx: Optional[Array[B, T, R]] = None,
        num_latents: int = 1,
    ) -> Union[Array[B, L, T, X + R + Z], Array[B, L, T, X + Z]]:

        num_tars = x_tar.shape[-2]
        z_samples = z_samples.repeat(num_tars, axis=-2)                                             # [batch, latent, target, z_dim]
        x_tar = F.repeat_axis(x_tar, repeats=num_latents, axis=1)

        if r_ctx is not None:
            r_ctx = F.repeat_axis(r_ctx, repeats=num_latents, axis=1)
            query = jnp.concatenate((x_tar, r_ctx, z_samples), axis=-1)                             # [batch, latent, target, x_dim + r_dim + z_dim]
        else:
            query = jnp.concatenate((x_tar, z_samples), axis=-1)                                    # [batch, latent, target, x_dim + z_dim]

        return query

    def _decode(self,
        query: Union[Array[B, L, T, X + R + Z], Array[B, L, T, X + Z]],
        mask_tar: Array[B, T],
    ) -> Tuple[Array[B, L, T, Y], Array[B, L, T, Y]]:

        mu_log_sigma = self.decoder(query)                                                          # [batch, latent, target, y_dim x 2]
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, latent, target, y_dim] x 2
        sigma = 0.1 + 0.9 * nn.softplus(log_sigma)                                                  # [batch, latent, target, y_dim]

        mu    = F.apply_mask(mu,    mask_tar, mask_axis=(0, -2))                                    # [batch, latent, target, y_dim]
        sigma = F.apply_mask(sigma, mask_tar, mask_axis=(0, -2))                                    # [batch, latent, target, y_dim]
        return mu, sigma

    def _predict(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_latents:  int = 1,
    ) -> Tuple[Array[B, L, [T], Y], Array[B, L, [T], Y], Array[B, 1, Z], Array[B, 1, Z]]:

        x_ctx,    _ = F.flatten(x_ctx,    start=1, stop=-1)                                         # [batch, context, x_dim]
        y_ctx,    _ = F.flatten(y_ctx,    start=1, stop=-1)                                         # [batch, context, y_dim]
        x_tar, meta = F.flatten(x_tar,    start=1, stop=-1)                                         # [batch, target,  x_dim]
        mask_ctx, _ = F.flatten(mask_ctx, start=1)                                                  # [batch, context]
        mask_tar, _ = F.flatten(mask_tar, start=1)                                                  # [batch, target]

        # Encode
        z_i_ctx, r_i_ctx = self._encode(x_ctx, y_ctx, mask_ctx)                                     # [batch, context, z_dim], [batch, context, r_dim]

        # Latent representation
        z_ctx_mu, z_ctx_sigma = self._latent_dist(z_i_ctx, mask_ctx)                                # [batch, 1, z_dim] x 2

        rng = self.make_rng("sample")
        num_batch, z_dim = z_ctx_mu.shape[0], z_ctx_mu.shape[2]
        z_samples = z_ctx_mu + z_ctx_sigma * random.normal(rng, shape=(num_batch, num_latents, z_dim)) # [batch, latent, z_dim]
        z_samples = jnp.expand_dims(z_samples, axis=-2)                                             # [batch, latent, 1, z_dim]

        # Deterministic representation
        if r_i_ctx is not None:
            r_ctx = self._determ_aggregate(r_i_ctx, x_ctx, x_tar, mask_ctx)                         # [batch, target, r_dim])
        else:
            r_ctx = None

        # Decode
        query = self._build_query(x_tar, z_samples, r_ctx, num_latents)                             # [batch, latent, target, x_dim (+ r_dim) + z_dim]
        mu, sigma = self._decode(query, mask_tar)                                                   # [batch, latent, target, y_dim] x 2
        mu    = F.unflatten(mu,    meta, axis=-2)                                                   # [batch, *target, y_dim]
        sigma = F.unflatten(sigma, meta, axis=-2)                                                   # [batch, *target, y_dim]

        return mu, sigma, z_ctx_mu, z_ctx_sigma

    @nn.compact
    def __call__(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_latents: int = 1,
    ) -> Tuple[Array[B, L, [T], Y], Array[B, L, [T], Y]]:

        mu, sigma, _, _ = self._predict(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_latents)
        return mu, sigma

    def vi_loss(self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        num_latents: int = 1,
    ) -> Array:

        mu, sigma, z_ctx_mu, z_ctx_sigma = self._predict(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_latents)

        # Target latent distribution
        _x_tar,    _ = F.flatten(x_tar,    start=1, stop=-1)                                        # [batch, target,  x_dim]
        _y_tar,    _ = F.flatten(y_tar,    start=1, stop=-1)                                        # [batch, target,  y_dim]
        _mask_tar, _ = F.flatten(mask_tar, start=1)                                                 # [batch, target]

        z_i_tar = self._encode(_x_tar, _y_tar, _mask_tar, latent_only=True)                         # [batch, target, z_dim]
        z_tar_mu, z_tar_sigma = self._latent_dist(z_i_tar, _mask_tar)                               # [batch, 1, z_dim] x 2

        # Loss
        y_tar = jnp.expand_dims(y_tar, axis=1)                                                      # [batch, 1, *target, y_dim]
        log_likelihood = self._log_likelihood(y_tar, mu, sigma)                                     # [batch, latent, *target]
        log_likelihood = F.masked_mean(log_likelihood, mask_tar, mask_axis=(0, -1))                 # [1]

        kl_divergence = self._kl_divergence(z_tar_mu, z_tar_sigma, z_ctx_mu, z_ctx_sigma)           # [batch, 1, z_dim]
        kl_divergence = jnp.mean(kl_divergence)                                                     # [1]

        loss = -log_likelihood + kl_divergence                                                      # [1]
        return loss


class NP:
    """
    Conditional Neural Process
    """

    def __new__(cls,
        y_dim: int,
        r_dim: int = 128,
        z_dim: int = 128,
        common_encoder_dims: Optional[Sequence[int]] = None,
        latent_encoder_dims: Optional[Sequence[int]] = (128, 128),
        determ_encoder_dims: Optional[Sequence[int]] = (128, 128, 128, 128, 128),
        decoder_dims: Sequence[int] = (128, 128, 128, 128),
        loss_type: str = "vi",
    ):

        if common_encoder_dims is not None:
            if r_dim != z_dim * 2:
                raise ValueError("Dimension mismatch: r_dim != z_dim * 2")

            latent_encoder = MLP(hidden_features=common_encoder_dims, out_features=(z_dim * 2))
            determ_encoder = latent_encoder

        else:
            if latent_encoder_dims is None:
                raise ValueError("Invalid combination of encoders")

            latent_encoder = MLP(hidden_features=latent_encoder_dims, out_features=(z_dim * 2))

            if determ_encoder_dims is not None:
                determ_encoder = MLP(hidden_features=determ_encoder_dims, out_features=r_dim)
            else:
                determ_encoder = None

        decoder = MLP(hidden_features=decoder_dims, out_features=(y_dim * 2))

        return NPBase(
            latent_encoder=latent_encoder,
            determ_encoder=determ_encoder,
            decoder=decoder,
            loss_type=loss_type,
        )
