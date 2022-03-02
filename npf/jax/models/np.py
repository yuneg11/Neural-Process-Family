from ..type import *

import numpy as np

import jax
from jax import random
from jax import numpy as jnp
from flax import linen as nn

from .base import LatentNPF

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
        latent_encoder : [batch, context, x_dim + y_dim]
                      -> [batch, context, z_dim * 2]
        determ_encoder : [batch, context, x_dim + y_dim]
                      -> [batch, context, r_dim]
        decoder        : [batch, latent, target, x_dim (+ r_dim) + z_dim]
                      -> [batch, latent, target, y_dim * 2]
        loss_type      : Callable (vi_loss or ml_loss)
    """

    latent_encoder: nn.Module
    determ_encoder: nn.Module
    decoder: nn.Module

    def setup(self):
        if self.latent_encoder is None:
            raise ValueError("latent_encoder is required")
        elif self.determ_encoder is self.latent_encoder:
            self._encode = self._common_encode
            self._latent_encode_only = lambda d: self.common_encoder(d)
        else:
            if self.determ_encoder is None:
                self._encode = self._latent_encode
            else:
                self._encode = self._latent_determ_encode

            self._latent_encode_only = lambda d: self.latent_encoder(d)

    def _encode(self,
        ctx: Float[B, C, X + Y],
    ) -> Tuple[Float[B, C, Z], Float[B, C, R]]:
        raise NotImplementedError

    def _latent_encode_only(self,
        data: Float[B, T, X + Y],
    ) -> Float[B, T, Z]:
        raise NotImplementedError

    def _common_encode(self,
        ctx: Float[B, C, X + Y],
    ) -> Tuple[Float[B, C, Z], Float[B, C, R]]:

        z_i_ctx = r_i_ctx = self.common_encoder(ctx)
        return z_i_ctx, r_i_ctx

    def _latent_encode(self,
        ctx: Float[B, C, X + Y],
    ) -> Tuple[Float[B, C, Z], None]:

        z_i_ctx = self.latent_encoder(ctx)
        return z_i_ctx, None

    def _latent_determ_encode(self,
        ctx: Float[B, C, X + Y],
    ) -> Tuple[Float[B, C, Z], Float[B, C, R]]:

        z_i_ctx = self.latent_encoder(ctx)
        r_i_ctx = self.determ_encoder(ctx)
        return z_i_ctx, r_i_ctx

    # Aggregate and Distribute

    @staticmethod
    def _determ_aggregate(
        r_i_ctx:  Float[B, C, R],
        x_ctx:    Float[B, C, R],
        x_tar:    Float[B, T, X],
        mask_ctx: Float[B, C],
    ) -> Float[B, T, R]:

        mask_ctx = jnp.expand_dims(mask_ctx, axis=-1)                           # [batch, ctx, 1]
        r_i_ctx = jnp.where(mask_ctx, r_i_ctx, 0.)                                      # [batch, ctx, r_dim]
        r_ctx = jnp.sum(r_i_ctx,  axis=1, keepdims=True) \
                  / jnp.sum(mask_ctx, axis=1, keepdims=True)                                    # [batch, 1, r_dim]
        r_ctx = r_ctx.repeat(x_tar.shape[1], axis=1)                                     # [batch, tar, r_dim]
        return r_ctx

    @staticmethod
    def _latent_dist(
        z_i:      Float[B, P, Z * 2],
        mask_ctx: Float[B, C],
    ) -> Tuple[Float[B, 1, Z], Float[B, 1, Z]]:

        mask_ctx = jnp.expand_dims(mask_ctx, axis=-1)                                           # [batch, ctx, 1]
        mu_log_sigma = jnp.where(mask_ctx, z_i, 0.)                                             # [batch, ctx, z_dim x 2]
        mu_log_sigma = jnp.sum(mu_log_sigma, axis=1, keepdims=True) \
                     / jnp.sum(mask_ctx, axis=1, keepdims=True)                                 # [batch, 1, z_dim x 2]
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, 1, z_dim] x 2
        sigma = 0.1 + 0.9 * nn.sigmoid(log_sigma)                                                   # [batch, 1, z_dim]

        return mu, sigma

    # Decode

    def _build_query(self,
        x_tar:     Float[B, T, X],
        z_samples: Float[B, L, 1, Z],
        r_ctx: Optional[Float[B, T, R]] = None,
        num_latents: int = 1,
    ) -> Union[Float[B, L, T, X + R + Z], Float[B, L, T, X + Z]]:

        num_tars = x_tar.shape[1]
        z_samples = z_samples.repeat(num_tars, axis=2)                                           # [batch, latent, tar, z_dim]
        x_tar = jnp.expand_dims(x_tar, axis=1).repeat(num_latents, axis=1)                    # [batch, latent, tar, x_dim]

        if r_ctx is not None:
            r_ctx = jnp.expand_dims(r_ctx, axis=1).repeat(num_latents, axis=1)              # [batch, latent, tar, r_dim]
            query = jnp.concatenate([x_tar, r_ctx, z_samples], axis=-1)                      # [batch, latent, tar, x_dim + r_dim + z_dim]
        else:
            query = jnp.concatenate([x_tar, z_samples], axis=-1)                                 # [batch, latent, tar, x_dim + z_dim]

        return query

    def _decode(self,
        query: Union[Float[B, L, T, X + R + Z], Float[B, L, T, X + Z]],
        mask_tar: Float[B, T],
    ) -> Tuple[Float[B, L, T, Y], Float[B, L, T, Y]]:

        mu_log_sigma = self.decoder(query)                                                          # [batch, latent, tar, y_dim x 2]
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, latent, tar, y_dim] x 2
        sigma = 0.1 + 0.9 * nn.softplus(log_sigma)                                                  # [batch, latent, tar, y_dim]

        mask_tar = jnp.expand_dims(mask_tar, axis=(1, -1))                                          # [batch, 1, tar, 1]
        mu    = jnp.where(mask_tar, mu, 0.)
        sigma = jnp.where(mask_tar, sigma, 0.)

        return mu, sigma

    def __call__(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:    Float[B, T, X],
        mask_ctx: Float[B, C],
        mask_tar: Float[B, T],
        num_latents:  int = 1
    ) -> Tuple[Float[B, L, T, Y], Float[B, L, T, Y]]:

        # Encode
        ctx = jnp.concatenate((x_ctx, y_ctx), axis=-1)                                  # [batch, ctx, x_dim + y_dim]
        z_i_ctx, r_i_ctx = self._encode(ctx)                                            # [batch, ctx, z_dim], [batch, ctx, r_dim]

        # Latent representation
        z_ctx_mu, z_ctx_sigma = self._latent_dist(z_i_ctx, mask_ctx)                # [batch, 1, z_dim] x 2

        rng = self.make_rng("sample")
        num_batch, z_dim = z_ctx_mu.shape[0], z_ctx_mu.shape[2]
        z_samples = z_ctx_mu + z_ctx_sigma * random.normal(rng, shape=(num_batch, num_latents, z_dim))  # [batch, latent, z_dim]
        z_samples = jnp.expand_dims(z_samples, axis=2)                                              # [batch, latent, 1, z_dim]

        # Deterministic representation
        if r_i_ctx is not None:
            r_ctx = self._determ_aggregate(r_i_ctx, x_ctx, x_tar, mask_ctx)      # [batch, tar, r_dim])
        else:
            r_ctx = None

        # Decode
        query = self._build_query(x_tar, z_samples, r_ctx, num_latents)                      # [batch, latent, tar, x_dim (+ r_dim) + z_dim]
        mu, sigma = self._decode(query, mask_tar)                                                # [batch, latent, tar, y_dim] * 2

        return mu, sigma

    # Loss

    def vi_loss(self,
        x_ctx:    Float[B, C, X],
        y_ctx:    Float[B, C, Y],
        x_tar:     Float[B, T, X],
        y_tar:     Float[B, T, Y],
        mask_ctx: Float[B, C],
        mask_tar:  Float[B, T],
        num_latents:  int = 1,
    ) -> Float:

        # Encode
        ctx = jnp.concatenate((x_ctx, y_ctx), axis=-1)                                              # [batch, ctx, x_dim + y_dim]
        tar = jnp.concatenate((x_tar, y_tar), axis=-1)                                              # [batch, tar, x_dim + y_dim]

        z_i_ctx, r_i_ctx = self._encode(ctx)                                                        # [batch, ctx, z_dim], [batch, ctx, r_dim]
        z_i_tar = self._latent_encode_only(tar)                                                     # [batch, tar, z_dim]

        # Latent representation
        z_ctx_mu, z_ctx_sigma = self._latent_dist(z_i_ctx, mask_ctx)                                # [batch, 1, z_dim] x 2
        z_tar_mu, z_tar_sigma = self._latent_dist(z_i_tar, mask_tar)                                # [batch, 1, z_dim] x 2

        rng = self.make_rng("sample")
        num_batch, z_dim = z_ctx_mu.shape[0], z_ctx_mu.shape[2]
        z_samples = z_ctx_mu + z_ctx_sigma * random.normal(rng, shape=(num_batch, num_latents, z_dim))  # [batch, latent, z_dim]
        z_samples = jnp.expand_dims(z_samples, axis=2)                                              # [batch, latent, 1, z_dim]

        # Deterministic representation
        if r_i_ctx is not None:
            r_ctx = self._determ_aggregate(r_i_ctx, x_ctx, x_tar, mask_ctx)                         # [batch, tar, r_dim])
        else:
            r_ctx = None

        # Decode
        query = self._build_query(x_tar, z_samples, r_ctx, num_latents)                      # [batch, latent, tar, x_dim (+ r_dim) + z_dim]
        mu, sigma = self._decode(query, mask_tar)                                                # [batch, latent, tar, y_dim] * 2

        # Loss
        log_likelihood = self._log_likelihood(y_tar, mu, sigma, mask_tar)                     # [batch, latent, tar]
        log_likelihood = jnp.sum(log_likelihood) / (jnp.sum(mask_tar) * num_latents)             # [1]

        kl_divergence = self._kl_divergence(z_tar_mu, z_tar_sigma, z_ctx_mu, z_ctx_sigma)  # [batch, 1, z_dim]
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
        )
