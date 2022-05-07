from ..typing import *

from jax import random
from jax import numpy as jnp
from jax.scipy import stats
from flax import linen as nn

from .base import NPF
from .. import functional as F
from ..modules import (
    MLP,
)

__all__ = [
    "NPBase",
    "NP",
]


class NPBase(NPF):
    """
    Base class of Neural Process
    """

    latent_encoder: nn.Module = None
    determ_encoder: Optional[nn.Module] = None
    decoder:        nn.Module = None
    loss_type:      str = "vi"
    min_sigma:      float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.latent_encoder is None:
            raise ValueError("latent_encoder is not specified")
        if self.decoder is None:
            raise ValueError("decoder is not specified")
        if self.loss_type not in ("vi", "ml"):
            raise ValueError(f"Invalid loss_type: {self.loss_type}. loss_type must be either 'vi' or 'ml'.")

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
        z_i = self.latent_encoder(xy)                                                               # [batch, point, z_dim x 2]

        if latent_only:
            return z_i                                                                              # [batch, point, z_dim x 2]
        else:
            if self.determ_encoder is None:
                r_i = None                                                                          # None
            elif self.determ_encoder is self.latent_encoder:
                r_i = z_i                                                                           # [batch, point, r_dim]
            else:
                r_i = self.determ_encoder(xy)                                                       # [batch, point, r_dim]
            return z_i, r_i                                                                         # [batch, point, z_dim x 2], ([batch, point, r_dim] | None)

    def _latent_dist(
        self,
        z_i:  Array[B, P, Z * 2],
        mask: Array[B, P],
    ) -> Tuple[Array[B, 1, Z], Array[B, 1, Z]]:

        z_mu_log_sigma = F.masked_mean(z_i, mask, axis=-2, non_mask_axis=-1, keepdims=True)         # [batch, 1, z_dim x 2]
        z_mu, z_log_sigma = jnp.split(z_mu_log_sigma, 2, axis=-1)                                   # [batch, 1, z_dim] x 2
        z_sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(z_log_sigma)                  # [batch, 1, z_dim]
        return z_mu, z_sigma                                                                        # [batch, 1, z_dim] x 2

    def _latent_sample(
        self,
        z_mu:    Array[B, 1, Z],
        z_sigma: Array[B, 1, Z],
        num_latents: int = 1,
    ) -> Array[B, L, 1, Z]:

        rng = self.make_rng("sample")
        num_batches, z_dim = z_mu.shape[0], z_mu.shape[2]
        z = z_mu + z_sigma * random.normal(rng, shape=(num_batches, num_latents, z_dim))            # [batch, latent, z_dim]
        z = jnp.expand_dims(z, axis=-2)                                                             # [batch, latent, 1, z_dim]
        return z                                                                                    # [batch, latent, 1, z_dim]

    def _determ_aggregate(
        self,
        x_tar:    Array[B, T, X],
        x_ctx:    Array[B, C, X],
        r_i_ctx:  Array[B, C, R],
        mask_ctx: Array[B, C],
    ) -> Array[B, T, R]:

        r_ctx = F.masked_mean(r_i_ctx, mask_ctx, axis=-2, non_mask_axis=-1, keepdims=True)          # [batch, 1,      r_dim]
        r_ctx = jnp.repeat(r_ctx, x_tar.shape[-2], axis=-2)                                         # [batch, target, r_dim]
        return r_ctx                                                                                # [batch, target, r_dim]

    def _decode(
        self,
        x_tar:    Array[B, T, X],
        z_ctx:    Array[B, L, 1, Z],
        r_ctx:    Array[B, T, R],
        mask_tar: Array[B, T],
    ) -> Tuple[Array[B, L, T, Y], Array[B, L, T, Y]]:

        z_ctx = z_ctx.repeat(x_tar.shape[-2], axis=-2)                                              # [batch, latent, target, z_dim]
        x_tar = F.repeat_axis(x_tar, repeats=z_ctx.shape[1], axis=1)                                # [batch, latent, target, z_dim]

        if r_ctx is not None:
            r_ctx = F.repeat_axis(r_ctx, repeats=z_ctx.shape[1], axis=1)
            query = jnp.concatenate((x_tar, z_ctx, r_ctx), axis=-1)                                 # [batch, latent, target, x_dim + z_dim + r_dim]
        else:
            query = jnp.concatenate((x_tar, z_ctx), axis=-1)                                        # [batch, latent, target, x_dim + z_dim]

        query = F.flatten(query, start=0, stop=2)                                                   # [batch x latent, target, x_dim + z_dim (+ r_dim)]
        mu_log_sigma = self.decoder(query)                                                          # [batch x latent, target, y_dim x 2]
        mu_log_sigma = F.unflatten(mu_log_sigma, z_ctx.shape[0:2], axis=0)                          # [batch, latent, target, y_dim, 2]

        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, latent, target, y_dim] x 2
        sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(log_sigma)                      # [batch, latent, target, y_dim]
        return mu, sigma                                                                            # [batch, latent, target, y_dim] x 2

    @nn.compact
    def __call__(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        *,
        num_latents: int = 1,
        return_aux: bool = False,
    ) -> Union[
        Tuple[Array[B, L, [T], Y], Array[B, L, [T], Y]],
        Tuple[Array[B, L, [T], Y], Array[B, L, [T], Y], Tuple[Array[B, 1, Z], Array[B, 1, Z]]],
    ]:

        # Flatten
        shape_tar = x_tar.shape[1:-1]
        x_ctx    = F.flatten(x_ctx,    start=1, stop=-1)                                            # [batch, context, x_dim]
        y_ctx    = F.flatten(y_ctx,    start=1, stop=-1)                                            # [batch, context, y_dim]
        x_tar    = F.flatten(x_tar,    start=1, stop=-1)                                            # [batch, target,  x_dim]
        mask_ctx = F.flatten(mask_ctx, start=1)                                                     # [batch, context]
        mask_tar = F.flatten(mask_tar, start=1)                                                     # [batch, target]

        # Algorithm
        z_i_ctx, r_i_ctx = self._encode(x_ctx, y_ctx, mask_ctx)                                     # [batch, context, z_dim x 2], ([batch, context, r_dim] | None)
        z_mu_ctx, z_sigma_ctx = self._latent_dist(z_i_ctx, mask_ctx)                                # [batch, 1,       z_dim] x 2
        z_ctx = self._latent_sample(z_mu_ctx, z_sigma_ctx, num_latents)                             # [batch, latent, 1, z_dim]

        if r_i_ctx is None:
            r_ctx = None
        else:
            r_ctx = self._determ_aggregate(x_tar, x_ctx, r_i_ctx, mask_ctx)                         # [batch, target, r_dim]

        mu, sigma = self._decode(x_tar, z_ctx, r_ctx, mask_tar)                                     # [batch, latent, target, y_dim] x 2

        # Unflatten and mask
        mu    = F.masked_fill(mu,    mask_tar, fill_value=0.,   non_mask_axis=(1, -1))              # [batch, latent, target, y_dim]
        sigma = F.masked_fill(sigma, mask_tar, fill_value=1e-6, non_mask_axis=(1, -1))              # [batch, latent, target, y_dim]
        mu    = F.unflatten(mu,    shape_tar, axis=-2)                                              # [batch, latent, *target, y_dim]
        sigma = F.unflatten(sigma, shape_tar, axis=-2)                                              # [batch, latent, *target, y_dim]

        if return_aux:
            return mu, sigma, (z_mu_ctx, z_sigma_ctx)                                               # [batch, latent, *target, y_dim] x 2, ([batch, 1, z_dim] x 2)
        else:
            return mu, sigma                                                                        # [batch, latent, *target, y_dim] x 2

    def log_likelihood(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        *,
        num_latents: int = 1,
        as_mixture: bool = True,
        return_aux: bool = False,
    ) -> Union[
        Array,
        Tuple[Array, Tuple[Array[B, 1, Z], Array[B, 1, Z]]],
    ]:

        mu, sigma, aux = \
            self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_latents=num_latents, return_aux=True) # [batch, latent, *target, y_dim] x 2, ([batch, 1, z_dim] x 2)

        s_y_tar = jnp.expand_dims(y_tar, axis=1)                                                    # [batch, 1,      *target, y_dim]
        log_prob = stats.norm.logpdf(s_y_tar, mu, sigma)                                            # [batch, latent, *target, y_dim]
        ll = jnp.sum(log_prob, axis=-1)                                                             # [batch, latent, *target]

        if as_mixture:
            ll = F.logmeanexp(ll, axis=1)                                                           # [batch, *target]
            ll = F.masked_mean(ll, mask_tar)                                                        # (1)
        else:
            axis = [-d for d in range(1, mask_tar.ndim)]
            ll = F.masked_mean(ll, mask_tar, axis=axis, non_mask_axis=1)                            # [batch, latent]
            ll = F.logmeanexp(ll, axis=1)                                                           # [batch]
            ll = jnp.mean(ll)                                                                       # (1)

        if return_aux:
            return ll, aux                                                                          # (1), ([batch, 1, z_dim] x 2)
        else:
            return ll                                                                               # (1)

    def loss(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        *,
        num_latents: int = 1,
        as_mixture: bool = False,
    ) -> Array:

        if self.loss_type == "vi":
            return self.vi_loss(                                                                    # (1)
                x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar,
                num_latents=num_latents, as_mixture=as_mixture,
            )
        elif self.loss_type == "ml":
            return self.ml_loss(                                                                    # (1)
                x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar,
                num_latents=num_latents, as_mixture=as_mixture,
            )

    def vi_loss(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        *,
        num_latents: int = 1,
        as_mixture: bool = False,
    ) -> Array:

        ll, (z_mu_ctx, z_sigma_ctx) = self.log_likelihood(                                          # (1), ([batch, 1, z_dim] x 2)
            x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar,
            num_latents=num_latents, as_mixture=as_mixture, return_aux=True,
        )

        x_tar    = F.flatten(x_tar,    start=1, stop=-1)                                            # [batch, target,  x_dim]
        y_tar    = F.flatten(y_tar,    start=1, stop=-1)                                            # [batch, target,  y_dim]
        mask_tar = F.flatten(mask_tar, start=1)                                                     # [batch, target]

        z_i_tar = self._encode(x_tar, y_tar, mask_tar, latent_only=True)                            # [batch, target, z_dim x 2]
        z_mu_tar, z_sigma_tar = self._latent_dist(z_i_tar, mask_tar)                                # [batch, 1,      z_dim] x 2

        kld = jnp.mean(                                                                             # (1)
            jnp.log(z_sigma_ctx) - jnp.log(z_sigma_tar)
            + (jnp.square(z_sigma_tar) + jnp.square(z_mu_tar - z_mu_ctx)) / (2 * jnp.square(z_sigma_ctx))
            - 0.5
        )

        loss = -ll + kld                                                                            # (1)
        return loss

    def ml_loss(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        *,
        num_latents: int = 1,
        as_mixture: bool = False,
    ) -> Array:

        loss = -self.log_likelihood(                                                                # (1)
            x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar,
            num_latents=num_latents, as_mixture=as_mixture,
        )
        return loss                                                                                 # (1)


class NP:
    """
    Neural Process
    """

    def __new__(
        cls,
        y_dim: int,
        r_dim: int = 128,
        z_dim: int = 128,
        common_encoder_dims: Optional[Sequence[int]] = None,
        latent_encoder_dims: Optional[Sequence[int]] = (128, 128),
        determ_encoder_dims: Optional[Sequence[int]] = (128, 128, 128, 128, 128),
        decoder_dims: Sequence[int] = (128, 128, 128),
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
