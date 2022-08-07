from ..typing import *

from jax import random
from jax import numpy as jnp
from flax import linen as nn

from .base import NPF
from .. import functional as F
from ..data import NPData
from ..utils import npf_io, MultivariateNormalDiag
from ..modules import MLP


__all__ = [
    "NPBase",
    "NP",
]


class NPBase(NPF):
    """
    Base class of Neural Process
    """

    latent_encoder:   nn.Module = None
    determ_encoder:   Optional[nn.Module] = None
    decoder:          nn.Module = None
    loss_type:        str = "vi"
    min_sigma:        float = 0.1
    min_latent_sigma: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.latent_encoder is None:
            raise ValueError("latent_encoder is not specified")
        if self.decoder is None:
            raise ValueError("decoder is not specified")
        if self.loss_type not in ("vi", "iwae", "elbo", "ml"):
            raise ValueError(f"Invalid loss_type: {self.loss_type}. loss_type: 'vi', 'iwae', 'elbo', 'ml'")

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
        z_sigma = self.min_latent_sigma + (1 - self.min_latent_sigma) * nn.sigmoid(z_log_sigma)     # [batch, 1, z_dim]
        return z_mu, z_sigma                                                                        # [batch, 1, z_dim] x 2

    def _latent_sample(
        self,
        z_mu:    Array[B, 1, Z],
        z_sigma: Array[B, 1, Z],
        num_latents: int = 1,
    ) -> Array[B, L, 1, Z]:

        rng = self.make_rng("sample")
        num_batches, z_dim = z_mu.shape[0], z_mu.shape[2]
        eps = random.normal(rng, shape=(num_batches, num_latents, z_dim))                           # [batch, latent, z_dim]

        z = z_mu + z_sigma * eps                                                                    # [batch, latent, z_dim]
        z = jnp.expand_dims(z, axis=-2)                                                             # [batch, latent, 1, z_dim]
        return z                                                                                    # [batch, latent, 1, z_dim]

    def _determ_aggregate(
        self,
        x:        Array[B, P, X],
        x_ctx:    Array[B, C, X],
        r_i_ctx:  Array[B, C, R],
        mask_ctx: Array[B, C],
    ) -> Array[B, P, R]:

        r_ctx = F.masked_mean(r_i_ctx, mask_ctx, axis=-2, non_mask_axis=-1, keepdims=True)          # [batch, 1,     r_dim]
        r_ctx = jnp.repeat(r_ctx, x.shape[-2], axis=-2)                                             # [batch, point, r_dim]
        return r_ctx                                                                                # [batch, point, r_dim]

    def _decode(
        self,
        query: Array[B, L, P, X + Z + (R)],
        mask:  Array[B, P],
    ) -> Tuple[Array[B, L, P, Y], Array[B, L, P, Y]]:

        query, shape = F.flatten(query, start=0, stop=-2, return_shape=True)                        # [batch x latent, point, x_dim + z_dim (+ r_dim)]
        mu_log_sigma = self.decoder(query)                                                          # [batch x latent, point, y_dim]
        mu_log_sigma = F.unflatten(mu_log_sigma, shape, axis=0)                                     # [batch,  latent, point, y_dim, 2]

        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, latent, point, y_dim] x 2
        sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(log_sigma)                      # [batch, latent, point, y_dim]
        return mu, sigma                                                                            # [batch, latent, point, y_dim] x 2

    @nn.compact
    @npf_io(flatten=True)
    def __call__(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
        training: bool = False,
        return_aux: bool = False,
    ):

        if training:
            z_i, r_i_ctx = self._encode(data.x, data.y, data.mask)                                  # [batch, point, z_dim x 2], ([batch, point, r_dim] | None)
            z_mu, z_sigma = self._latent_dist(z_i, data.mask)                                       # [batch, 1,     z_dim] x 2
        else:
            z_i_ctx, r_i_ctx = self._encode(data.x_ctx, data.y_ctx, data.mask_ctx)                  # [batch, context, z_dim x 2], ([batch, context, r_dim] | None)
            z_mu, z_sigma = self._latent_dist(z_i_ctx, data.mask_ctx)                               # [batch, 1,       z_dim] x 2

        z = self._latent_sample(z_mu, z_sigma, num_latents)                                         # [batch, latent, 1,     z_dim]
        s_z = z.repeat(data.x.shape[-2], axis=-2)                                                   # [batch, latent, point, z_dim]
        s_x = F.repeat_axis(data.x, repeats=num_latents, axis=1)                                    # [batch, latent, point, x_dim]

        if r_i_ctx is None:
            query = jnp.concatenate((s_x, s_z), axis=-1)                                            # [batch, latent, point, x_dim + z_dim]
        else:
            r_ctx = self._determ_aggregate(data.x, data.x_ctx, r_i_ctx, data.mask_ctx)              # [batch, point, r_dim]
            r_ctx = F.repeat_axis(r_ctx, repeats=num_latents, axis=1)                               # [batch, latent, point, r_dim]
            query = jnp.concatenate((s_x, s_z, r_ctx), axis=-1)                                     # [batch, latent, point, x_dim + z_dim + r_dim]

        mu, sigma = self._decode(query, data.mask)                                                  # [batch, latent, point, y_dim] x 2

        # Mask
        mu    = F.masked_fill(mu,    data.mask, fill_value=0., non_mask_axis=(1, -1))               # [batch, latent, point, y_dim]
        sigma = F.masked_fill(sigma, data.mask, fill_value=0., non_mask_axis=(1, -1))               # [batch, latent, point, y_dim]

        if return_aux:
            z = jnp.squeeze(z, axis=-2)                                                             # [batch, latent, z_dim]
            z_i_ctx = self._encode(data.x_ctx, data.y_ctx, data.mask, latent_only=True)             # [batch, point, z_dim x 2]
            z_mu_ctx, z_sigma_ctx = self._latent_dist(z_i_ctx, data.mask)                           # [batch, 1,     z_dim] x 2

            return mu, sigma, (z, z_mu, z_sigma, z_mu_ctx, z_sigma_ctx)                             # [batch, latent, point, y_dim] x 2, (aux)
        else:
            return mu, sigma                                                                        # [batch, latent, point, y_dim] x 2

    @npf_io(flatten_input=True)
    def log_likelihood(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
        joint: bool = False,
        split_set: bool = False,
    ):

        mu, sigma = self(data, num_latents=num_latents, skip_io=True)                               # [batch, latent, point, y_dim] x 2

        s_y = jnp.expand_dims(data.y, axis=1)                                                       # [batch, 1,      point, y_dim]
        log_prob = MultivariateNormalDiag(mu, sigma).log_prob(s_y)                                  # [batch, latent, point]

        if joint:
            ll = F.masked_sum(log_prob, data.mask, axis=-1, non_mask_axis=1)                        # [batch, latent]
            ll = F.logmeanexp(ll, axis=1) / jnp.sum(data.mask, axis=-1)                             # [batch]

            if split_set:
                ll_ctx = F.masked_sum(log_prob, data.mask_ctx, axis=-1, non_mask_axis=1)            # [batch, latent]
                ll_tar = F.masked_sum(log_prob, data.mask_tar, axis=-1, non_mask_axis=1)            # [batch, latent]
                ll_ctx = F.logmeanexp(ll_ctx, axis=1) / jnp.sum(data.mask_ctx, axis=-1)             # [batch]
                ll_tar = F.logmeanexp(ll_tar, axis=1) / jnp.sum(data.mask_tar, axis=-1)             # [batch]

        else:
            ll_all = F.logmeanexp(log_prob, axis=1)                                                 # [batch, point]
            ll = F.masked_mean(ll_all, data.mask, axis=-1)                                          # [batch]

            if split_set:
                ll_ctx = F.masked_mean(ll_all, data.mask_ctx, axis=-1)                              # [batch]
                ll_tar = F.masked_mean(ll_all, data.mask_tar, axis=-1)                              # [batch]

        ll = jnp.mean(ll)                                                                           # (1)

        if split_set:
            ll_ctx = jnp.mean(ll_ctx)                                                               # (1)
            ll_tar = jnp.mean(ll_tar)                                                               # (1)

            return ll, ll_ctx, ll_tar                                                               # (1) x 3
        else:
            return ll                                                                               # (1)

    @npf_io(flatten_input=True)
    def loss(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
        joint: bool = True,        # For `ml_loss`
        return_aux: bool = False,  # For `elbo_loss`
    ) -> Array:
        if self.loss_type == "vi" or self.loss_type == "iwae":
            return self.iwae_loss(data, num_latents=num_latents, skip_io=True)
        elif self.loss_type == "elbo":
            return self.elbo_loss(data, num_latents=num_latents, return_aux=return_aux, skip_io=True)
        elif self.loss_type == "ml":
            return self.ml_loss(data, num_latents=num_latents, joint=joint, skip_io=True)

    @npf_io(flatten_input=True)
    def iwae_loss(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
    ) -> Array:

        mu, sigma, (z, z_mu, z_sigma, z_mu_ctx, z_sigma_ctx) = self(                                # [batch, latent, point, y_dim] x 2,
            data, num_latents=num_latents, training=True, return_aux=True, skip_io=True,            #    ([batch, latent, z_dim], [batch, 1, z_dim] x 4)
        )

        s_y = jnp.expand_dims(data.y, axis=1)                                                       # [batch, 1,      point, y_dim]
        log_prob = MultivariateNormalDiag(mu, sigma).log_prob(s_y)                                  # [batch, latent, point]
        ll = F.masked_sum(log_prob, data.mask, axis=-1, non_mask_axis=1)                            # [batch, latent]

        log_p = MultivariateNormalDiag(z_mu_ctx, z_sigma_ctx).log_prob(z)                           # [batch, latent]
        log_q = MultivariateNormalDiag(z_mu, z_sigma).log_prob(z)                                   # [batch, latent]

        loss = -F.logmeanexp(ll + log_p - log_q, axis=1) / jnp.sum(data.mask, axis=-1)              # [batch]
        loss = jnp.mean(loss)                                                                       # (1)

        return loss                                                                                 # (1)

    @npf_io(flatten_input=True)
    def elbo_loss(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
        return_aux: bool = False,
    ) -> Array:

        mu, sigma, (_, z_mu, z_sigma, z_mu_ctx, z_sigma_ctx) = self(                                # [batch, latent, point, y_dim] x 2,
            data, num_latents=num_latents, training=True, return_aux=True, skip_io=True,            #    (_, [batch, 1, z_dim] x 4)
        )

        s_y = jnp.expand_dims(data.y, axis=1)                                                       # [batch, 1,      point, y_dim]
        log_prob = MultivariateNormalDiag(mu, sigma).log_prob(s_y)                                  # [batch, latent, point]

        ll = F.masked_mean(log_prob, data.mask, axis=(1, -1), non_mask_axis=1)                      # [batch]

        q_z = MultivariateNormalDiag(z_mu, z_sigma)                                                 # [batch, 1, z_dim]
        p_z = MultivariateNormalDiag(z_mu_ctx, z_sigma_ctx)                                         # [batch, 1, z_dim]
        kld = jnp.squeeze(q_z.kl_divergence(p_z), axis=1)                                           # [batch]


        loss = -ll + kld                                                                            # [batch]
        loss = jnp.mean(loss)                                                                       # (1)

        if return_aux:
            ll = jnp.mean(ll)                                                                       # (1)
            kld = jnp.mean(kld)                                                                     # (1)
            return loss, dict(ll=ll, kld=kld)                                                       # (1), (aux)
        else:
            return loss                                                                             # (1)

    @npf_io(flatten_input=True)
    def ml_loss(
        self,
        data: NPData,
        *,
        num_latents: int = 1,
        joint: bool = True,
    ) -> Array:

        loss = -self.log_likelihood(data, num_latents=num_latents, joint=joint, skip_io=True)       # (1)
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
        min_sigma: float = 0.1,
        min_latent_sigma: float = 0.1,
    ):

        if common_encoder_dims is not None:
            if r_dim != z_dim * 2:
                raise ValueError("Cannot use common encoder: r_dim != z_dim x 2")

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
            min_sigma=min_sigma,
            min_latent_sigma=min_latent_sigma,
        )
