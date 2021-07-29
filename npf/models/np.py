from ..type import *

import math

import torch
from torch.nn import functional as F
from torch.distributions import Normal

from .base import LatentNPF

from ..modules import (
    MLP,
    Sample,
    LogLikelihood,
    KLDivergence,
)


__all__ = ["NPBase", "NP"]


class NPBase(LatentNPF):
    """Neural Process Base"""

    def __init__(self,
        latent_encoder,
        determ_encoder,
        decoder,
        loss_type: str = "vi",
    ):
        """
        Args:
            latent_encoder : [batch, context, x_dim + y_dim]
                          -> [batch, context, z_dim * 2]
            determ_encoder : [batch, context, x_dim + y_dim]
                          -> [batch, context, r_dim]
            decoder        : [batch,  latent, target, x_dim (+ r_dim) + z_dim]
                          -> [batch,  latent, target, y_dim * 2]
            loss_type      : str ("vi" or "ml")
        """
        super().__init__(
            loss_type=("vi" if loss_type is None else loss_type),
        )

        if latent_encoder is None:
            raise ValueError("latent_encoder is required")
        elif determ_encoder is latent_encoder:
            self.common_encoder = latent_encoder
            self._encode = self._common_encode
            self._latent_encode_only = lambda d: self.common_encoder(d)
        else:
            self.latent_encoder = latent_encoder
            self.determ_encoder = determ_encoder

            if self.determ_encoder is None:
                self._encode = self._latent_encode
            else:
                self._encode = self._latent_determ_encode

            self._latent_encode_only = lambda d: self.latent_encoder(d)

        self.decoder = decoder

        self.sample_fn = Sample()
        self.log_likelihood_fn = LogLikelihood()
        self.kl_divergence_fn = KLDivergence()

    # Encodes
    def _encode(self,
        context: TensorType[B, C, X + Y],
    ) -> Tuple[TensorType[B, C, Z], TensorType[B, C, R]]:
        raise NotImplementedError  # Implementation is choosed at __init__

    def _latent_encode_only(self,
        data: TensorType[B, C + T, X + Y],
    ) -> TensorType[B, C + T, Z]:
        raise NotImplementedError  # Implementation is choosed at __init__

    def _common_encode(self,
        context: TensorType[B, C, X + Y],
    ) -> Tuple[TensorType[B, C, Z], TensorType[B, C, R]]:

        z_i_context = r_i_context = self.common_encoder(context)
        return z_i_context, r_i_context

    def _latent_encode(self,
        context: TensorType[B, C, X + Y],
    ) -> Tuple[TensorType[B, C, Z], None]:

        z_i_context = self.latent_encoder(context)
        return z_i_context, None

    def _latent_determ_encode(self,
        context: TensorType[B, C, X + Y],
    ) -> Tuple[TensorType[B, C, Z], TensorType[B, C, R]]:

        z_i_context = self.latent_encoder(context)
        r_i_context = self.determ_encoder(context)
        return z_i_context, r_i_context

    # Aggregate and Distribute

    def _determ_aggregate(self,
        r_i_context: TensorType[B, C, R],
        x_context:   TensorType[B, C, X],
        x_target:    TensorType[B, T, X],
    ) -> TensorType[B, T, R]:

        r_context = torch.mean(r_i_context, dim=1, keepdim=True)                # [batch, 1, r_dim]
        r_context = r_context.repeat(1, x_target.shape[1], 1)                   # [batch, target, r_dim]
        return r_context

    def _latent_dist(self,
        z_i: TensorType[B, P, Z * 2],
    ) -> TensorType[B, 1, Z]:
        # latent aggregate
        mu_log_sigma = torch.mean(z_i, dim=1, keepdim=True)                     # [batch, 1, z_dim * 2]

        # distribution
        z_dim = mu_log_sigma.shape[-1] // 2
        mu, log_sigma = torch.split(mu_log_sigma, (z_dim, z_dim), dim=-1)       # [batch, 1, z_dim] * 2
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)                            # [batch, 1, z_dim]
        z_dist = Normal(loc=mu, scale=sigma)                                    # [batch, 1, z_dim]
        return z_dist

    # Decode

    def _build_query(self,
        x_target:  TensorType[B, T, X],
        z_samples: TensorType[B, L, 1, Z],
        r_context: Optional[TensorType[B, T, R]],
        num_latents: int = 1,
    ) -> Union[TensorType[B, L, T, X + R + Z], TensorType[B, L, T, X + Z]]:
        num_targets = x_target.shape[1]

        z_samples = z_samples.repeat(1, 1, num_targets, 1)                      # [batch, latent, target, z_dim]

        x_target = x_target[:, None, :, :]                                      # [batch, 1, target, r_dim]
        x_target = x_target.repeat(1, num_latents, 1, 1)                        # [batch, latent, target, x_dim]

        if r_context is not None:
            r_context = r_context[:, None, :, :]                                # [batch, 1, target, r_dim]
            r_context = r_context.repeat(1, num_latents, 1, 1)                  # [batch, latent, target, r_dim]

            query = torch.cat((x_target, r_context, z_samples), dim=-1)         # [batch, latent, target, x_dim + r_dim + z_dim]
        else:
            query = torch.cat((x_target, z_samples), dim=-1)                    # [batch, latent, target, x_dim + z_dim]

        return query

    def _decode(self,
        query: Union[TensorType[B, L, T, X + R + Z], TensorType[B, L, T, X + Z]],
    ) -> Tuple[TensorType[B, L, T, Y], TensorType[B, L, T, Y]]:

        mu_log_sigma = self.decoder(query)                                      # [batch x latent, target, y_dim * 2]

        y_dim = mu_log_sigma.shape[-1] // 2
        mu, log_sigma = torch.split(mu_log_sigma, (y_dim, y_dim), dim=-1)       # [batch, latent, target, y_dim] * 2
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)                               # [batch, latent, target, y_dim]

        return mu, sigma

    # Forward

    def forward(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
        num_latents: int = 1,
    ) -> Tuple[TensorType[B, L, T, Y], TensorType[B, L, T, Y]]:

        # Encode
        context = torch.cat((x_context, y_context), dim=-1)                     # [batch, context, x_dim + y_dim]
        z_i_context, r_i_context = self._encode(context)                        # [batch, context, z_dim], [batch, context, r_dim]

        # Latent representation
        z_dist = self._latent_dist(z_i_context)                                 # [batch, 1, z_dim]
        z_samples = self.sample_fn(z_dist, num_latents)                         # [batch, latent, 1, z_dim]

        # Deterministic representation
        if r_i_context is not None:
            r_context = self._determ_aggregate(r_i_context, x_context, x_target)# [batch, target, r_dim])
        else:
            r_context = None

        # Decode
        query = self._build_query(x_target, z_samples, r_context, num_latents)  # [batch, latent, target, x_dim (+ r_dim) + z_dim]
        mu, sigma = self._decode(query)                                         # [batch, latent, target, y_dim] * 2

        return mu, sigma

    # Likelihood and Loss

    def log_likelihood(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
        num_latents: int = 1,
    ) -> TensorType[float]:

        mu, sigma = self(x_context, y_context, x_target, num_latents)           # [batch, latent, target, y_dim] * 2
        log_likelihood = self.log_likelihood_fn(y_target, mu, sigma)            # [batch, latent, target]

        log_likelihood = torch.mean(log_likelihood, dim=-1)                     # [batch, latent]
        log_likelihood = torch.logsumexp(log_likelihood, dim=-1) \
                       - math.log(num_latents)                                  # [batch]
        log_likelihood = torch.mean(log_likelihood)                             # [1]

        return log_likelihood

    def vi_loss(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
        num_latents: int = 1,
    ) -> TensorType[float]:

        # Encode
        context = torch.cat((x_context, y_context), dim=-1)                     # [batch, context, x_dim + y_dim]
        target  = torch.cat((x_target, y_target), dim=-1)                       # [batch, target, x_dim + y_dim]
        data    = torch.cat((context, target), dim=1)                           # [batch, context + target, x_dim + y_dim]))

        z_i_context, r_i_context = self._encode(context)                        # [batch, context, z_dim], [batch, context, r_dim]
        z_i_data = self._latent_encode_only(data)                               # [batch, context + target, z_dim]

        # Latent representation
        z_context = self._latent_dist(z_i_context)                              # [batch, 1, z_dim]
        z_data    = self._latent_dist(z_i_data)                                 # [batch, 1, z_dim]

        z_samples = self.sample_fn(z_context, num_latents)                      # [batch, latent, 1, z_dim]

        # Deterministic representation
        if r_i_context is not None:
            r_context = self._determ_aggregate(r_i_context, x_context, x_target)# [batch, target, r_dim])
        else:
            r_context = None

        # Decode
        query = self._build_query(x_target, z_samples, r_context, num_latents)  # [batch, latent, target, x_dim (+ r_dim) + z_dim]
        mu, sigma = self._decode(query)                                         # [batch, latent, target, y_dim] * 2

        # Loss
        log_likelihood = self.log_likelihood_fn(y_target, mu, sigma)            # [batch, latent, target]
        log_likelihood = torch.mean(log_likelihood)                             # [1]

        kl_divergence = self.kl_divergence_fn(z_data, z_context)                # [batch, 1, z_dim]
        kl_divergence = torch.mean(kl_divergence)                               # [1]

        loss = -log_likelihood + kl_divergence                                  # [1]

        return loss


class NP(NPBase):
    """Neural Process"""

    def __init__(self,
        x_dim: int, y_dim: int,
        r_dim: int = 128, z_dim: int = 128,
        common_encoder_dims: Optional[List[int]] = None,
        latent_encoder_dims: Optional[List[int]] = [128, 128],
        determ_encoder_dims: Optional[List[int]] = [128, 128, 128, 128, 128],
        decoder_dims: List[int] = [128, 128, 128, 128],
        loss_type: str = "vi",
    ):

        if common_encoder_dims is not None:
            if r_dim != z_dim * 2:
                raise ValueError("Dimension mismatch: r_dim != z_dim * 2")

            latent_encoder = MLP(
                in_features=(x_dim + y_dim),
                hidden_features=common_encoder_dims,
                out_features=(z_dim * 2),
            )
            determ_encoder = latent_encoder

            decoder_input_dim = x_dim + r_dim + z_dim

        else:
            if latent_encoder_dims is None:
                raise ValueError("Invalid combination of encoders")

            latent_encoder = MLP(
                in_features=(x_dim + y_dim),
                hidden_features=latent_encoder_dims,
                out_features=(z_dim * 2),
            )

            if determ_encoder_dims is not None:
                determ_encoder = MLP(
                    in_features=(x_dim + y_dim),
                    hidden_features=determ_encoder_dims,
                    out_features=r_dim,
                )

                decoder_input_dim = x_dim + r_dim + z_dim

            else:
                determ_encoder = None
                decoder_input_dim = x_dim + z_dim

        decoder = MLP(
            in_features=decoder_input_dim,
            hidden_features=decoder_dims,
            out_features=(y_dim * 2),
        )

        super().__init__(
            latent_encoder=latent_encoder,
            determ_encoder=determ_encoder,
            decoder=decoder,
            loss_type=loss_type,
        )
