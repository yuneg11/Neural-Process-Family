from typing import List, Tuple, Optional
from torchtyping import TensorType
from ..type import *

import math

import torch
from torch.nn import functional as F
from torch.distributions import (
    Normal,
    kl_divergence as KLDivergence,
)

from .base import LatentNPF

from ..modules import (
    MLP,
    LogLikelihood,
)


__all__ = ["NPBase", "NP"]


class NPBase(LatentNPF):
    """Neural Process Base"""

    def __init__(
        self,
        decoder,
        deterministic_encoder=None,
        latent_encoder=None,
        common_encoder=None,
        loss_type: str = "vi",
    ):
        """
        Args:
            deterministic_encoder : [batch, context, x_dim + y_dim]
                                 -> [batch, context, r_dim]
            latent_encoder        : [batch, context, x_dim + y_dim]
                                 -> [batch, context, z_dim * 2]
            common_encoder        : [batch, context, x_dim + y_dim]
                                 -> [batch, context, r_dim = z_dim * 2]
            decoder               : [batch,  latent, target, x_dim (+ r_dim) + z_dim]
                                 -> [batch,  latent, target, y_dim * 2]
            loss_type             : str ("vi" or "ml")
        """
        super().__init__()

        if common_encoder is not None and \
            latent_encoder is None and deterministic_encoder is None:
            self.common_encoder = common_encoder
            self.latent_encoder = None
            self.deterministic_encoder = None
            self.deterministic_path = True
        elif common_encoder is None and \
            latent_encoder is not None and deterministic_encoder is None:
            self.common_encoder = None
            self.latent_encoder = latent_encoder
            self.deterministic_encoder = None
            self.deterministic_path = False
        elif common_encoder is None and \
            latent_encoder is not None and deterministic_encoder is not None:
            self.common_encoder = None
            self.latent_encoder = latent_encoder
            self.deterministic_encoder = deterministic_encoder
            self.deterministic_path = True
        else:
            raise ValueError("Invalid combination of encoders")

        self.decoder = decoder

        self.log_likelihood_fn = LogLikelihood()

        if loss_type is None:
            loss_type = "vi"

        self.loss_type = loss_type

        if loss_type == "vi":
            self.loss = self.vi_loss
        elif loss_type == "ml":
            self.loss = self.ml_loss
        else:
            raise ValueError(f"Invalid loss type: '{loss_type}'")

    def _deterministic_encode(self,
        context: TensorType[B, C, X + Y],
    ) -> TensorType[B, 1, R]:

        if self.common_encoder is not None:
            r_i_context = self.common_encoder(context)                          # [batch, context, r_dim]
        elif self.deterministic_encoder is not None:
            r_i_context = self.deterministic_encoder(context)                   # [batch, context, r_dim]
        else:
            raise ValueError("Invalid encoder")

        r_context = r_i_context.mean(dim=1, keepdim=True)                       # [batch, 1, r_dim]

        return r_context

    def _latent_encode(self,
        input_set: TensorType[B, C, X + Y],
    ) -> Normal:  # [B, 1, Z]

        if self.common_encoder is not None:
            z_i = self.common_encoder(input_set)                                # [batch, points, z_dim * 2]
        elif self.latent_encoder is not None:
            z_i = self.latent_encoder(input_set)                                # [batch, points, z_dim * 2]
        else:
            raise ValueError("Invalid encoder")

        mu_log_sigma = torch.mean(z_i, dim=1, keepdim=True)                     # [batch, 1, z_dim * 2]

        z_dim = mu_log_sigma.shape[-1] // 2
        mu, log_sigma = torch.split(mu_log_sigma, (z_dim, z_dim), dim=-1)       # [batch, 1, z_dim] * 2
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)

        z_dist = Normal(loc=mu, scale=sigma)                                    # [batch, 1, z_dim]

        return z_dist

    def _decode(self,
        query: TensorType[B, L, T, Q],
    ) -> Tuple[TensorType[B, L, T, Y], TensorType[B, L, T, Y]]:

        mu_log_sigma = self.decoder(query)                                      # [batch x latent, target, y_dim * 2]

        y_dim = mu_log_sigma.shape[-1] // 2
        mu, log_sigma = torch.split(mu_log_sigma, (y_dim, y_dim), dim=-1)       # [batch, latent, target, y_dim] * 2
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        return mu, sigma

    def forward(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
        num_latents: int = 1,
    ) -> Tuple[TensorType[B, L, T, Y], TensorType[B, L, T, Y]]:
        num_targets = x_target.shape[1]

        context = torch.cat((x_context, y_context), dim=-1)                     # [batch, context, x_dim + y_dim]

        # Deterministic Encode
        if self.deterministic_path:
            r_context = self._deterministic_encode(context)                     # [batch, 1, r_dim]

        # Latent Encode
        z_dist = self._latent_encode(context)                                   # [batch, 1, z_dim]
        z_samples = z_dist.rsample([num_latents]).transpose(1, 0)               # [batch, latent, 1, z_dim]
        z_samples = z_samples.repeat(1, 1, num_targets, 1)                      # [batch, latent, target, z_dim]

        # Decode
        x_target = x_target[:, None, :, :]                                      # [batch, 1, target, r_dim]
        x_target = x_target.repeat(1, num_latents, 1, 1)                        # [batch, latent, target, x_dim]

        if self.deterministic_path:
            r_context = r_context[:, None, :, :]                                # [batch, 1, 1, r_dim]
            r_context = r_context.repeat(1, num_latents, num_targets, 1)        # [batch, latent, target, r_dim]
            query = torch.cat((x_target, r_context, z_samples), dim=-1)         # [batch, latent, target, x_dim + r_dim + z_dim]
        else:
            query = torch.cat((x_target, z_samples), dim=-1)                    # [batch, latent, target, x_dim + z_dim]

        mu, sigma = self._decode(query)                                         # [batch, latent, target, y_dim] * 2

        return mu, sigma

    def log_likelihood(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
        num_latents: int = 1,
    ) -> TensorType[float]:

        mu, sigma = self(x_context, y_context, x_target, num_latents)
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

        num_targets = x_target.shape[1]

        context = torch.cat((x_context, y_context), dim=-1)                     # [batch, context, x_dim + y_dim]
        target  = torch.cat(( x_target,  y_target), dim=-1)                     # [batch, target, x_dim + y_dim]
        data    = torch.cat((  context,    target), dim=1)                      # [batch, context + target, x_dim + y_dim]))

        # Deterministic Encode
        if self.deterministic_path:
            r_context = self._deterministic_encode(context)                     # [batch, 1, r_dim]

        # Latent Encode
        z_context = self._latent_encode(context)                                # [batch, 1, z_dim]
        z_data    = self._latent_encode(data)                                   # [batch, 1, z_dim]

        kl_divergence = KLDivergence(z_data, z_context).mean()                  # [1]

        z_samples = z_context.rsample([num_latents]).transpose(1, 0)            # [batch, latent, 1, z_dim]
        z_samples = z_samples.repeat(1, 1, num_targets, 1)                      # [batch, latent, target, z_dim]

        # Decode
        x_target = x_target[:, None, :, :]                                      # [batch, 1, target, r_dim]
        x_target = x_target.repeat(1, num_latents, 1, 1)                        # [batch, latent, target, x_dim]

        if self.deterministic_path:
            r_context = r_context[:, None, :, :]                                # [batch, 1, 1, r_dim]
            r_context = r_context.repeat(1, num_latents, num_targets, 1)        # [batch, latent, target, r_dim]
            query = torch.cat((x_target, r_context, z_samples), dim=-1)         # [batch, latent, target, x_dim + r_dim + z_dim]
        else:
            query = torch.cat((x_target, z_samples), dim=-1)                    # [batch, latent, target, x_dim + z_dim]

        mu, sigma = self._decode(query)                                         # [batch, latent, target, y_dim] * 2

        log_likelihood = self.log_likelihood_fn(y_target, mu, sigma).mean()     # [1]
        loss = -log_likelihood + kl_divergence                                  # [1]

        return loss

    def ml_loss(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
        num_latents: int = 1,
    ) -> TensorType[float]:

        log_likelihood = self.log_likelihood(x_context, y_context, x_target, y_target, num_latents)
        loss = -log_likelihood

        return loss


class NP(NPBase):
    """Neural Process"""

    def __init__(
        self,
        x_dim: int, y_dim: int,
        r_dim: int = 128, z_dim: int = 128,
        deterministic_encoder_dims: Optional[List[int]] = [128, 128, 128, 128, 128],
        latent_encoder_dims: Optional[List[int]] = [128, 128],
        common_encoder_dims: Optional[List[int]] = None,
        decoder_dims: List[int] = [128, 128, 128, 128],
        loss_type: str = "vi",
    ):
        encoders = {}

        if common_encoder_dims is not None:
            if z_dim != r_dim * 2:
                raise ValueError("r_dim = z_dim * 2")

            encoders["common_encoder"] = MLP(
                in_features=(x_dim + y_dim),
                hidden_features=common_encoder_dims,
                out_features=(r_dim * 2),
            )

            decoder_input_dim = x_dim + r_dim + z_dim

        else:
            if latent_encoder_dims is None:
                raise ValueError("Invalid combination of encoders")

            encoders["latent_encoder"] = MLP(
                in_features=(x_dim + y_dim),
                hidden_features=latent_encoder_dims,
                out_features=(z_dim * 2),
            )

            if deterministic_encoder_dims is not None:
                encoders["deterministic_encoder"] = MLP(
                    in_features=(x_dim + y_dim),
                    hidden_features=deterministic_encoder_dims,
                    out_features=r_dim,
                )

                decoder_input_dim = x_dim + r_dim + z_dim

            else:
                decoder_input_dim = x_dim + z_dim

        decoder = MLP(
            in_features=decoder_input_dim,
            hidden_features=decoder_dims,
            out_features=(y_dim * 2),
        )

        super().__init__(
            **encoders,
            decoder=decoder,
            loss_type=loss_type,
        )
