from ..type import *

import math

import torch
from torch.nn import functional as F
from torch.distributions import Normal

from .base import LatentNPF

from ..modules import (
    UNet,
    SimpleConvNet,
    Discretization1d,
    SetConv1dEncoder,
    SetConv1dDecoder,
    LogLikelihood,
    KLDivergence,
)


__all__ = ["ConvNPBase", "ConvNP"]


class ConvNPBase(LatentNPF):
    """Convolutional Neural Process Base"""

    def __init__(self,
        discretizer,
        encoder,
        determ_cnn,
        latent_cnn,
        decoder,
        loss_type: str = "ml",
    ):
        """
        Args:
            discretizer : [[batch, context, x_dim]
                           [batch, target, x_dim]]
                       -> [1, descrete, 1]
            encoder     : [[batch, descrete, x_dim]
                           [batch, context, x_dim]
                           [batch, context, y_dim]]
                       -> [batch, discrete, y_dim + 1]
            determ_cnn  : [batch, y_dim + 1, discrete]
                       -> [batch, z_dim * 2, discrete]
            latent_cnn  : [batch, latent, z_dim, discrete]
                       -> [batch, latent, y_dim * 2, discrete]
            decoder     : [[batch, latent, target, x_dim]
                           [batch, latent, discrete, x_dim]
                           [batch, latent, discrete, y_dim]]
                       -> [batch, latent, target, y_dim]
            loss_type   : str ("vi" or "ml")
        """
        super().__init__(
            loss_type=("ml" if loss_type is None else loss_type),
        )

        self.discretizer = discretizer
        self.encoder = encoder
        self.determ_cnn = determ_cnn
        self.latent_cnn = latent_cnn
        self.decoder = decoder

        self.log_likelihood_fn = LogLikelihood()
        self.kl_divergence_fn = KLDivergence()

    def _latent_dist(self,
        z: TensorType[B, Z * 2, D],
    ) -> TensorType[B, Z, D]:

        z_dim = z.shape[1] // 2
        mu, log_sigma = torch.split(z, (z_dim, z_dim), dim=1)                   # [batch, z_dim, discrete] * 2
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)                            # [batch, z_dim, discrete]
        z_dist = Normal(loc=mu, scale=sigma)                                    # [batch, z_dim, discrete]
        return z_dist

    def _latent_conv(self,
        z_samples: TensorType[L, B, Z, D],
    ) -> Tuple[TensorType[B, L, D, Y], TensorType[B, L, D, Y]]:

        num_latents, num_batches, z_dim, num_discretes = z_samples.shape
        z_samples = \
            z_samples.view(num_latents * num_batches, z_dim, num_discretes)     # [latent * batch, z_dim, discrete])

        mu_log_sigma = self.latent_cnn(z_samples)                               # [latent * batch, y_dim * 2, discrete]
        mu_log_sigma = \
            mu_log_sigma.view(num_latents, num_batches, -1, num_discretes)      # [latent, batch, y_dim * 2, discrete]
        mu_log_sigma = mu_log_sigma.permute(1, 0, 3, 2)                         # [batch, latent, discrete, y_dim * 2]

        y_dim = mu_log_sigma.shape[-1] // 2
        mu, log_sigma = torch.split(mu_log_sigma, (y_dim, y_dim), dim=-1)       # [batch, latent, discrete, y_dim] * 2
        sigma = F.softplus(log_sigma)

        return mu, sigma

    def forward(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
        num_latents: int = 1,
    ) -> Tuple[TensorType[B, L, T, Y], TensorType[B, L, T, Y]]:

        # Discretize
        x_grid = self.discretizer(x_context, x_target)                          # [batch, discrete, x_dim]

        # Encode
        h = self.encoder(x_grid, x_context, y_context)                          # [batch, discrete, y_dim + 1]
        h = h.transpose(-1, -2)                                                 # [batch, y_dim + 1, discrete]

        # Deterministic Convolution
        z_context_mu_log_sigma = self.determ_cnn(h)                             # [batch, z_dim * 2, discrete]

        # Latent Sample
        z_context = self._latent_dist(z_context_mu_log_sigma)                   # Normal[batch, z_dim, discrete]
        z_samples = z_context.rsample([num_latents])                            # [latent, batch, z_dim, discrete]

        # Latent Convolution
        mu, sigma = self._latent_conv(z_samples)                                # [batch, latent, discrete, y_dim] * 2

        # Decode
        mu    = self.decoder(x_target, x_grid, mu)                              # [batch, target, y_dim]
        sigma = self.decoder(x_target, x_grid, sigma)                           # [batch, target, y_dim]

        return mu, sigma

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

        x_data = torch.cat((x_context, x_target), dim=1)                        # [batch, context + target, x_dim]
        y_data = torch.cat((y_context, y_target), dim=1)                        # [batch, context + target, y_dim]

        # Discretize
        x_grid = self.discretizer(x_context, x_target)                          # [batch, discrete, x_dim]

        # Encode
        h_context = self.encoder(x_grid, x_context, y_context)                  # [batch, discrete, y_dim + 1]
        h_context = h_context.transpose(-1, -2)                                 # [batch, y_dim + 1, discrete]

        h_data = self.encoder(x_grid, x_data, y_data)                           # [batch, discrete, y_dim + 1]
        h_data = h_data.transpose(-1, -2)                                       # [batch, y_dim + 1, discrete]

        # Deterministic Convolution
        z_context_mu_log_sigma = self.determ_cnn(h_context)                     # [batch, z_dim * 2, discrete]
        z_data_mu_log_sigma    = self.determ_cnn(h_data)                        # [batch, z_dim * 2, discrete]

        # Latent Sample
        z_context = self._latent_dist(z_context_mu_log_sigma)                   # Normal[batch, z_dim, discrete]
        z_data    = self._latent_dist(z_data_mu_log_sigma)                      # Normal[batch, z_dim, discrete]

        z_samples = z_context.rsample([num_latents])                            # [latent, batch, z_dim, discrete]

        # Latent Convolution
        mu, sigma = self._latent_conv(z_samples)                                # [batch, latent, discrete, y_dim] * 2

        # Decode
        mu    = self.decoder(x_target, x_grid, mu)                              # [batch, target, y_dim]
        sigma = self.decoder(x_target, x_grid, sigma)                           # [batch, target, y_dim]

        # Loss
        log_likelihood = self.log_likelihood_fn(y_target, mu, sigma)            # [batch, latent, target]
        log_likelihood = torch.mean(log_likelihood)                             # [1]

        kl_divergence = self.kl_divergence_fn(z_data, z_context)                # [batch, z_dim, discrete]
        kl_divergence = torch.mean(kl_divergence)                               # [1]

        loss = -log_likelihood + kl_divergence                                  # [1]

        return loss


class ConvNP(ConvNPBase):
    def __init__(self,
        y_dim: int,
        z_dim: int,
        determ_cnn_dims: Optional[List[int]] = None,
        latent_cnn_dims: Optional[List[int]] = None,
        determ_cnn_xl: bool = False,
        latent_cnn_xl: bool = False,
        points_per_unit: int = 64,
        discrete_margin: float = 0.1,
        loss_type: str = "ml",
    ):
        if determ_cnn_xl:
            DetermConvNet = UNet
            if determ_cnn_dims is None:
                determ_cnn_dims = [8, 16, 16, 32, 32, 64]
            determ_num_halving_layers = len(determ_cnn_dims)
        else:
            DetermConvNet = SimpleConvNet
            if determ_cnn_dims is None:
                determ_cnn_dims = [16, 32, 16]
            determ_num_halving_layers = 0

        if latent_cnn_xl:
            LatentConvNet = UNet
            if latent_cnn_dims is None:
                latent_cnn_dims = [8, 16, 16, 32, 32, 64]
            latent_num_halving_layers = len(latent_cnn_dims)
        else:
            LatentConvNet = SimpleConvNet
            if latent_cnn_dims is None:
                latent_cnn_dims = [16, 32, 16]
            latent_num_halving_layers = 0

        init_log_scale = math.log(2.0 / points_per_unit)
        multiple = 2 ** max(determ_num_halving_layers, latent_num_halving_layers)

        discretizer = Discretization1d(
            points_per_unit=points_per_unit,
            multiple=multiple,
            margin=discrete_margin,
        )

        encoder = SetConv1dEncoder(
            init_log_scale=init_log_scale,
        )

        determ_cnn = DetermConvNet(
            dimension=1,
            in_channels=(y_dim + 1),
            hidden_channels=determ_cnn_dims,
            out_channels=(z_dim * 2),
        )

        latent_cnn = LatentConvNet(
            dimension=1,
            in_channels=z_dim,
            hidden_channels=latent_cnn_dims,
            out_channels=(y_dim * 2),
        )

        decoder = SetConv1dDecoder(
            init_log_scale=init_log_scale,
        )

        super().__init__(
            discretizer=discretizer,
            encoder=encoder,
            determ_cnn=determ_cnn,
            latent_cnn=latent_cnn,
            decoder=decoder,
            loss_type=loss_type,
        )




























"""
Node(inputs=["x_context", "x_target"],                                     Node(inputs=["x_context", "x_target"],
     outputs=["x_grid"],                                                        outputs=["x_grid"],
     func=self.discretizer),                                                    func=self.discretizer),
Node(inputs=["x_context", "y_context", "x_grid"],                          Node(inputs=["x_context", "y_context", "x_grid"],
     outputs=["representation"],                                                outputs=["representation"],
     func=self.encoder),                                                        func=self.encoder),

Node(inputs=["representation"],
     outputs=["f_representation"],
     func=self.decoder),

                                                                           Node(inputs=["representation"],
                                                                                outputs=["f_noise"],
                                                                                func=self.noise_decoder),
                                                                           Node(inputs=["f_noise"],
                                                                                outputs=["q_context"],
                                                                                func=self.distributor),
                                                                           Node(inputs=["q_context", "num_latents"],
                                                                                outputs=["z_samples"],
                                                                                func=self.sampler),
                                                                           Node(inputs=["z_samples"],
                                                                                outputs=["f_representation"],
                                                                                func=self.decoder),

Node(inputs=["x_grid", "f_representation", "x_target"],                    Node(inputs=["x_grid", "f_representation", "x_target"],
     outputs=["mu"],                                                            outputs=["mu"],
     func=self.mu_set_conv),                                                    func=self.mu_set_conv),
Node(inputs=["x_grid", "f_representation", "x_target"],                    Node(inputs=["x_grid", "f_representation", "x_target"],
     outputs=["sigma"],                                                         outputs=["sigma"],
     func=self.sigma_set_conv),                                                 func=self.sigma_set_conv),
Node(inputs=["mu", "sigma", "y_target"],                                   Node(inputs=["mu", "sigma", "y_target"],
     outputs=["log_likelihood"],                                                outputs=["log_likelihood"],
     func=self.log_likelihood),                                                 func=self.log_likelihood),
Node(inputs=["log_likelihood", "x_target"],                                Node(inputs=["log_likelihood", "x_target"],
     outputs=["loss"],                                                          outputs=["loss"],
     func=self.loss_function),                                                  func=self.loss_function),
"""