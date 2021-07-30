from ..type import *

import math

import torch
from torch.nn import functional as F

from .base import ConditionalNPF

from ..modules import (
    UNet,
    SimpleConvNet,
    Discretization1d,
    SetConv1dEncoder,
    SetConv1dDecoder,
    LogLikelihood,
)


__all__ = ["ConvCNPBase", "ConvCNP"]


class ConvCNPBase(ConditionalNPF):
    """Convolutional Conditional Neural Process Base"""

    def __init__(self,
        discretizer,
        encoder,
        cnn,
        decoder,
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
            cnn         : [batch, y_dim + 1, discrete]
                       -> [batch, y_dim * 2, discrete]
            decoder     : [[batch, target, x_dim]
                           [batch, discrete, x_dim]
                           [batch, discrete, y_dim]]
                       -> [batch, target, y_dim]
        """
        super().__init__()

        self.discretizer = discretizer
        self.encoder = encoder
        self.cnn = cnn
        self.decoder = decoder

        self.log_likelihood_fn = LogLikelihood()

    def forward(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
    ) -> Tuple[TensorType[B, T, Y], TensorType[B, T, Y]]:

        # Discretize
        x_grid = self.discretizer(x_context, x_target)                          # [batch, discrete, x_dim]

        # Encode
        h = self.encoder(x_grid, x_context, y_context)                          # [batch, discrete, y_dim + 1]
        h = h.transpose(2, 1)                                                   # [batch, y_dim + 1, discrete]

        # Convolution
        mu_log_sigma = self.cnn(h)                                              # [batch, y_dim * 2, discrete]
        mu_log_sigma = mu_log_sigma.transpose(2, 1)                             # [batch, discrete, y_dim * 2]

        y_dim = mu_log_sigma.shape[-1] // 2
        mu, log_sigma = torch.split(mu_log_sigma, (y_dim, y_dim), dim=-1)       # [batch, discrete, y_dim] * 2
        log_sigma = F.softplus(log_sigma)                                       # [batch, target, y_dim]

        # Decode
        mu    = self.decoder(x_target, x_grid, mu)                              # [batch, target, y_dim]
        sigma = self.decoder(x_target, x_grid, log_sigma)                       # [batch, target, y_dim]

        return mu, sigma

    def log_likelihood(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
    ) -> TensorType[float]:

        mu, sigma = self(x_context, y_context, x_target)
        log_likelihood = self.log_likelihood_fn(y_target, mu, sigma)
        log_likelihood = torch.mean(log_likelihood)
        return log_likelihood


class ConvCNP(ConvCNPBase):
    def __init__(self,
        y_dim: int,
        cnn_dims: Optional[List[int]] = None,
        cnn_xl: bool = False,
        points_per_unit: int = 64,
        discrete_margin: float = 0.1,
    ):
        if cnn_xl:
            ConvNet = UNet
            if cnn_dims is None:
                cnn_dims = [8, 16, 16, 32, 32, 64]
            num_halving_layers = len(cnn_dims)
        else:
            ConvNet = SimpleConvNet
            if cnn_dims is None:
                cnn_dims = [16, 32, 16]
            num_halving_layers = 0

        init_log_scale = math.log(2.0 / points_per_unit)
        multiple = 2 ** num_halving_layers

        discretizer = Discretization1d(
            points_per_unit=points_per_unit,
            multiple=multiple,
            margin=discrete_margin,
        )

        encoder = SetConv1dEncoder(
            init_log_scale=init_log_scale,
        )

        cnn = ConvNet(
            dimension=1,
            in_channels=(y_dim + 1),
            hidden_channels=cnn_dims,
            out_channels=(y_dim * 2),
        )

        decoder = SetConv1dDecoder(
            init_log_scale=init_log_scale,
        )

        super().__init__(
            discretizer=discretizer,
            encoder=encoder,
            cnn=cnn,
            decoder=decoder,
        )
