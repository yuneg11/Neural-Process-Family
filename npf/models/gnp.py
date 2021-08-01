from ..type import *

import math

import torch
from torch import nn

from .base import MultivariateNPF

from ..modules import (
    UNet,
    SimpleConvNet,
    Discretization1d,
    SetConv1dEncoder,
    SetConv1dDecoder,
    SetConv2dEncoder,
    SetConv2dDecoder,
)


__all__ = ["GNPBase", "GNP"]


class GNPBase(MultivariateNPF):
    """Gaussian Neural Process Base"""

    def __init__(self,
        discretizer,
        mean_encoder,
        mean_cnn,
        mean_decoder,
        kernel_encoder,
        kernel_cnn,
        kernel_decoder,
        init_log_sigma: float = 0.1,
        likelihood_type: str = "multivariate",
        loss_type: str = "multivariate",
        noise_eps: float = 1e-5,
    ):
        """
        Args:
            discretizer    : [[batch, context, x_dim]
                              [batch, target, x_dim]]
                          -> [1, discrete, 1]
            mean_encoder   : [[batch, discrete, x_dim]
                              [batch, context, x_dim]
                              [batch, context, y_dim]]
                          -> [batch, y_dim + 1, discrete]
            mean_cnn       : [batch, y_dim + 1, discrete]
                          -> [batch, y_dim, discrete]
            mean_decoder   : [[batch, target, x_dim]
                              [batch, discrete, x_dim]
                              [batch, y_dim, discrete]]
                          -> [batch, y_dim, target]
            kernel_encoder : [[batch, discrete, x_dim]
                              [batch, context, x_dim]
                              [batch, context, y_dim]]
                          -> [batch, y_dim + 1, discrete]
            kernel_cnn     : [batch, y_dim + 2, discrete]
                          -> [batch, y_dim, discrete]
            kernel_decoder : [[batch, target, x_dim]
                              [batch, discrete, x_dim]
                              [batch, y_dim, discrete]]
                          -> [batch, y_dim, target]
            init_log_sigma : float
            noise_eps      : float
        """
        super().__init__(
            likelihood_type=likelihood_type,
            loss_type=loss_type,
        )

        self.discretizer = discretizer

        self.mean_encoder = mean_encoder
        self.mean_cnn     = mean_cnn
        self.mean_decoder = mean_decoder

        self.kernel_encoder = kernel_encoder
        self.kernel_cnn     = kernel_cnn
        self.kernel_decoder = kernel_decoder

        self.log_sigma = nn.Parameter(
            torch.tensor(init_log_sigma, dtype=torch.float),
        )
        self.noise_eps = noise_eps

    def forward(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
        as_univariate: bool = False,
    ) -> Union[
        Tuple[TensorType[B, Y, T], TensorType[B, Y, T, T]],
        Tuple[TensorType[B, T, Y], TensorType[B, T, Y]],
    ]:

        # Discretize
        x_grid = self.discretizer(x_context, x_target)                          # [batch, discrete, x_dim]

        # Mean
        h_mu = self.mean_encoder(x_grid, x_context, y_context)                  # [batch, y_dim + 1, discrete]
        mu = self.mean_cnn(h_mu)                                                # [batch, y_dim, discrete]
        mu = self.mean_decoder(x_target, x_grid, mu)                            # [batch, y_dim, target]

        # Kernel
        h_cov = self.kernel_encoder(x_grid, x_context, y_context)               # [batch, y_dim + 2, discrete, discrete]
        cov = self.kernel_cnn(h_cov)                                            # [batch, y_dim, discrete, discrete]
        cov = self.kernel_decoder(x_target, x_grid, cov)                        # [batch, y_dim, target, target]

        identity = torch.eye(cov.shape[-1], device=cov.device)[None, None, :, :]# [1, 1, target, target]
        identity = identity.repeat(*cov.shape[:2], 1, 1)                        # [batch, y_dim, target, target]
        noise = identity * (torch.exp(self.log_sigma) + self.noise_eps)         # [batch, y_dim, target, target]
        #? Add a small epsilon to avoid numerical issues (Not as original impl.)

        cov = cov + noise                                                       # [batch, y_dim, target, target]

        if as_univariate:
            mu    = mu.transpose(1, 2)                                          # [batch, target, y_dim]
            sigma = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))           # [batch, y_dim, target]
            sigma = sigma.transpose(1, 2)                                       # [batch, target, y_dim]
            return mu, sigma
        else:
            return mu, cov


class GNP(GNPBase):
    def __init__(self,
        y_dim: int,
        mean_cnn_dims: Optional[List[int]] = None,
        mean_cnn_xl: bool = False,
        kernel_cnn_dims: Optional[List[int]] = None,
        kernel_cnn_xl: bool = False,
        points_per_unit: int = 64,
        discrete_margin: float = 0.1,
        init_log_sigma: float = 0.1,
        likelihood_type: str = "multivariate",
        loss_type: str = "multivariate",
        noise_eps: float = 1e-5,
    ):
        if mean_cnn_xl:
            MeanConvNet = UNet
            if mean_cnn_dims is None:
                mean_cnn_dims = [8, 16, 16, 32, 32, 64]
            mean_num_halving_layers = len(mean_cnn_dims)
        else:
            MeanConvNet = SimpleConvNet
            if mean_cnn_dims is None:
                mean_cnn_dims = [16, 32, 16]
            mean_num_halving_layers = 0

        if kernel_cnn_xl:
            KernelConvNet = UNet
            if kernel_cnn_dims is None:
                kernel_cnn_dims = [8, 16, 16, 32, 32, 64]
            kernel_num_halving_layers = len(kernel_cnn_dims)
        else:
            KernelConvNet = SimpleConvNet
            if kernel_cnn_dims is None:
                kernel_cnn_dims = [16, 32, 16]
            kernel_num_halving_layers = 0

        init_log_scale = math.log(2.0 / points_per_unit)
        multiple = 2 ** max(mean_num_halving_layers, kernel_num_halving_layers)

        discretizer = Discretization1d(
            points_per_unit=points_per_unit,
            multiple=multiple,
            margin=discrete_margin,
        )

        mean_encoder = SetConv1dEncoder(
            init_log_scale=init_log_scale,
        )

        mean_cnn = MeanConvNet(
            dimension=1,
            in_channels=(y_dim + 1),
            hidden_channels=mean_cnn_dims,
            out_channels=y_dim,
        )

        mean_decoder = SetConv1dDecoder(
            init_log_scale=init_log_scale,
            dim_last=False,
        )

        kernel_encoder = SetConv2dEncoder(
            init_log_scale=init_log_scale,
        )

        kernel_cnn = KernelConvNet(
            dimension=2,
            in_channels=(y_dim + 2),
            hidden_channels=kernel_cnn_dims,
            out_channels=y_dim,
        )

        kernel_decoder = SetConv2dDecoder(
            init_log_scale=init_log_scale,
        )

        super().__init__(
            discretizer=discretizer,
            mean_encoder=mean_encoder,
            mean_cnn=mean_cnn,
            mean_decoder=mean_decoder,
            kernel_encoder=kernel_encoder,
            kernel_cnn=kernel_cnn,
            kernel_decoder=kernel_decoder,
            init_log_sigma=init_log_sigma,
            likelihood_type=likelihood_type,
            loss_type=loss_type,
            noise_eps=noise_eps,
        )
