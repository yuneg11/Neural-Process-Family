from ..type import *

import abc
import math

import torch
from torch import nn

from .base import NPF

from ..modules import (
    MLP,
    MultiheadCrossAttention,
    UNet,
    SimpleConvNet,
    Discretization1d,
    SetConv1dEncoder,
    SetConv1dDecoder,
    SetConv2dEncoder,
    SetConv2dDecoder,
)
from ..modules.nflows import *


__all__ = [
    "get_model",
    "FlowNPBase",
]


def get_model(model_name, **model_kwargs):
    model_name = model_name.lower()


    transform1 = NaiveLinear(1)
    transform2 = CompositeTransform(
        NaiveLinear(1),
        LeakyReLU(),
        NaiveLinear(1),
    )
    transform3 = CompositeTransform(
        NaiveLinear(1),
        LeakyReLU(),
        NaiveLinear(1),
        LeakyReLU(),
        NaiveLinear(1),
    )
    transform4 = CompositeTransform(
        NaiveLinear(1),
        Tanh(),
        NaiveLinear(1),
    )
    transform5 = CompositeTransform(
        NaiveLinear(1),
        Tanh(),
        NaiveLinear(1),
        Tanh(),
        NaiveLinear(1),
    )
    transform6 = CompositeTransform(
        NaiveLinear(1),
        Tanh(),
        NaiveLinear(1),
        Tanh(),
        NaiveLinear(1),
        Tanh(),
        NaiveLinear(1),
    )
    transform7 = CompositeTransform(
        NaiveLinear(1),
        Sigmoid(),
        NaiveLinear(1),
        Sigmoid(),
        NaiveLinear(1),
    )
    transform8 = Affine(1)
    transform9 = CompositeTransform(
        Affine(1),
        Sigmoid(),
        Affine(1),
    )

    transform = transform9

    if model_name == "flownp":
        model = FlowNP(
            x_dim=1, y_dim=1, r_dim=128,
            transform=transform,
            base_dist=StandardNormal(),
        )
    elif model_name == "flowattnnp":
        model = FlowAttnNP(
            x_dim=1, y_dim=1, r_dim=128,
            transform=transform,
            base_dist=StandardNormal(),
        )
    elif model_name == "flowconvnp":
        model = FlowConvNP(
            y_dim=1,
            cnn_xl=False,
            transform=transform,
            base_dist=StandardNormal(),
        )
    elif model_name == "flowconvnpxl":
        model = FlowConvNP(
            y_dim=1,
            cnn_xl=True,
            transform=transform,
            base_dist=StandardNormal(),
        )
    elif model_name == "flowgnp":
        model = FlowGNP(
            y_dim=1,
            cnn_xl=False,
            transform=transform,
            base_dist=StandardNormal(),
        )
    else:
        raise ValueError(f"Unsupported model: '{model_name}'")

    return model


class FlowNPBase(NPF):
    """
    Base class for Flow NPF models
    """

    is_multivariate_model: bool = False
    is_latent_model: bool = False

    def __init__(self,
        transform: Transform,
        base_dist: Distribution,
    ):
        super().__init__()
        self.flow = Flow(transform, base_dist)

    @abc.abstractmethod
    def _encode_params(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
    ) -> TensorType[B, T, P]:
        raise NotImplementedError

    def forward(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
    ) -> TensorType[B, T, Y]:
        raise NotImplementedError

    def log_likelihood(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
    ) -> TensorType[float]:

        params = self._encode_params(x_context, y_context, x_target)            # [param]
        log_prob = self.flow(params).log_prob(y_target)                         # [batch] ?[batch, target]
        log_likelihood = torch.mean(log_prob)                                   # [1]
        return log_likelihood

    def loss(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
    ) -> TensorType[float]:

        log_likelihood = \
            self.log_likelihood(x_context, y_context, x_target, y_target)
        loss = -log_likelihood
        return loss


class FlowNP(FlowNPBase):
    def __init__(self,
        x_dim: int, y_dim: int, r_dim: int,
        transform: Transform,
        base_dist: Distribution,
    ):
        super().__init__(
            transform=transform,
            base_dist=base_dist,
        )

        param_dim = self.flow.param_dim

        context_encoder_dims = [128, 128]
        target_encoder_dims  = [64, 64]

        context_encoder = MLP(
            in_features=(x_dim + y_dim),
            hidden_features=context_encoder_dims,
            out_features=r_dim,
        )

        target_encoder = MLP(
            in_features=(x_dim + r_dim),
            hidden_features=target_encoder_dims,
            out_features=param_dim,
        )

        self.context_encoder = context_encoder
        self.target_encoder  = target_encoder

    def _encode_params(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
    ) -> TensorType[B, T, P]:

        # Context encode
        context = torch.cat((x_context, y_context), dim=-1)                     # [batch, context, x_dim + y_dim]
        r_i_context = self.context_encoder(context)                             # [batch, context, r_dim]

        # Aggregate
        r_context = torch.mean(r_i_context, dim=1, keepdim=True)                # [batch, 1, r_dim]
        r_context = r_context.repeat(1, x_target.shape[1], 1)                   # [batch, target, r_dim]

        # Target encode
        query = torch.cat((x_target, r_context), dim=-1)                        # [batch, target, x_dim + r_dim]
        params = self.target_encoder(query)                                     # [batch, target, param]

        return params


class FlowAttnNP(FlowNPBase):
    def __init__(self,
        x_dim: int, y_dim: int, r_dim: int,
        transform: Transform,
        base_dist: Distribution,
        ca_heads: Optional[int] = 8,
    ):
        super().__init__(
            transform=transform,
            base_dist=base_dist,
        )

        param_dim = self.flow.param_dim

        context_encoder_dims = [128, 128]
        target_encoder_dims  = [64, 64]

        context_encoder = MLP(
            in_features=(x_dim + y_dim),
            hidden_features=context_encoder_dims,
            out_features=r_dim,
        )

        target_encoder = MLP(
            in_features=(x_dim + r_dim),
            hidden_features=target_encoder_dims,
            out_features=param_dim,
        )

        cross_attention = MultiheadCrossAttention(
            num_heads=ca_heads,
            qk_dim=x_dim,
            v_dim=r_dim,
        )

        self.context_encoder = context_encoder
        self.target_encoder  = target_encoder
        self.cross_attention = cross_attention

    def _encode_params(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
    ) -> TensorType[B, T, P]:

        # Context encode
        context = torch.cat((x_context, y_context), dim=-1)                     # [batch, context, x_dim + y_dim]
        r_i_context = self.context_encoder(context)                             # [batch, context, r_dim]

        # Aggregate
        r_context = self.cross_attention(x_target, x_context, r_i_context)      # [batch, target, r_dim]

        # Target encode
        query = torch.cat((x_target, r_context), dim=-1)                        # [batch, target, x_dim + r_dim]
        params = self.target_encoder(query)                                     # [batch, target, param]

        return params


class FlowConvNP(FlowNPBase):
    def __init__(self,
        y_dim: int,
        transform: Transform,
        base_dist: Distribution,
        cnn_dims: Optional[List[int]] = None,
        cnn_xl: bool = False,
        points_per_unit: int = 64,
        discrete_margin: float = 0.1,
    ):
        super().__init__(
            transform=transform,
            base_dist=base_dist,
        )

        param_dim = self.flow.param_dim

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
            out_channels=param_dim,
        )

        decoder = SetConv1dDecoder(
            init_log_scale=init_log_scale,
            dim_last=True,
        )

        self.discretizer = discretizer
        self.encoder     = encoder
        self.cnn         = cnn
        self.decoder     = decoder

    def _encode_params(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
    ) -> TensorType[B, T, P]:

        # Discretize
        x_grid = self.discretizer(x_context, x_target)                          # [batch, discrete, x_dim]

        # Encode
        h = self.encoder(x_grid, x_context, y_context)                          # [batch, y_dim + 1, discrete]

        # Convolution
        params = self.cnn(h)                                                    # [batch, y_dim * 2, discrete]
        params = self.decoder(x_target, x_grid, params)                         # [batch, target, y_dim]

        return params


class FlowGNP(FlowNPBase):
    def __init__(self,
        y_dim: int,
        transform: Transform,
        base_dist: Distribution,
        cnn_dims: Optional[List[int]] = None,
        cnn_xl: bool = False,
        points_per_unit: int = 64,
        discrete_margin: float = 0.1,
        init_log_sigma: float = 0.1,
        noise_eps: float = 0.,

    ):
        super().__init__(
            transform=transform,
            base_dist=base_dist,
        )

        param_dim = self.flow.param_dim

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

        encoder = SetConv2dEncoder(
            init_log_scale=init_log_scale,
        )

        cnn = ConvNet(
            dimension=2,
            in_channels=(y_dim + 2),
            hidden_channels=cnn_dims,
            out_channels=param_dim,
        )

        decoder = SetConv2dDecoder(
            init_log_scale=init_log_scale,
        )

        self.discretizer = discretizer

        self.encoder = encoder
        self.cnn     = cnn
        self.decoder = decoder

        self.log_sigma = nn.Parameter(
            torch.tensor(init_log_sigma, dtype=torch.float),
        )
        self.noise_eps = noise_eps

    def _encode_params(self,
        x_context: TensorType[B, C, X],
        y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X],
    ) -> Union[
        Tuple[TensorType[B, Y, T], TensorType[B, Y, T, T]],
        Tuple[TensorType[B, T, Y], TensorType[B, T, Y]],
    ]:

        # Discretize
        x_grid = self.discretizer(x_context, x_target)                          # [batch, discrete, x_dim]

        # Kernel
        h_cov = self.encoder(x_grid, x_context, y_context)                      # [batch, y_dim + 2, discrete, discrete]
        cov = self.cnn(h_cov)                                                   # [batch, y_dim, discrete, discrete]
        cov = self.decoder(x_target, x_grid, cov)                               # [batch, y_dim, target, target]

        identity = torch.eye(cov.shape[-1], device=cov.device)[None, None, :, :]# [1, 1, target, target]
        identity = identity.repeat(*cov.shape[:2], 1, 1)                        # [batch, y_dim, target, target]
        noise = identity * (torch.exp(self.log_sigma) + self.noise_eps)         # [batch, y_dim, target, target]
        #? Add a small epsilon to avoid numerical issues (Not as original impl.)

        params = cov + noise                                                    # [batch, y_dim, target, target]
        params = params.permute(0, 2, 3, 1)                                     # [batch, target, target, y_dim]

        return params

    def log_likelihood(self,
        x_context: TensorType[B, C, X], y_context: TensorType[B, C, Y],
        x_target:  TensorType[B, T, X], y_target:  TensorType[B, T, Y],
    ) -> TensorType[float]:

        params = self._encode_params(x_context, y_context, x_target)            # [param]
        log_prob = self.flow(params).log_prob(y_target)                         # [batch]
        log_likelihood = torch.mean(log_prob)                                   # [1]
        return log_likelihood
