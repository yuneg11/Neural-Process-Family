from torch import nn


__all__ = [
    "PointwiseLinear",
    "LatentPointwiseLinear",
    "PointwiseMLP",
    "LatentPointwiseMLP",
]


class PointwiseLinear(nn.Conv1d):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(
            in_channels=in_features,
            out_channels=out_features,
            bias=bias,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, input):
        return super().forward(input.transpose(2, 1)).transpose(2, 1)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.bias is not None
        )


class LatentPointwiseLinear(nn.Conv2d):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(
            in_channels=in_features,
            out_channels=out_features,
            bias=bias,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, input):
        return super().forward(input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.bias is not None
        )


class PointwiseMLPBase(nn.Sequential):
    def __init__(
        self,
        linear_factory,
        in_features,
        hidden_features,
        out_features,
        bias=True,
        activation_factory=nn.ReLU,
    ):
        layers = []

        for input_dim, output_dim in zip([in_features] + hidden_features, hidden_features):
            layers.extend([
                linear_factory(input_dim, output_dim, bias=bias),
                activation_factory(),
            ])

        layers.append(linear_factory(hidden_features[-1], out_features, bias=bias))

        super().__init__(*layers)

        for layer in self.children():
            if hasattr(layer, "weight"):
                nn.init.xavier_normal_(layer.weight, gain=1)
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.constant_(layer.bias, 1e-3)


class PointwiseMLP(PointwiseMLPBase):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        bias=True,
        activation_factory=nn.ReLU,
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
            linear_factory=PointwiseLinear,
            activation_factory=activation_factory,
        )


class LatentPointwiseMLP(PointwiseMLPBase):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        bias=True,
        activation_factory=nn.ReLU,
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
            linear_factory=LatentPointwiseLinear,
            activation_factory=activation_factory,
        )
