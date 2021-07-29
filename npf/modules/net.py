from typing import List

import torch
from torch import nn


__all__ = [
    "MLP",
    "UNet",
    "SimpleConvNet",
]


def init_layer(layer: nn.Module):
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.xavier_normal_(layer.weight, gain=1)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, 1e-3)


def init_model(model: nn.Module):
    for layer in model.children():
        init_layer(layer)


class MLP(nn.Sequential):
    def __init__(self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
        last_activation: bool = False,
    ):
        layers = []

        for input_dim, output_dim in zip([in_features] + hidden_features, hidden_features):
            layers.extend([
                nn.Linear(input_dim, output_dim, bias=bias),
                nn.ReLU(),
            ])

        if not hidden_features:
            raise ValueError("'hidden_features' is empty")

        layers.append(nn.Linear(hidden_features[-1], out_features, bias=bias))

        if last_activation:
            layers.append(nn.ReLU())

        super().__init__(*layers)

        init_model(self)


class UNet(nn.Module):
    def __init__(self,
        dimension: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int] = [8, 16, 16, 32, 32, 64],
    ):

        super(UNet, self).__init__()

        self.activation = nn.ReLU()
        # self.num_halving_layers = len(hidden_channels)

        if dimension == 1:
            Conv = nn.Conv1d
            ConvTranspose = nn.ConvTranspose1d
        elif dimension == 2:
            Conv = nn.Conv2d
            ConvTranspose = nn.ConvTranspose2d
        else:
            raise ValueError(f"Invalid dimension: {dimension}")

        # First linear layer:
        self.initial_linear = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels[0],
            kernel_size=1,
            padding=0,
            stride=1,
        )
        init_layer(self.initial_linear)

        # Final linear layer:
        self.final_linear = Conv(
            in_channels=hidden_channels[0],
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=1,
        )
        init_layer(self.final_linear)

        # Before turn layers:
        kernel_size = 5
        padding = kernel_size // 2
        self.before_turn_layers = nn.Sequential(
            *[
                Conv(
                    in_channels=hidden_channels[max(i - 1, 0)],
                    out_channels=hidden_channels[i],
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=2,
                )
                for i in range(len(hidden_channels))
            ]
        )
        init_model(self.before_turn_layers)

        # After turn layers:

        def get_num_in_channels(i):
            if i == len(hidden_channels) - 1:
                # No skip connection yet.
                return hidden_channels[i]
            else:
                # Add the skip connection.
                return 2 * hidden_channels[i]

        self.after_turn_layers = nn.Sequential(
            *[
                ConvTranspose(
                    in_channels=get_num_in_channels(i),
                    out_channels=hidden_channels[max(i - 1, 0)],
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=2,
                    output_padding=1,
                )
                for i in range(len(hidden_channels))
            ]
        )
        init_model(self.after_turn_layers)

    def forward(self, x):
        h = self.initial_linear(x)

        hs = [self.activation(self.before_turn_layers[0](h))]
        for layer in self.before_turn_layers[1:]:
            hs.append(self.activation(layer(hs[-1])))

        # Now make the turn!

        h = self.activation(self.after_turn_layers[-1](hs[-1]))
        for h_prev, layer in zip(
            reversed(hs[:-1]), reversed(self.after_turn_layers[:-1])
        ):
            h = self.activation(layer(torch.cat((h_prev, h), axis=1)))

        return self.final_linear(h)


class SimpleConvNet(nn.Sequential):
    def __init__(self,
        dimension: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int] = [16, 32, 16],
    ):
        if dimension == 1:
            Conv = nn.Conv1d
        elif dimension == 2:
            Conv = nn.Conv2d
        else:
            raise ValueError(f"Invalid dimension: {dimension}")

        layers = []

        for input_dim, output_dim in zip([in_channels] + hidden_channels, hidden_channels):
            layers.extend([
                Conv(in_channels=input_dim, out_channels=output_dim,
                     kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
            ])

        if not hidden_channels:
            raise ValueError("'hidden_features' is empty")

        layers.append(Conv(in_channels=hidden_channels[-1],
                           out_channels=out_channels,
                           kernel_size=5, stride=1, padding=2))

        super().__init__(*layers)

        init_model(self)
        # self.num_halving_layers = 0
