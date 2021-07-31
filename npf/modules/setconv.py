from typing import Union
from torchtyping import TensorType
from ..type import *

import abc
import math

import torch
from torch import nn


__all__ = [
    "Discretization1d",
    "SetConv1dEncoder",
    "SetConv1dDecoder",
    # "SetConv2dEncoder",
    # "SetConv2dDecoder",
]


def to_multiple(x, multiple):
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple


class Discretization1d(nn.Module):
    def __init__(self,
        points_per_unit: int,
        multiple: int,
        margin: float = 0.1,
    ):
        super().__init__()

        self.points_per_unit = points_per_unit
        self.multiple = multiple
        self.margin = margin

    def forward(self,
        x_context: TensorType[B, C, X],
        x_target:  TensorType[B, T, X],
    ) -> TensorType[1, D, 1]:

        with torch.no_grad():
            x_min = min(torch.min(x_context).item(), torch.min(x_target).item())
            x_max = max(torch.max(x_context).item(), torch.max(x_target).item())

            grid_min = x_min - self.margin - (1 / self.points_per_unit)
            grid_max = x_max + self.margin + (1 / self.points_per_unit)

            raw_points = self.points_per_unit * (grid_max - grid_min)

            if math.isclose(raw_points % self.multiple, 0):
                num_points = int(raw_points)
            else:
                num_points = int(raw_points + self.multiple - raw_points % self.multiple)

            x_grid = torch.linspace(grid_min, grid_max, num_points, device=x_context.device)
            x_grid = x_grid[None, :, None]

        return x_grid


class SetConvBase(nn.Module):
    def __init__(self,
        init_log_scale: float = 1.0,
    ):
        super().__init__()
        self.log_scale = nn.Parameter(
            torch.tensor(init_log_scale, dtype=torch.float),
        )

    @abc.abstractmethod
    def forward(self,
        query: TensorType[B, T, Q],
        key:   TensorType[B, S, K],
        value: TensorType[B, S, V],
    ) -> Union[TensorType[B, T, V + 1], TensorType[B, T, V]]:
        """
        Args:
            query: location to interpolate
            key:   location of value
            value: value to be interpolated

        Returns:
            interpolated_value
        """

    @staticmethod
    def _get_distance(
        a: TensorType[B, N, X],
        b: TensorType[B, M, X],
    ) -> TensorType[B, N, M]:

        if a.shape[-1] == 1 and b.shape[-1] == 1:
            distance = (a - b.transpose(2, 1)) ** 2
        else:
            a_norm = torch.sum(a ** 2, axis=-1)[..., :, None]
            b_norm = torch.sum(b ** 2, axis=-1)[..., None, :]
            distance = a_norm + b_norm - 2 * torch.matmul(a, b.transpose(-1, -2))
        return distance

    def _get_weight(self,
        a: TensorType[B, N, X],
        b: TensorType[B, M, X],
    ) -> TensorType[B, N, M]:

        distance = self._get_distance(a, b)
        weight = torch.exp(-0.5 * distance / torch.exp(2 * self.log_scale))
        return weight


class SetConv1dEncoder(SetConvBase):
    def forward(self,
        query: TensorType[B, T, QK],
        key:   TensorType[B, S, QK],
        value: TensorType[B, S, V],
    ) -> TensorType[B, T, V + 1]:

        density = torch.ones((*value.shape[:-1], 1), device=value.device)       # [batch, source, 1]
        value = torch.cat((density, value), dim=-1)                             # [batch, source, v_dim + 1]

        weight = self._get_weight(query, key)                                   # [batch, target, source]
        value = torch.matmul(weight, value)                                     # [batch, target, v_dim + 1]
        value = torch.cat((                                                     # [batch, target, v_dim + 1]
            value[..., :1],
            value[..., 1:] / (value[..., :1] + 1e-8)
        ), axis=-1)

        return value


class SetConv1dDecoder(SetConvBase):
    def forward(self,
        query: TensorType[B, T, QK],
        key:   TensorType[B, S, QK],
        value: Union[TensorType[B, S, V], TensorType[B, L, S, V]],
    ) -> Union[TensorType[B, T, V], TensorType[B, L, T, V]]:

        weight = self._get_weight(query, key)                                   # [batch, target, source]
        if value.dim() == 4:
            weight = weight[:, None, :, :]                                      # [batch, 1, target, source]
        value = torch.matmul(weight, value)                                     # [batch, target, v_dim]
        return value                                                            # or [batch, latent, target, v_dim]
