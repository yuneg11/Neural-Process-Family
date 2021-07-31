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
    "SetConv2dEncoder",
    "SetConv2dDecoder",
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
        query: TensorType[...],
        key:   TensorType[...],
        value: TensorType[...],
    ) -> TensorType[...]:
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
            distance = (a - b.transpose(1, 2)) ** 2
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
    ) -> TensorType[B, V + 1, T]:

        density = torch.ones((*value.shape[:2], 1), device=value.device)        # [batch, source, 1]
        value = torch.cat((density, value), dim=2)                              # [batch, source, v_dim + 1]
        value = value.transpose(2, 1)                                           # [batch, v_dim + 1, source]

        weight = self._get_weight(key, query)                                   # [batch, source, target]
        value = torch.matmul(value, weight)                                     # [batch, v_dim + 1, target]
        value = torch.cat((                                                     # [batch, v_dim + 1, target]
            value[:, :1, ...],
            value[:, 1:, ...] / (value[:, :1, ...] + 1e-8)
        ), axis=1)

        return value


class SetConv1dDecoder(SetConvBase):
    def __init__(self,
        init_log_scale: float,
        dim_last: bool = True,
    ):
        super().__init__(init_log_scale=init_log_scale)
        self.dim_last = dim_last

    def forward(self,
        query: TensorType[B, T, QK],
        key:   TensorType[B, S, QK],
        value: Union[TensorType[B, V, S], TensorType[B, L, V, S]],
    ) -> Union[
        TensorType[B, V, T], TensorType[B, L, V, T],
        TensorType[B, T, V], TensorType[B, L, T, V],
    ]:

        weight = self._get_weight(key, query)                                   # [batch, source, target]
        if value.dim() == 4:
            weight = weight[:, None, :, :]                                      # [batch, 1, source, target]
        value = torch.matmul(value, weight)                                     # [batch, (latent,) v_dim, target]
        if self.dim_last:
            value = value.transpose(-2, -1)                                     # [batch, (latent,) target, v_dim]
        return value


class SetConv2dEncoder(SetConvBase):
    def forward(self,
        query: TensorType[B, T, QK],
        key:   TensorType[B, S, QK],
        value: TensorType[B, S, V],
    ) -> TensorType[B, V + 2, T, T]:

        density = torch.ones((*value.shape[:-1], 1), device=value.device)       # [batch, source, 1]
        value = torch.cat((density, value), dim=-1)                             # [batch, source, v_dim + 1]
        value = value.transpose(-2, -1)[..., None]                              # [batch, v_dim + 1, source, 1]

        weight = self._get_weight(key, query)                                   # [batch, source, target]
        weight = weight[:, None, :, :]                                          # [batch, 1, source, target]

        value = (weight * value).transpose(-2, -1)                              # [batch, v_dim + 1, target, source]
        value = torch.matmul(value, weight)                                     # [batch, v_dim + 1, target, target]
        value = torch.cat((                                                     # [batch, v_dim + 1, target, target]
            value[:, :1, ...],
            value[:, 1:, ...] / (value[:, :1, ...] + 1e-8)
        ), axis=1)

        identity = torch.eye(query.shape[1], device=value.device)[None, None, ...]# [1, 1, target, target]
        identity = identity.repeat(value.shape[0], 1, 1, 1)                     # [batch, 1, target, target]
        value = torch.cat((identity, value), dim=1)                             # [batch, v_dim + 2, target, target]

        return value


class SetConv2dDecoder(SetConvBase):
    def forward(self,
        query: TensorType[B, T, QK],
        key:   TensorType[B, S, QK],
        value: TensorType[B, V, S, S],
    ) -> TensorType[B, V, T, T]:

        weight = self._get_weight(query, key)                                   # [batch, target, source]
        weight = weight[:, None, :, :]                                          # [batch, 1, target, source]

        value = torch.matmul(weight, value)                                     # [batch, v_dim, target, source]
        value_t = value.transpose(-2, -1)                                       # [batch, v_dim, source, target]
        value = torch.matmul(value, value_t)                                    # [batch, v_dim, target, target]
        return value
