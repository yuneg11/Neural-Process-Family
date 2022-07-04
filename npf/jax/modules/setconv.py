from ..typing import *

from abc import abstractmethod

import numpy as np

from jax import numpy as jnp
from flax import linen as nn

from .. import functional as F

__all__ = [
    "Discretization1d",
    "SetConv1dEncoder",
    "SetConv1dDecoder",
    "SetConv2dEncoder",
    "SetConv2dDecoder",
]

class Discretization1d(nn.Module):
    minval: float
    maxval: float
    points_per_unit: int
    multiple: int
    margin: float = 0.1

    @nn.compact
    def __call__(self,
        x_ctx:    Array[B, C, X],
        x_tar:    Array[B, T, X],
        mask_ctx: Array[B, C],
        mask_tar: Array[B, T],
    ) -> Tuple[Array[1, D, 1], Array[D]]:

        x_ctx_min = F.masked_min(x_ctx, mask_ctx, axis=(-2, -1), non_mask_axis=-1)                  # [batch]
        x_ctx_max = F.masked_max(x_ctx, mask_ctx, axis=(-2, -1), non_mask_axis=-1)                  # [batch]
        x_tar_min = F.masked_min(x_tar, mask_tar, axis=(-2, -1), non_mask_axis=-1)                  # [batch]
        x_tar_max = F.masked_max(x_tar, mask_tar, axis=(-2, -1), non_mask_axis=-1)                  # [batch]

        x_min = jnp.min(jnp.stack((x_ctx_min, x_tar_min), axis=-1), axis=-1)                        # [batch]
        x_max = jnp.max(jnp.stack((x_ctx_max, x_tar_max), axis=-1), axis=-1)                        # [batch]

        data_min = jnp.expand_dims(x_min - (self.margin + (1 / self.points_per_unit)), axis=-1)     # [batch]
        data_max = jnp.expand_dims(x_max + (self.margin + (1 / self.points_per_unit)), axis=-1)     # [batch]

        grid_min = self.minval - (self.margin + (1 / self.points_per_unit))                         # [1]
        grid_max = self.maxval + (self.margin + (1 / self.points_per_unit))                         # [1]

        raw_points = self.points_per_unit * (grid_max - grid_min)                                   # [1]
        num_points = np.where(                                                                      # [1]
            np.isclose(raw_points % self.multiple, 0),
            np.around(raw_points),
            np.around(raw_points + self.multiple - raw_points % self.multiple),
        ).astype(int)

        x_grid = np.linspace(grid_min, grid_max, num_points, endpoint=True)                         # [discrete]
        x_grid = np.repeat(np.expand_dims(x_grid, axis=0), x_ctx.shape[0], axis=0)                  # [batch, discrete]
        mask_grid = (data_min <= x_grid) & (x_grid <= data_max)                                     # [batch, discrete]

        x_grid = np.expand_dims(x_grid, axis=-1)                                                    # [batch, discrete, 1]

        return x_grid, mask_grid

class SetConvBase(nn.Module):
    init_log_scale: float = 1.0

    @abstractmethod
    def __call__(self,
        query: Array[B, T, QK],
        key:   Array[B, S, QK],
        value: Array[...],
        mask:  Array[B, S],
    ) -> Array[...]:
        """
        Args:
            query: location to interpolate
            key: location of value
            value: value to be interpolated
            mask: mask of k and v

        Returns:
            interpolated_value
        """
        raise NotImplementedError

    @staticmethod
    def _get_distance(
        a: Array[B, N, X],
        b: Array[B, M, X],
    ) -> Array[B, N, M]:

        if a.shape[-1] == 1 and b.shape[-1] == 1:
            distance = jnp.square(a - jnp.swapaxes(b, -1, -2))
        else:
            a_norm = jnp.expand_dims(jnp.sum(jnp.square(a), axis=-1), axis=-1)                      # [batch, n, 1]
            b_norm = jnp.expand_dims(jnp.sum(jnp.square(b), axis=-1), axis=-2)                      # [batch, 1, m]
            distance = a_norm + b_norm - 2 * jnp.matmul(a, jnp.swapaxes(b, -1, -2))                 # [batch, n, m]
        return distance

    def _get_weight(self,
        a: Array[B, N, X],
        b: Array[B, M, X],
    ) -> Array[B, N, M]:

        log_scale = self.param("log_scale", lambda key: self.init_log_scale)
        distance = self._get_distance(a, b)                                                         # [batch, n, m]
        weight = jnp.exp(-0.5 * distance / jnp.exp(2 * log_scale))                                  # [batch, n, m]
        return weight

class SetConv1dEncoder(SetConvBase):
    @nn.compact
    def __call__(self,
        query: Array[B, T, QK],
        key:   Array[B, S, QK],
        value: Array[B, S, V],
        mask:  Array[B, S],
    ) -> Array[B, T, V + 1]:

        weight = self._get_weight(query, key)                                                       # [batch, target, source]
        weight = F.masked_fill(weight, mask, non_mask_axis=-2)                                      # [batch, target, source]

        density = np.ones((*value.shape[:-1], 1))                                                   # [batch, source, 1]
        density_value = jnp.concatenate((density, value), axis=-1)                                  # [batch, source, v_dim + 1]
        density_value = jnp.matmul(weight, density_value)                                           # [batch, target, v_dim + 1]
        density, value = jnp.split(density_value, (1,), axis=-1)                                    # [batch, target, 1], [batch, target, v_dim]
        value = jnp.concatenate((density, value / (density + 1e-8)), axis=-1)                       # [batch, target, v_dim + 1]

        return value

class SetConv1dDecoder(SetConvBase):
    @nn.compact
    def __call__(self,
        query: Array[B, T, QK],
        key:   Array[B, S, QK],
        value: Union[Array[B, S, V], Array[B, L, S, V]],
        mask:  Array[B, S],
    ) -> Union[Array[B, T, V], Array[B, L, T, V]]:

        weight = self._get_weight(query, key)                                                       # [batch, target, source]
        weight = F.masked_fill(weight, mask, non_mask_axis=-2)                                      # [batch, target, source]

        if value.ndim == 4 and query.ndim == 3:
            weight = jnp.expand_dims(weight, axis=-3)                                               # [batch, 1, target, source]
        value = jnp.matmul(weight, value)                                                           # [batch, (latent,) target, v_dim]

        return value

class SetConv2dEncoder(SetConvBase):
    @nn.compact
    def __call__(self,
        query: Array[B, T, QK],
        key:   Array[B, S, QK],
        value: Array[B, S, V],
        mask:  Array[S],
    ) -> Array[B, T, T, V + 2]:

        weight = self._get_weight(key, query)                                                       # [batch, source, target]
        weight = F.masked_fill(weight, mask, non_mask_axis=-1)                                      # [batch, source, target]
        weight = jnp.expand_dims(weight, axis=-3)                                                   # [batch, 1, source, target]

        density = np.ones((*value.shape[:-1], 1))                                                   # [batch, source, 1]
        density_value = jnp.concatenate((density, value), axis=-1)                                  # [batch, source, v_dim + 1]
        density_value = jnp.expand_dims(jnp.swapaxes(density_value, -2, -1), axis=-1)               # [batch, v_dim + 1, source, 1]

        density_value = jnp.swapaxes(weight * density_value, -2, -1)                                # [batch, v_dim + 1, target, source]
        density_value = jnp.matmul(density_value, weight)                                           # [batch, v_dim + 1, target, target]
        density_value = jnp.transpose(density_value, axes=(0, 2, 3, 1))                             # [batch, target, target, v_dim + 1]
        density, value = jnp.split(density_value, (1,), axis=-1)                                    # [batch, target, target, 1], [batch, target, target, v_dim]
        value = jnp.concatenate((density, value / (density + 1e-8)), axis=-1)                       # [batch, target, target, v_dim + 1]

        identity = np.expand_dims(np.eye(query.shape[-2]), axis=(-4, -1))                           # [1, target, target, 1]
        identity = np.repeat(identity, value.shape[-4], axis=-4)                                    # [batch, target, target, 1]
        value = jnp.concatenate((identity, value), axis=-1)                                         # [batch, target, target, v_dim + 2]

        return value

class SetConv2dDecoder(SetConvBase):
    @nn.compact
    def __call__(self,
        query: Array[B, T, QK],
        key:   Array[B, S, QK],
        value: Array[B, S, S, V],
        mask:  Array[S],
    ) -> Array[B, T, T, V]:

        weight = self._get_weight(query, key)                                                       # [batch, target, source]
        weight = F.masked_fill(weight, mask, non_mask_axis=-1)                                      # [batch, source, target]
        weight = jnp.expand_dims(weight, axis=-3)                                                   # [batch, 1, target, source]

        # TODO: Optimize this
        value = jnp.transpose(value, axes=(0, 3, 1, 2))                                             # [batch, v_dim, source, source]
        value = jnp.matmul(weight, value)                                                           # [batch, v_dim, target, source]
        value_t = jnp.swapaxes(value, -2, -1)                                                       # [batch, v_dim, source, target]
        value = jnp.matmul(value, value_t)                                                          # [batch, v_dim, target, target]
        value = jnp.transpose(value, axes=(0, 2, 3, 1))                                             # [batch, target, target, v_dim]

        return value
