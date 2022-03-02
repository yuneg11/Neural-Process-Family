import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random
from jax.random import multivariate_normal, t, randint, uniform

from . import functional as F


__all__ = [
    "GPPriorSampler",
    "GPSampler",
    "RBFKernel",
    "PeriodicKernel",
    "Matern52Kernel",
]


@dataclass
class GPData:
    x: jnp.ndarray
    y: jnp.ndarray
    x_ctx: jnp.ndarray
    x_tar: jnp.ndarray
    y_ctx: jnp.ndarray
    y_tar: jnp.ndarray
    mask:     jnp.ndarray
    mask_ctx: jnp.ndarray
    mask_tar: jnp.ndarray


class GPPriorSampler:
    def __init__(self, key, kernel, t_noise = None):
        self.kernel = kernel
        self.t_noise = t_noise
        self.key = key

    def sample(self, x):
        key, subkey = jax.random.split(self.key, 2)
        cov = self.kernel(subkey, x)
        mean = jnp.zeros((1,x.shape[1]))

        key, subkey = jax.random.split(key,2)

        y = multivariate_normal(subkey, mean, cov)
        y = jnp.expand_dims(y, axis = -1)

        if self.t_noise is not None:
            key, subkey = jax.random.split(key, 2)
            y += self.t_noise * t(subkey, shape = y.shape)

        return y


class GPSampler:
    def __init__(self, kernel, t_noise=None):
        self.kernel = kernel
        self.t_noise = t_noise

    def sample(self, key, batch_size=16, num_ctx=None, num_tar=None, max_num_points=50, x_range=(-2, 2)):
        keys = random.split(key, 6)
        shape = (batch_size, max_num_points, 1)

        num_ctx = int(num_ctx or (randint(keys[0], shape=[1], minval=3, maxval=max_num_points - 3))[0])
        num_tar = int(num_tar or (randint(keys[1], shape=[1], minval=3, maxval=max_num_points - num_ctx))[0])
        num_points = num_ctx + num_tar

        mask     = F.get_mask(max_num_points, start=0,       stop=num_points)
        mask_ctx = F.get_mask(max_num_points, start=0,       stop=num_ctx)
        mask_tar = F.get_mask(max_num_points, start=num_ctx, stop=num_points)

        x = x_range[0] + (x_range[1] - x_range[0]) * uniform(keys[2], shape=shape)

        mean = jnp.zeros(shape).squeeze(axis=2)
        cov = self.kernel(keys[3], x)

        y = multivariate_normal(keys[4], mean, cov).reshape(shape)

        if self.t_noise is not None:
            if self.t_noise == -1:
                t_noise = 0.15 * uniform(keys[5], shape=y.shape)
            else:
                t_noise = self.t_noise
            y += t_noise * t(keys[6], shape=y.shape)

        batch = GPData(
            x     = F.apply_mask(x, mask,     fill_value=0, axis=-2),
            y     = F.apply_mask(y, mask,     fill_value=0, axis=-2),
            x_ctx = F.apply_mask(x, mask_ctx, fill_value=0, axis=-2),
            y_ctx = F.apply_mask(y, mask_ctx, fill_value=0, axis=-2),
            x_tar = F.apply_mask(x, mask_tar, fill_value=0, axis=-2),
            y_tar = F.apply_mask(y, mask_tar, fill_value=0, axis=-2),
            mask     = mask,
            mask_ctx = mask_ctx,
            mask_tar = mask_tar,
        )
        return batch


class RBFKernel:
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, key, x):
        subkey_1, subkey_2 = random.split(key)
        length = 0.1 + (self.max_length - 0.1) * uniform(subkey_1, shape=(x.shape[0], 1, 1, 1))
        scale  = 0.1 + (self.max_scale  - 0.1) * uniform(subkey_2, shape=(x.shape[0], 1, 1))

        dist = (jnp.expand_dims(x, axis=-2) - jnp.expand_dims(x, axis=-3)) / length
        cov = jnp.power(scale, 2) * jnp.exp(-0.5 * jnp.power(dist, 2).sum(axis=-1)) + self.sigma_eps ** 2 * jnp.eye(x.shape[-2])

        return cov


class Matern52Kernel:
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, key, x):
        subkey_1, subkey_2 = random.split(key)
        length = 0.1 + (self.max_length - 0.1) * uniform(subkey_1, shape=(x.shape[0], 1, 1, 1))
        scale  = 0.1 + (self.max_scale  - 0.1) * uniform(subkey_2, shape=(x.shape[0], 1, 1))

        dist = jnp.linalg.norm((jnp.expand_dims(x, axis=-2) - jnp.expand_dims(x, axis=-3)) / length, axis=-1)
        cov = jnp.power(scale, 2) * (1 + math.sqrt(5.0) * dist + 5.0 / 3.0 * jnp.power(dist, 2)) * jnp.exp(-math.sqrt(5.0) * dist) + self.sigma_eps ** 2 * jnp.eye(x.shape[-2])

        return cov


class PeriodicKernel:
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, key, x):
        subkey_0, subkey_1, subkey_2 = random.split(key, 3)

        p = 0.1 + 0.4 * uniform(subkey_0, shape=(x.shape[0], 1, 1))
        length = 0.1 + (self.max_length - 0.1) * uniform(subkey_1, shape=(x.shape[0], 1, 1))
        scale  = 0.1 + (self.max_scale  - 0.1) * uniform(subkey_2, shape=(x.shape[0], 1, 1))

        dist = jnp.expand_dims(x, axis=-2) - jnp.expand_dims(x, axis=-3)
        cov = jnp.power(scale, 2) * jnp.exp(-2 * jnp.power((jnp.sin(math.pi * jnp.abs(dist).sum(axis=-1) / p) / length), 2)) + self.sigma_eps ** 2 * jnp.eye(x.shape[-2])

        return cov
