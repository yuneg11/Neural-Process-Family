import math
from logging import warning
from functools import partial
from collections import namedtuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from jax.random import multivariate_normal, t, uniform

from . import functional as F


__all__ = [
    "GPPriorSampler",
    "GPSampler",
    "RBFKernel",
    "PeriodicKernel",
    "Matern52Kernel",
]


NPData = namedtuple("NPData", (
    "x",
    "y",
    "x_ctx",
    "x_tar",
    "y_ctx",
    "y_tar",
    "mask",
    "mask_ctx",
    "mask_tar",
))


class GPPriorSampler:
    def __init__(self, key, kernel, t_noise = None):
        warning.warn("'npf.jax._data' is deprecated and will be removed in a future version.")

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
        warning.warn("'npf.jax._data' is deprecated and will be removed in a future version.")

        self.kernel = kernel
        self.t_noise = t_noise

    @partial(jax.jit, static_argnames=("self", "batch_size", "num_ctx", "num_tar", "max_num_points", "x_range"))
    def sample(self, key, batch_size=16, num_ctx=None, num_tar=None, max_num_points=50, x_range=(-2, 2)):
        keys = random.split(key, 6)
        shape = (batch_size, max_num_points, 1)

        if num_ctx is None:
            num_ctx = random.randint(keys[0], shape=(batch_size,), minval=3, maxval=max_num_points - 2)
        else:
            num_ctx = np.full(shape=(batch_size,), fill_value=num_ctx)

        if num_tar is None:
            num_tar = random.randint(keys[1], shape=(batch_size,), minval=3, maxval=max_num_points - num_ctx + 1)
        else:
            num_tar = np.full(shape=(batch_size,), fill_value=num_tar)

        num_points = num_ctx + num_tar

        mask     = jax.vmap(lambda _p:     F.get_mask(max_num_points, start=0,  stop=_p))(num_points)
        mask_ctx = jax.vmap(lambda _c:     F.get_mask(max_num_points, start=0,  stop=_c))(num_ctx)
        mask_tar = jax.vmap(lambda _c, _p: F.get_mask(max_num_points, start=_c, stop=_p))(num_ctx, num_points)

        x = x_range[0] + (x_range[1] - x_range[0]) * uniform(keys[2], shape=shape)

        mean = jnp.zeros(shape[:-1])
        cov = self.kernel(keys[3], x)

        y = random.multivariate_normal(keys[4], mean, cov).reshape(shape)

        if self.t_noise is not None:
            if self.t_noise == -1:
                t_noise = 0.15 * random.uniform(keys[5], shape=y.shape)
            else:
                t_noise = self.t_noise
            y += t_noise * random.t(keys[6], shape=y.shape)

        batch = NPData(
            x     = F.masked_fill(x, mask,     fill_value=0., non_mask_axis=-1),
            y     = F.masked_fill(y, mask,     fill_value=0., non_mask_axis=-1),
            x_ctx = F.masked_fill(x, mask_ctx, fill_value=0., non_mask_axis=-1),
            y_ctx = F.masked_fill(y, mask_ctx, fill_value=0., non_mask_axis=-1),
            x_tar = F.masked_fill(x, mask_tar, fill_value=0., non_mask_axis=-1),
            y_tar = F.masked_fill(y, mask_tar, fill_value=0., non_mask_axis=-1),
            mask     = mask,
            mask_ctx = mask_ctx,
            mask_tar = mask_tar,
        )

        return batch

    # @partial(jax.jit, static_argnames=("self", "batch_size", "num_ctx", "num_tar", "max_num_points", "x_range"))
    def sample_for_plot(self, key, batch_size=16, num_ctx=None, num_points=50, x_range=(-2, 2)):
        keys = random.split(key, 6)
        shape = (batch_size, num_points, 1)

        if num_ctx is None:
            num_ctx = random.randint(keys[0], shape=(batch_size,), minval=3, maxval=50 - 2)
        else:
            num_ctx = jnp.full(shape=(batch_size,), fill_value=num_ctx)

        mask     = jnp.ones((batch_size, num_points), dtype=jnp.bool_)
        mask_ctx = jax.vmap(lambda _c:  F.get_mask(num_points, start=0, stop=_c))(num_ctx)
        mask_ctx = random.permutation(keys[1], mask_ctx, axis=1, independent=True)
        mask_tar = jnp.ones((batch_size, num_points), dtype=jnp.bool_)

        x = jnp.expand_dims(
            jnp.linspace(x_range[0], x_range[1], num_points, endpoint=True), axis=(0, 2),
        ).repeat(batch_size, axis=0)

        mean = jnp.zeros(shape[:-1])
        cov = self.kernel(keys[3], x)

        y = random.multivariate_normal(keys[4], mean, cov).reshape(shape)

        if self.t_noise is not None:
            if self.t_noise == -1:
                t_noise = 0.15 * random.uniform(keys[5], shape=y.shape)
            else:
                t_noise = self.t_noise
            y += t_noise * random.t(keys[6], shape=y.shape)

        batch = NPData(
            x     = x,
            y     = y,
            x_ctx = F.masked_fill(x, mask_ctx, fill_value=0., non_mask_axis=-1),
            y_ctx = F.masked_fill(y, mask_ctx, fill_value=0., non_mask_axis=-1),
            x_tar = x,
            y_tar = y,
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
