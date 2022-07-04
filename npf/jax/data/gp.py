# Source code modified from https://github.com/juho-lee/bnp
#
# See the original PyTorch implementation below.
# https://github.com/juho-lee/bnp/blob/master/regression/data/gp.py


import math
from functools import partial
from collections import deque

import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import numpy as np

from npf.jax import functional as F

from .base import Dataset, IterableDataset, NPData


__all__ = [
    "sample_gp_for_plot",
    "RBFKernel",
    "Matern52Kernel",
    "PeriodicKernel",
    "GPDatasetBase",
    "GPDataset",
    "GPPriorDataset",
    "GPIterableDataset",
]


def sample_gp_for_plot(key, kernel, batch_size=16, num_ctx=None, max_num_points=50, x_range=(-2, 2), t_noise=None):
    keys = random.split(key, 6)
    shape = (batch_size, max_num_points, 1)

    if num_ctx is None:
        num_ctx = random.randint(keys[0], shape=(batch_size,), minval=3, maxval=max_num_points - 2)
    else:
        num_ctx = np.full(shape=(batch_size,), fill_value=num_ctx)

    mask     = np.ones((batch_size, max_num_points), dtype=np.bool_)
    mask_tar = np.ones((batch_size, max_num_points), dtype=np.bool_)
    mask_ctx = jax.vmap(lambda _c:  F.get_mask(max_num_points, start=0, stop=_c))(num_ctx)
    mask_ctx = random.permutation(keys[1], mask_ctx, axis=1, independent=True)

    x = x_range[0] + (x_range[1] - x_range[0]) * random.uniform(keys[2], shape=shape)
    x = np.expand_dims(
        np.linspace(x_range[0], x_range[1], max_num_points, endpoint=True), axis=(0, 2),
    ).repeat(batch_size, axis=0)

    mean = np.zeros(shape[:-1])
    cov = kernel(keys[3], x)

    y = random.multivariate_normal(keys[4], mean, cov).reshape(shape)

    if t_noise is not None:
        if t_noise == -1:
            t_noise = 0.15 * random.uniform(keys[5], shape=y.shape)
        else:
            t_noise = t_noise
        y += t_noise * random.t(keys[6], df=2.1, shape=y.shape)

    batch = NPData(
        x        = x,
        y        = y,
        mask     = mask,
        x_ctx    = F.masked_fill(x, mask_ctx, mask_axis=(0, 1), fill_value=0),
        y_ctx    = F.masked_fill(y, mask_ctx, mask_axis=(0, 1), fill_value=0),
        mask_ctx = mask_ctx,
        x_tar    = x,
        y_tar    = y,
        mask_tar = mask_tar,
    )

    return batch


class RBFKernel:
    def __init__(self, sigma_eps: float = 2e-2, max_length: float = 0.6, max_scale: float = 1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, key, x):
        subkey_0, subkey_1 = random.split(key)
        length = 0.1 + (self.max_length - 0.1) * random.uniform(subkey_0, shape=(x.shape[0], 1, 1, 1))
        scale  = 0.1 + (self.max_scale  - 0.1) * random.uniform(subkey_1, shape=(x.shape[0], 1, 1))

        dist = (jnp.expand_dims(x, axis=-2) - jnp.expand_dims(x, axis=-3)) / length
        cov = jnp.power(scale, 2) * jnp.exp(-0.5 * jnp.power(dist, 2).sum(axis=-1)) + self.sigma_eps ** 2 * jnp.eye(x.shape[-2])
        return cov


class Matern52Kernel:
    def __init__(self, sigma_eps: float = 2e-2, max_length: float = 0.6, max_scale: float = 1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, key, x):
        subkey_0, subkey_1 = random.split(key)
        length = 0.1 + (self.max_length - 0.1) * random.uniform(subkey_0, shape=(x.shape[0], 1, 1, 1))
        scale  = 0.1 + (self.max_scale  - 0.1) * random.uniform(subkey_1, shape=(x.shape[0], 1, 1))

        dist = jnp.linalg.norm((jnp.expand_dims(x, axis=-2) - jnp.expand_dims(x, axis=-3)) / length, axis=-1)
        cov = jnp.power(scale, 2) * (1 + math.sqrt(5.0) * dist + 5.0 / 3.0 * jnp.power(dist, 2)) * jnp.exp(-math.sqrt(5.0) * dist) + self.sigma_eps ** 2 * jnp.eye(x.shape[-2])

        return cov


class PeriodicKernel:
    def __init__(self, sigma_eps: float = 2e-2, max_length: float = 0.6, max_scale: float = 1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, key, x):
        subkey_0, subkey_1, subkey_2 = random.split(key, 3)

        length = 0.1 + (self.max_length - 0.1) * random.uniform(subkey_0, shape=(x.shape[0], 1, 1))
        scale  = 0.1 + (self.max_scale  - 0.1) * random.uniform(subkey_1, shape=(x.shape[0], 1, 1))

        p = 0.1 + 0.4 * random.uniform(subkey_2, shape=(x.shape[0], 1, 1))
        dist = jnp.expand_dims(x, axis=-2) - jnp.expand_dims(x, axis=-3)
        cov = jnp.power(scale, 2) * jnp.exp(-2 * jnp.power((jnp.sin(math.pi * jnp.abs(dist).sum(axis=-1) / p) / length), 2)) + self.sigma_eps ** 2 * jnp.eye(x.shape[-2])

        return cov


class GPDatasetBase(Dataset):
    chunk_size = 256
    key_queue_size = 100
    randomize_x = True

    def __init__(
        self,
        key,
        kernel,
        num_ctx = None,
        num_tar = None,
        max_num_points = 50,
        x_range = (-2, 2),
        t_noise = None,
    ):
        self.key = key
        self.kernel = kernel
        self.num_ctx = num_ctx
        self.num_tar = num_tar
        self.max_num_points = max_num_points
        self.x_range = x_range
        self.t_noise = t_noise

    @partial(jax.jit, static_argnames=("self",))
    def build_chunk(self, key):
        keys = random.split(key, 6)
        shape = (self.chunk_size, self.max_num_points, 1)

        if self.num_ctx is None:
            num_ctx = random.randint(keys[0], shape=(self.chunk_size,), minval=3, maxval=self.max_num_points - 2)
        else:
            num_ctx = np.full(shape=(self.chunk_size,), fill_value=self.num_ctx)

        if self.num_tar is None:
            num_tar = random.randint(keys[1], shape=(self.chunk_size,), minval=3, maxval=self.max_num_points - num_ctx + 1)
        else:
            num_tar = np.full(shape=(self.chunk_size,), fill_value=self.num_tar)

        num_points = num_ctx + num_tar

        mask     = jax.vmap(lambda _p:     F.get_mask(self.max_num_points, start=0,  stop=_p))(num_points)
        mask_ctx = jax.vmap(lambda _c:     F.get_mask(self.max_num_points, start=0,  stop=_c))(num_ctx)
        mask_tar = jax.vmap(lambda _c, _p: F.get_mask(self.max_num_points, start=_c, stop=_p))(num_ctx, num_points)

        if self.randomize_x:
            x = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * random.uniform(keys[2], shape=shape)
        else:
            x0 = np.linspace(self.x_range[0], self.x_range[1], num=self.max_num_points, endpoint=True)
            x = np.expand_dims(x0, axis=(0, -1)).repeat(self.chunk_size, axis=0)

        mean = jnp.zeros(shape[:-1])
        cov = self.kernel(keys[3], x)

        y = random.multivariate_normal(keys[4], mean, cov).reshape(shape)

        if self.t_noise is not None:
            if self.t_noise == -1:
                t_noise = 0.15 * random.uniform(keys[5], shape=y.shape)
            else:
                t_noise = self.t_noise
            y += t_noise * random.t(keys[6], df=2.1, shape=y.shape)

        chunk = NPData(
            x        = F.masked_fill(x, mask,     mask_axis=(0, 1), fill_value=0),
            y        = F.masked_fill(y, mask,     mask_axis=(0, 1), fill_value=0),
            mask     = mask,
            x_ctx    = F.masked_fill(x, mask_ctx, mask_axis=(0, 1), fill_value=0),
            y_ctx    = F.masked_fill(y, mask_ctx, mask_axis=(0, 1), fill_value=0),
            mask_ctx = mask_ctx,
            x_tar    = F.masked_fill(x, mask_tar, mask_axis=(0, 1), fill_value=0),
            y_tar    = F.masked_fill(y, mask_tar, mask_axis=(0, 1), fill_value=0),
            mask_tar = mask_tar,
        )

        return chunk


class GPDataset(GPDatasetBase):
    def __init__(
        self,
        key,
        kernel,
        data_size,
        num_ctx = None,
        num_tar = None,
        max_num_points = 50,
        x_range = (-2, 2),
        t_noise = None,
    ):
        super().__init__(
            key, kernel, num_ctx=num_ctx, num_tar=num_tar, max_num_points=max_num_points,
            x_range=x_range, t_noise=t_noise,
        )

        self.data_size = data_size
        self.keys = []

        num_chunks = math.ceil(self.data_size / self.chunk_size)
        keys_queue = deque()

        for _ in range(num_chunks):
            if len(keys_queue) == 0:
                key, *keys = random.split(key, self.key_queue_size + 1)
                keys_queue.extend(keys)
            self.keys.append(keys_queue.popleft())

        chunks = [self.build_chunk(key) for key in self.keys]

        self.num_elem = len(chunks[0])
        self._cache = NPData(*[
            np.concatenate([np.asarray(c[i]) for c in chunks], axis=0)
            for i in range(self.num_elem)
        ])

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return NPData(*[self._cache[i][index] for i in range(self.num_elem)])


class GPPriorDataset(GPDataset):
    randomize_x = False

    def __init__(
        self,
        key,
        kernel,
        data_size,
        num_ctx = 1,
        num_points = 50,
        x_range = (-2, 2),
        t_noise = None,
    ):
        super().__init__(
            key, kernel, data_size=data_size, num_ctx=num_ctx, num_tar=(num_points-num_ctx),
            x_range=x_range, t_noise=t_noise,
        )

        self._cache = NPData(
            *self._cache[:5],
            random.permutation(key, self._cache[5], axis=1, independent=True),
            *self._cache[6:],
        )

    def __iter__(self):
        return (
            NPData(*[self._cache[i][j] for i in range(self.num_elem)])
            for j in range(self.data_size)
        )


class GPIterableDataset(GPDatasetBase, IterableDataset):
    def __init__(
        self,
        key,
        kernel,
        batch_size,
        num_ctx = None,
        num_tar = None,
        max_num_points = 50,
        x_range = (-2, 2),
        t_noise = None,
    ):
        super().__init__(
            key, kernel, num_ctx=num_ctx, num_tar=num_tar, max_num_points=max_num_points,
            x_range=x_range, t_noise=t_noise,
        )

        self.batch_size = batch_size
        self.key = key
        self._key = key
        self._keys_queue = None
        self.chunk_size = batch_size

    def __iter__(self):
        self._keys_queue = deque()
        return self

    def __next__(self):
        if self._keys_queue is None:
            raise RuntimeError("Iterator is not initialized.")

        if len(self._keys_queue) == 0:
            self._key, *_keys = random.split(self._key, self.key_queue_size + 1)
            self._keys_queue.extend(_keys)

        key = self._keys_queue.popleft()
        chunk = self.build_chunk(key)
        return chunk
