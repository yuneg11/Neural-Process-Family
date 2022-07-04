import sys
sys.path.append(".")

import os
import math
from functools import partial
from collections import deque

import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import numpy as np

from npf.jax import functional as F

from .data import Dataset, IterableDataset, DataLoader

def sample_gp_for_plot(self, key, batch_size=16, num_ctx=None, max_num_points=50, x_range=(-2, 2)):
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
    cov = self.kernel(keys[3], x)

    y = random.multivariate_normal(keys[4], mean, cov).reshape(shape)

    if self.t_noise is not None:
        if self.t_noise == -1:
            t_noise = 0.15 * random.uniform(keys[5], shape=y.shape)
        else:
            t_noise = self.t_noise
        y += t_noise * random.t(keys[6], df=2.1, shape=y.shape)

    return (
        x,
        y,
        mask,
        F.masked_fill(x, mask_ctx, mask_axis=(0, 1), fill_value=0),
        F.masked_fill(y, mask_ctx, mask_axis=(0, 1), fill_value=0),
        mask_ctx,
        x,
        y,
        mask_tar,
    )

class RBFKernel:
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, key, x):
        subkey_1, subkey_2 = random.split(key)
        length = 0.1 + (self.max_length - 0.1) * random.uniform(subkey_1, shape=(x.shape[0], 1, 1, 1))
        scale  = 0.1 + (self.max_scale  - 0.1) * random.uniform(subkey_2, shape=(x.shape[0], 1, 1))

        dist = (jnp.expand_dims(x, axis=-2) - jnp.expand_dims(x, axis=-3)) / length
        return jnp.power(scale, 2) * jnp.exp(
            -0.5 * jnp.power(dist, 2).sum(axis=-1)
        ) + self.sigma_eps**2 * jnp.eye(x.shape[-2])

class Matern52Kernel:
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, key, x):
        subkey_1, subkey_2 = random.split(key)
        length = 0.1 + (self.max_length - 0.1) * random.uniform(subkey_1, shape=(x.shape[0], 1, 1, 1))
        scale  = 0.1 + (self.max_scale  - 0.1) * random.uniform(subkey_2, shape=(x.shape[0], 1, 1))

        dist = jnp.linalg.norm((jnp.expand_dims(x, axis=-2) - jnp.expand_dims(x, axis=-3)) / length, axis=-1)
        return jnp.power(scale, 2) * (
            1 + math.sqrt(5.0) * dist + 5.0 / 3.0 * jnp.power(dist, 2)
        ) * jnp.exp(-math.sqrt(5.0) * dist) + self.sigma_eps**2 * jnp.eye(
            x.shape[-2]
        )

class PeriodicKernel:
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, key, x):
        subkey_0, subkey_1, subkey_2 = random.split(key, 3)

        p = 0.1 + 0.4 * random.uniform(subkey_0, shape=(x.shape[0], 1, 1))
        length = 0.1 + (self.max_length - 0.1) * random.uniform(subkey_1, shape=(x.shape[0], 1, 1))
        scale  = 0.1 + (self.max_scale  - 0.1) * random.uniform(subkey_2, shape=(x.shape[0], 1, 1))

        dist = jnp.expand_dims(x, axis=-2) - jnp.expand_dims(x, axis=-3)
        return jnp.power(scale, 2) * jnp.exp(
            -2
            * jnp.power(
                (jnp.sin(math.pi * jnp.abs(dist).sum(axis=-1) / p) / length), 2
            )
        ) + self.sigma_eps**2 * jnp.eye(x.shape[-2])

def build_gp_dataset(config, key):
    config.setdefault("data_size", None)
    config.setdefault("num_ctx", None)
    config.setdefault("num_tar", None)
    config.setdefault("max_num_points", 50)
    config.setdefault("x_range", [-2, 2])
    config.setdefault("t_noise", None)

    kernel_name = config.kernel.name

    if kernel_name.startswith("RBF"):
        kernel_cls = RBFKernel
    elif kernel_name.startswith("Matern"):
        kernel_cls = Matern52Kernel
    elif kernel_name.startswith("Periodic"):
        kernel_cls = PeriodicKernel
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    kernel = kernel_cls(
        sigma_eps=config.kernel.get("sigma_eps", 2e-2),
        max_length=config.kernel.get("max_length", 0.6),
        max_scale=config.kernel.get("max_scale", 1.0),
    )

    return (
        GPIterableDataset(
            key=key,
            kernel=kernel,
            batch_size=config.batch_size,
            num_ctx=config.num_ctx,
            num_tar=config.num_tar,
            max_num_points=config.max_num_points,
            x_range=config.x_range,
            t_noise=config.t_noise,
        )
        if config.data_size is None
        else GPDataset(
            key=key,
            kernel=kernel,
            data_size=config.data_size,
            num_ctx=config.num_ctx,
            num_tar=config.num_tar,
            max_num_points=config.max_num_points,
            x_range=config.x_range,
            t_noise=config.t_noise,
        )
    )

def build_gp_prior_dataset(config, key):
    config.setdefault("data_size", None)
    config.setdefault("num_points", 1000)
    config.setdefault("x_range", [-2, 2])
    config.setdefault("t_noise", None)

    kernel_name = config.kernel.name

    if kernel_name.startswith("RBF"):
        kernel_cls = RBFKernel
    elif kernel_name.startswith("Matern"):
        kernel_cls = Matern52Kernel
    elif kernel_name.startswith("Periodic"):
        kernel_cls = PeriodicKernel
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    kernel = kernel_cls(
        sigma_eps=config.kernel.get("sigma_eps", 2e-2),
        max_length=config.kernel.get("max_length", 0.6),
        max_scale=config.kernel.get("max_scale", 1.0),
    )

    return GPPriorDataset(
        key=key,
        kernel=kernel,
        data_size=config.data_size,
        num_points=config.num_points,
        x_range=config.x_range,
        t_noise=config.t_noise,
    )

def build_image_dataset(config, key):
    pass

def build_sim2real_dataset(config, key):
    return Sim2RealDataset(
        root=config.root,
        name=config.name,
        split=config.split,
    )

def build_dataloader(config, collate_fn, key):
    dataloader_key, dataset_key = random.split(key)

    if config.type == "GP":
        dataset = build_gp_dataset(config.gp, dataset_key)
    elif config.type == "Image":
        dataset = build_image_dataset(config.image, dataset_key)
    elif config.type == "Sim2Real":
        dataset = build_sim2real_dataset(config.sim2real, dataset_key)
    else:
        raise ValueError(f"Unknown dataset type: {config.type}")

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=collate_fn,
        drop_last=config.drop_last,
        key=dataloader_key,
    )

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

        return (
            F.masked_fill(x, mask, mask_axis=(0, 1), fill_value=0),
            F.masked_fill(y, mask, mask_axis=(0, 1), fill_value=0),
            mask,
            F.masked_fill(x, mask_ctx, mask_axis=(0, 1), fill_value=0),
            F.masked_fill(y, mask_ctx, mask_axis=(0, 1), fill_value=0),
            mask_ctx,
            F.masked_fill(x, mask_tar, mask_axis=(0, 1), fill_value=0),
            F.masked_fill(y, mask_tar, mask_axis=(0, 1), fill_value=0),
            mask_tar,
        )

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
        self._cache = [
            np.concatenate([np.asarray(c[i]) for c in chunks], axis=0)
            for i in range(self.num_elem)
        ]

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return [self._cache[i][index] for i in range(self.num_elem)]

    # @cache
    # def _get_chunk(self, chunk_idx):
    #     return self.build_chunk(self._keys[chunk_idx])

    # @cache
    # def __getitem__(self, idx):
    #     chunk = self._get_chunk(int(idx // self.chunk_size))
    #     local_idx = idx % self.chunk_size
    #     return [v[local_idx] for v in chunk]

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

        self._cache[5] = random.permutation(key, self._cache[5], axis=1, independent=True)

    def __iter__(self):
        return (
            [self._cache[i][j] for i in range(self.num_elem)]
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
        return self.build_chunk(key)

SIM2REAL_DATASETS = {
    "lotkavolterra": "lotka_volterra",
}

class Sim2RealDataset(Dataset):
    def __init__(self, root, name, split):
        self.root = root
        self.name = name
        self.split = split

        if name.lower() not in SIM2REAL_DATASETS:
            raise ValueError(f"Unknown dataset name: {name}")

        self.base_path = os.path.join(root, SIM2REAL_DATASETS[name.lower()], split)

        self.x = jnp.load(os.path.join(self.base_path, "x.npy"))
        self.y = jnp.load(os.path.join(self.base_path, "y.npy"))
        self.mask_ctx = jnp.load(os.path.join(self.base_path, "mask_ctx.npy"))
        self.mask_tar = jnp.load(os.path.join(self.base_path, "mask_tar.npy"))
        self.mask = self.mask_ctx | self.mask_tar

        self.data_size = self.x.shape[0]

    def __len__(self):
        return self.data_size

    @partial(jax.jit, static_argnames=("self",))
    def __getitem__(self, index):
        _x = self.x[index]
        _y = self.y[index]

        mask     = self.mask[index]
        x        = F.masked_fill(_x, mask,     mask_axis=(0, 1), fill_value=0)
        y        = F.masked_fill(_y, mask,     mask_axis=(0, 1), fill_value=0)
        mask_ctx = self.mask_ctx[index]
        x_ctx    = F.masked_fill(_x, mask_ctx, mask_axis=(0, 1), fill_value=0)
        y_ctx    = F.masked_fill(_y, mask_ctx, mask_axis=(0, 1), fill_value=0)
        mask_tar = self.mask_tar[index]
        x_tar    = F.masked_fill(_x, mask_tar, mask_axis=(0, 1), fill_value=0)
        y_tar    = F.masked_fill(_y, mask_tar, mask_axis=(0, 1), fill_value=0)

        x_mu    = F.masked_mean(x_ctx, mask_ctx, axis=-2, mask_axis=(0, 1), keepdims=True)
        x_sigma = F.masked_std(x_ctx, mask_ctx,  axis=-2, mask_axis=(0, 1), keepdims=True)
        x_sigma = jnp.where(x_sigma == 0, 1, x_sigma)
        y_mu    = F.masked_mean(y_ctx, mask_ctx, axis=-2, mask_axis=(0, 1), keepdims=True)
        y_sigma = F.masked_std(y_ctx, mask_ctx,  axis=-2, mask_axis=(0, 1), keepdims=True)
        y_sigma = jnp.where(y_sigma == 0, 1, y_sigma)

        x     = (x     - x_mu) / (x_sigma + 1e-5)
        x_ctx = (x_ctx - x_mu) / (x_sigma + 1e-5)
        x_tar = (x_tar - x_mu) / (x_sigma + 1e-5)

        y     = (y     - y_mu) / (y_sigma + 1e-5)
        y_ctx = (y_ctx - y_mu) / (y_sigma + 1e-5)
        y_tar = (y_tar - y_mu) / (y_sigma + 1e-5)

        return (x, y, mask, x_ctx, y_ctx, mask_ctx, x_tar, y_tar, mask_tar)
