from typing import Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # Python 3.7 or below


import os
import math
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from jax._src.prng import PRNGKeyArray

import numpy as np

from npf.jax import functional as F

from .base import Dataset, NPData


__all__ = [
    "ImageDataset",
    "MNISTDataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "CelebADataset",
    "SVHNDataset",
]


SplitType = Literal["train", "valid", "test"]


class ImageDataset(Dataset):
    chunk_size = 64

    def __init__(
        self,
        root: os.PathLike,
        name: str,
        split: SplitType = "train",
        num_ctx: Optional[int] = None,
        num_tar: Optional[int] = None,
        full_tar: bool = False,
        max_num_points: int = 200,
        flatten: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ):

        if full_tar and num_tar is not None:
            raise ValueError("full_tar and num_tar cannot be used together")

        self.key = key
        self.root = root
        self.name = name
        self.split = split

        self.images = np.load(os.path.join(root, name, f"{split}.npy"))  # [Data num, Height, Width, Channel]

        self.num_ctx = num_ctx
        self.num_tar = num_tar
        self.full_tar = full_tar
        self.max_num_points = max_num_points
        self.flatten = flatten

        n, h, w, c = self.images.shape
        h_coord = np.linspace(-1 + (1 / h), +1 - (1 / h), num=h, endpoint=True)
        w_coord = np.linspace(-1 + (1 / w), +1 - (1 / w), num=w, endpoint=True)
        self.coord = np.transpose((np.repeat(w_coord, h), np.tile(h_coord, w))).reshape(1, h, w, 2)

        if self.key is None:
            self.key = random.PRNGKey(19)

        self.data_size = n

        num_chunks = math.ceil(self.data_size / self.chunk_size)
        key, key_list = self.key, []
        for _ in range(num_chunks):
            key, key_i = random.split(key)
            key_list.append(key_i)
        self.keys = jnp.vstack(key_list)

        chunks = [
            self.build_chunk(self.keys[i], self.images[i * self.chunk_size : (i + 1) * self.chunk_size])
            for i in range(num_chunks)
        ]

        self.num_elem = len(chunks[0])
        self._cache = NPData(*[
            np.concatenate([np.asarray(c[i]) for c in chunks], axis=0)
            for i in range(self.num_elem)
        ])

    @partial(jax.jit, static_argnames=("self",))
    def build_chunk(self, key, images):
        keys = random.split(key, 3)
        n, h, w, c = images.shape
        hw = h * w

        if self.num_ctx is None:
            num_ctx = random.randint(keys[0], shape=(n,), minval=3, maxval=self.max_num_points-3)
        else:
            num_ctx = np.full(shape=(n,), fill_value=self.num_ctx)

        if self.full_tar:
            num_tar = hw - num_ctx
        elif self.num_tar is None:
            num_tar = random.randint(keys[1], shape=(n,), minval=3, maxval=(self.max_num_points-num_ctx))
        else:
            num_tar = np.full(shape=(n,), fill_value=self.num_tar)

        num_points = num_ctx + num_tar

        _mask     = jax.vmap(lambda _p:     F.get_mask(hw, start=0,  stop=_p))(num_points)
        _mask_ctx = jax.vmap(lambda _c:     F.get_mask(hw, start=0,  stop=_c))(num_ctx)
        _mask_tar = jax.vmap(lambda _c, _p: F.get_mask(hw, start=_c, stop=_p))(num_ctx, num_points)
        _mask = random.permutation(keys[2], jnp.stack((_mask, _mask_ctx, _mask_tar), axis=-1), axis=1)
        mask, mask_ctx, mask_tar = _mask[..., 0], _mask[..., 1], _mask[..., 2]

        x = jnp.repeat(self.coord, repeats=n, axis=0)
        y = images

        if self.flatten:
            x = x.reshape(n, hw, 2)
            y = y.reshape(n, hw, c)
            mask_axis = (0, -2)
        else:
            mask = mask.reshape(n, h, w)
            mask_ctx = mask_ctx.reshape(n, h, w)
            mask_tar = mask_tar.reshape(n, h, w)
            mask_axis = (0, -3, -2)

        chunk = NPData(
            x        = F.masked_fill(x, mask,     mask_axis=mask_axis, fill_value=0),
            y        = F.masked_fill(y, mask,     mask_axis=mask_axis, fill_value=0),
            mask     = mask,
            x_ctx    = F.masked_fill(x, mask_ctx, mask_axis=mask_axis, fill_value=0),
            y_ctx    = F.masked_fill(y, mask_ctx, mask_axis=mask_axis, fill_value=0),
            mask_ctx = mask_ctx,
            x_tar    = F.masked_fill(x, mask_tar, mask_axis=mask_axis, fill_value=0),
            y_tar    = F.masked_fill(y, mask_tar, mask_axis=mask_axis, fill_value=0),
            mask_tar = mask_tar,
        )
        return chunk

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return NPData(*[self._cache[i][index] for i in range(self.num_elem)])


class MNISTDataset(ImageDataset):
    def __init__(
        self,
        root: os.PathLike,
        split: SplitType = "train",
        num_ctx: Optional[int] = None,
        num_tar: Optional[int] = None,
        full_tar: bool = False,
        max_num_points: int = 200,
        flatten: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ):
        super().__init__(root, "MNIST", split, num_ctx, num_tar, full_tar, max_num_points, flatten, key)


class CIFAR10Dataset(ImageDataset):
    def __init__(
        self,
        root: os.PathLike,
        split: SplitType = "train",
        num_ctx: Optional[int] = None,
        num_tar: Optional[int] = None,
        full_tar: bool = False,
        max_num_points: int = 200,
        flatten: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ):
        super().__init__(root, "cifar-10-batches-py", split, num_ctx, num_tar, full_tar, max_num_points, flatten, key)


class CIFAR100Dataset(ImageDataset):
    def __init__(
        self,
        root: os.PathLike,
        split: SplitType = "train",
        num_ctx: Optional[int] = None,
        num_tar: Optional[int] = None,
        full_tar: bool = False,
        max_num_points: int = 200,
        flatten: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ):
        super().__init__(root, "cifar-100-python", split, num_ctx, num_tar, full_tar, max_num_points, flatten, key)


class CelebADataset(ImageDataset):
    def __init__(
        self,
        root: os.PathLike,
        split: SplitType = "train",
        num_ctx: Optional[int] = None,
        num_tar: Optional[int] = None,
        full_tar: bool = False,
        max_num_points: int = 200,
        flatten: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ):
        super().__init__(root, "celeba", split, num_ctx, num_tar, full_tar, max_num_points, flatten, key)


class SVHNDataset(ImageDataset):
    def __init__(
        self,
        root: os.PathLike,
        split: SplitType = "train",
        num_ctx: Optional[int] = None,
        num_tar: Optional[int] = None,
        full_tar: bool = False,
        max_num_points: int = 200,
        flatten: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ):
        super().__init__(root, "svhn", split, num_ctx, num_tar, full_tar, max_num_points, flatten, key)
