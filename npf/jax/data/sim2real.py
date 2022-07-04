try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # Python 3.7 or below

import os
from functools import partial

import jax
import jax.numpy as jnp

from npf.jax import functional as F

from .base import Dataset, NPData


__all__ = [
    "Sim2RealDatasetBase",
    "LotkaVolterraDataset",
]


SplitType = Literal["train", "valid", "test", "real"]


class Sim2RealDatasetBase(Dataset):
    def __init__(self, root: os.PathLike, name: str, split: SplitType = "train"):
        self.root = root
        self.name = name
        self.split = split

        self.base_path = os.path.join(root, name, split)

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

        return NPData(x, y, mask, x_ctx, y_ctx, mask_ctx, x_tar, y_tar, mask_tar)


class LotkaVolterraDataset(Sim2RealDatasetBase):
    def __init__(self, root: os.PathLike, split: SplitType = "train"):
        super().__init__(root, "lotka_volterra", split)
