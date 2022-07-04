"""
This is a JAX version of [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html).
This implementation is very early and rough version of mimicing torch.utils.data.DataLoader.
It is not well-tested and should be improved in the future.
It does not guarantee to work same as torch.utils.data.DataLoader.
"""

from typing import (
    Any,
    Union,
    Optional,
)

import math
import itertools

import jax
from jax import random
from jax import numpy as jnp
from jax import tree_util
from jax._src.prng import PRNGKeyArray

from flax import jax_utils

Array = Any
KeyArray = Union[Array, PRNGKeyArray]

class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Any] = None,        # FIXME: Improve typing
        batch_sampler: Optional[Any] = None,  # FIXME: Improve typing
        collate_fn: Optional[Any] = None,     # FIXME: Improve typing
        drop_last: bool = False,
        key: Optional[KeyArray] = None,
        *,
        prefetch_factor: Optional[int] = 2,
        prefetch_devices: Optional[Any] = None,  # FIXME: Improve typing
        persistent_workers: bool = False,
    ):

        if batch_sampler is not None:
            raise NotImplementedError("batch_sampler is not supported yet")

        if persistent_workers:
            raise NotImplementedError("persistent_workers is not supported yet")

        if batch_size is None and drop_last:
            raise ValueError("batch_size must be specified if drop_last is True")

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")

        if shuffle and key is None:
            raise ValueError("shuffle requires a key")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.key = key
        self.prefetch_factor = prefetch_factor
        self.prefetch_devices = prefetch_devices
        self.persistent_workers = persistent_workers

        if hasattr(self.dataset, "__len__") and hasattr(self.dataset, "__getitem__"):
            self.is_map_dataset = True
        elif hasattr(self.dataset, "__iter__"):
            self.is_map_dataset = False
        else:
            raise ValueError('dataset should implement both ("__len__", "__getitem__") or "__iter__".')

        if not self.is_map_dataset and shuffle:
            raise NotImplementedError("shuffle is not supported for IterableDataset")

        if self.collate_fn is None:
            self.collate_fn = default_collate

        self._key = key
        self._cache_iter = None

    def __len__(self):
        if not self.is_map_dataset:
            raise RuntimeError("__len__ is not supported for IterableDataset")
        if self.batch_size is None:
            return len(self.dataset)
        elif self.drop_last:
            return math.floor(len(self.dataset) / self.batch_size)
        else:
            return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        if self._key is not None:
            _, self._key = random.split(self._key)

        if self.is_map_dataset:
            data_len = len(self.dataset)

            idxs = jnp.arange(data_len, dtype=int)

            if self.shuffle:
                idxs = random.permutation(self._key, idxs)

            if self.batch_size is None:
                self._iter = (self.dataset[i] for i in idxs)
            else:
                iter_len = data_len // self.batch_size
                if data_len % self.batch_size == 0:
                    batch_range = range(0, data_len, self.batch_size)
                elif self.drop_last:
                    batch_range = range(0, iter_len * self.batch_size, self.batch_size)
                else:
                    batch_range = range(0, (iter_len + 1) * self.batch_size, self.batch_size)

                self._iter = (
                    self.collate_fn(self.dataset[idxs[batch_start:batch_start + self.batch_size]])
                    for batch_start in batch_range
                )
        elif self.batch_size:
            _dataset_iter = iter(self.dataset)

            self._iter = (
                self.collate_fn(list(itertools.islice(_dataset_iter, self.batch_size)))
                for _ in itertools.count()
            )
        else:
            # TODO: temporary solution
            self._iter = (self.collate_fn(batch) for batch in self.dataset)

        if self.prefetch_factor is not None:
            if self.prefetch_devices is None:
                self.prefetch_devices = jax.local_devices()
            if len(self.prefetch_devices) > 1 and jax.default_backend() == "gpu":
                self._iter = jax_utils.prefetch_to_device(
                    self._iter, size=self.prefetch_factor, devices=self.prefetch_devices,
                )

        return self

    def __next__(self):
        if self._iter is None:
            raise RuntimeError("__iter__ is not called yet")

        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = None
            raise StopIteration

        return batch

# Sampler

# class Sampler:
#     def __init__(self, num_data):
#         self.num_data = num_data

#     def __iter__(self):
#         raise NotImplementedError

#     # def __len__(self):
#     #     pass

# class RandomSampler(Sampler):
#     def __init__(self, num_data, key):
#         super().__init__(num_data)
#         self.key = key

#     def __iter__(self):
#         pass

# collate_fn

# TODO: Improve implementation
def default_collate(batch):
    num_elements = len(batch[0])
    return [
        jax.lax.stop_gradient(
            jnp.stack([batch[j][i] for j in range(len(batch))], axis=0)
        )
        for i in range(num_elements)
    ]

# TODO: Improve implementation
def get_shard_collate(num_replicas: Optional[int] = None, jit: bool = False):
    if num_replicas is None:
        num_replicas = jax.local_device_count()

    def shard_batch(d):
        batch_size = d.shape[0]
        if batch_size % num_replicas != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be divisible by number of replicas ({num_replicas})"
            )
        return jnp.reshape(d, (num_replicas, batch_size // num_replicas, *d.shape[1:]))

    def shard_collate(batch):
        return tree_util.tree_map(shard_batch, batch)

    if jit:
        return jax.jit(shard_collate)
    else:
        return shard_collate

# Dataset

class BaseDataset:
    pass

class Dataset(BaseDataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    # def __len__(self):
    #     pass

class IterableDataset(BaseDataset):
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError

class ArrayDataset(Dataset):
    def __init__(self, *arrays):
        if len(arrays) == 0:
            raise ValueError("ArrayDataset must contain at least one array")

        data_len = len(arrays[0])

        for array in arrays:
            if not isinstance(array, jnp.ndarray):
                raise ValueError("ArrayDataset only accepts numpy.ndarray")
            if len(array) != data_len:
                raise ValueError("ArrayDataset must contain arrays of the same length")

        self.arrays = arrays
        self.data_len = data_len

    def __getitem__(self, index):
        return tuple(array[index] for array in self.arrays)

    def __len__(self):
        return self.data_len
