from os import path

from .data import CachedDataLoader


__all__ = [
    "get_data",
]


def get_data(data, data_root, device, **data_kwargs):
    train_loader = CachedDataLoader(path.join(data_root, data, "train.pt"), device=device, reuse=False)
    val_loader   = CachedDataLoader(path.join(data_root, data,   "val.pt"), device=device, reuse=False)
    test_loader  = CachedDataLoader(path.join(data_root, data,  "test.pt"), device=device, reuse=True)
    return train_loader, val_loader, test_loader
