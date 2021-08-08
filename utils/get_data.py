import regex
from os import path

from .data import CachedDataLoader


__all__ = [
    "get_data",
]


# def get_data(data, data_root, device, **data_kwargs):
#     train_loader = CachedDataLoader(path.join(data_root, data, "train.pt"), device=device, reuse=False)
#     val_loader   = CachedDataLoader(path.join(data_root, data,   "val.pt"), device=device, reuse=False)
#     test_loader  = CachedDataLoader(path.join(data_root, data,  "test.pt"), device=device, reuse=True)
#     return train_loader, val_loader, test_loader

import npd

_abbreviations = {
    "eq": "eq",
    "mt": "matern",
    "matern": "matern",
}

def get_data(data, data_root, device, **data_kwargs):
    if "/" in data:
        try:
            match = regex.match(r"(.+)\/s(\d+)", data)
            name, seed = match[1], int(match[2])
        except:
            raise ValueError("Data must be in the form of '{name}/s{seed}' or '{name}'")
    else:
        name, seed = data, 0

    name = _abbreviations[name]

    if name in ["eq", "matern"]:
        train_loader = npd.get_data_loader(name, seed=seed,     num_task=256*16,  sample_every_epoch=True,  **data_kwargs)
        val_loader   = npd.get_data_loader(name, seed=seed*11,  num_task=60*16,   sample_every_epoch=True,  **data_kwargs)
        test_loader  = npd.get_data_loader(name, seed=seed*111, num_task=2048*16, sample_every_epoch=False, **data_kwargs)
    else:
        raise ValueError(f"Unsupported data: '{name}'")

    return train_loader, val_loader, test_loader
