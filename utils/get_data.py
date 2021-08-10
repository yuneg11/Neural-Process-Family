import re

import npd

__all__ = [
    "get_data",
]


_abbreviations = {
    "eq": "eq",
    "mt": "matern",
    "matern": "matern",
    "wp": "weakly-periodic",
    "weakly-periodic": "weakly-periodic",
    "nm": "noisy-mixture",
    "noisy-mixture": "noisy-mixture",
    "st": "sawtooth",
    "sawtooth": "sawtooth",
}


def get_data(data, data_root, **data_kwargs):
    if "/" in data:
        try:
            match = re.match(r"(.+)\/s(\d+)", data)
            name, seed = match[1], int(match[2])
        except:
            raise ValueError("Data must be in the form of '{name}/s{seed}' or '{name}'")
    else:
        name, seed = data, 0

    name = _abbreviations[name]

    if name in ["eq", "matern", "weakly-periodic", "noisy-mixture", "sawtooth"]:
        train_loader = npd.get_data_loader(name, seed=seed,     num_task=256*16,  sample_every_epoch=True,  **data_kwargs)
        val_loader   = npd.get_data_loader(name, seed=seed*11,  num_task=60*16,   sample_every_epoch=True,  **data_kwargs)
        test_loader  = npd.get_data_loader(name, seed=seed*111, num_task=2048*16, sample_every_epoch=False, **data_kwargs)
    else:
        raise ValueError(f"Unsupported data: '{name}'")

    return train_loader, val_loader, test_loader
