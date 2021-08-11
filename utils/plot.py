import re

import torch

import npd


__all__ = [
    "get_plot",
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


def _collate_process(dataset, num_context_min=3, num_context_max=50):
    data_list = []

    for index in range(dataset.num_task):
        x, y = dataset[index]
        x, y = x[None, ...], y[None, ...]

        generator = torch.Generator().manual_seed(index)
        idx_prem = torch.randperm(x.shape[1], generator=generator, device=x.device)
        num_context = torch.randint(num_context_min, num_context_max + 1,
                                    size=(1,), generator=generator, device=x.device)
        context_idx = idx_prem[:num_context]
        x_context, y_context = x[:, context_idx, :], y[:, context_idx, :]
        x_target, y_target = x, y

        data_list.append((x_context, y_context, x_target, y_target))

    return data_list


def get_plot(data, num_task=16, **data_kwargs):
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
        dataset = npd.get_dataset(name, seed=seed*1111, num_task=num_task, plot_mode=True, **data_kwargs)
        plot_data = _collate_process(dataset, num_context_max=(100 if name == "sawtooth" else 50))
        plot_fn = npd.utils.plot.process
    else:
        raise ValueError(f"Unsupported data: '{name}'")

    return plot_data, plot_fn
