import torch

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device(device=None):
    if device is None:
        return default_device
    else:
        if isinstance(device, str):
            return torch.device(device)
        else:
            return device
