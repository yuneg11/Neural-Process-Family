from jax import random

from .base import DataLoader
from .gp import *
from .sim2real import *
from .image import *


__all__ = [
    "build_dataloader",
    "build_gp_dataset",
    "build_gp_prior_dataset",
    "build_sim2real_dataset",
    "build_image_dataset",
]


def build_dataloader(config, key, collate_fn):
    dataloader_key, dataset_key = random.split(key)

    if config.type == "GP":
        dataset = build_gp_dataset(config.gp, dataset_key)
    elif config.type == "Sim2Real":
        dataset = build_sim2real_dataset(config.sim2real)
    elif config.type == "Image":
        dataset = build_image_dataset(config.image, dataset_key)
    else:
        raise ValueError(f"Unknown dataset type: {config.type}")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
        collate_fn=collate_fn,
        key=dataloader_key,
    )

    return dataloader


def build_gp_dataset(config, key):
    if config.kernel.name == "RBF":
        kernel_cls = RBFKernel
    elif config.kernel.name == "Matern":
        kernel_cls = Matern52Kernel
    elif config.kernel.name == "Periodic":
        kernel_cls = PeriodicKernel
    else:
        raise ValueError(f"Unknown kernel: {config.kernel.name}")

    kernel = kernel_cls(
        sigma_eps=config.kernel.get("sigma_eps", 2e-2),
        max_length=config.kernel.get("max_length", 0.6),
        max_scale=config.kernel.get("max_scale", 1.0),
    )

    if config.data_size is None:
        dataset = GPIterableDataset(
            key=key,
            kernel=kernel,
            batch_size=config.batch_size,
            num_ctx=config.get("num_ctx", None),
            num_tar=config.get("num_tar", None),
            max_num_points=config.get("max_num_points", 50),
            x_range=config.get("x_range", [-2, 2]),
            t_noise=config.get("t_noise", None),
        )
    else:
        dataset = GPDataset(
            key=key,
            kernel=kernel,
            data_size=config.data_size,
            num_ctx=config.get("num_ctx", None),
            num_tar=config.get("num_tar", None),
            max_num_points=config.get("max_num_points", 50),
            x_range=config.get("x_range", [-2, 2]),
            t_noise=config.get("t_noise", None),
        )

    return dataset


def build_gp_prior_dataset(config, key):
    config.setdefault("data_size", None)
    config.setdefault("num_points", 1000)
    config.setdefault("x_range", [-2, 2])
    config.setdefault("t_noise", None)

    if config.kernel.name == "RBF":
        kernel_cls = RBFKernel
    elif config.kernel.name == "Matern":
        kernel_cls = Matern52Kernel
    elif config.kernel.name == "Periodic":
        kernel_cls = PeriodicKernel
    else:
        raise ValueError(f"Unknown kernel: {config.kernel.name}")

    kernel = kernel_cls(
        sigma_eps=config.kernel.get("sigma_eps", 2e-2),
        max_length=config.kernel.get("max_length", 0.6),
        max_scale=config.kernel.get("max_scale", 1.0),
    )

    dataset = GPPriorDataset(
        key=key,
        kernel=kernel,
        data_size=config.data_size,
        num_points=config.get("num_points", 1000),
        x_range=config.get("x_range", [-2, 2]),
        t_noise=config.get("t_noise", None),
    )

    return dataset


def build_sim2real_dataset(config):
    if config.name == "LotkaVolterra":
        dataset = LotkaVolterraDataset(
            root=config.root,
            split=config.split,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.name}")

    return dataset


def build_image_dataset(config, key):
    if config.name == "MNIST":
        dataset_cls = MNISTDataset
    elif config.name == "CIFAR10":
        dataset_cls = CIFAR10Dataset
    elif config.name == "CIFAR100":
        dataset_cls = CIFAR100Dataset
    elif config.name == "CelebA":
        dataset_cls = CelebADataset
    elif config.name == "SVHN":
        dataset_cls = SVHNDataset
    else:
        raise ValueError(f"Unknown dataset: {config.name}")

    dataset = dataset_cls(
        root=config.root,
        split=config.split,
        num_ctx=config.get("num_ctx", None),
        num_tar=config.get("num_tar", None),
        full_tar=config.get("full_tar", False),
        max_num_points=config.get("max_num_points", 200),
        flatten=config.get("flatten", False),
        key=key,
    )

    return dataset
