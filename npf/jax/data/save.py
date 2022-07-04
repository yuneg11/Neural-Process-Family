# Source code modified from https://github.com/juho-lee/bnp
#
# See the original implementation below.
# https://github.com/juho-lee/bnp/blob/master/regression/data/lotka_volterra.py


import sys
sys.path.append(".")

import os
import random as pyrandom

import json
import numpy as np
from argparse import ArgumentParser

from tqdm.auto import trange

import torch
from torchvision import datasets, transforms

try:
    import numba as nb
    numba_enabled = True
except ImportError:
    numba_enabled = False

@nb.njit(nb.i4(nb.f8[:]))
def catrnd(prob):
    cprob = prob.cumsum()
    u = np.random.rand()
    for i in range(len(cprob)):
        if u < cprob[i]:
            return i
    return i

if numba_enabled:
    @nb.njit(nb.types.Tuple((nb.f8[:,:,:], nb.f8[:,:,:], nb.i4, nb.i4)) \
            (nb.i4, nb.i4, nb.i4, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8))
    def _simulate(batch_size, num_steps, max_num_points, x0, y0, theta0, theta1, theta2, theta3):
        time = np.zeros((batch_size, num_steps, 1))
        pop = np.zeros((batch_size, num_steps, 2))
        length = num_steps * np.ones((batch_size))

        for b in range(batch_size):
            pop[b, 0, 0] = max(int(x0 + np.random.randn()), 1)
            pop[b, 0, 1] = max(int(y0 + np.random.randn()), 1)

            for i in range(1, num_steps):
                X, Y = pop[b, i - 1, 0], pop[b, i - 1, 1]
                rates = np.array([theta0 * X * Y, theta1 * X, theta2 * Y, theta3 * X * Y])
                total_rate = rates.sum()

                time[b, i, 0] = time[b, i - 1, 0] + np.random.exponential(scale=1./total_rate)

                pop[b, i, 0] = pop[b, i - 1, 0]
                pop[b, i, 1] = pop[b, i - 1, 1]
                a = catrnd(rates / total_rate)
                if a == 0:
                    pop[b, i, 0] += 1
                elif a == 1:
                    pop[b, i, 0] -= 1
                elif a == 2:
                    pop[b, i, 1] += 1
                else:
                    pop[b, i, 1] -= 1

                if pop[b, i, 0] == 0 or pop[b, i, 1] == 0:
                    length[b] = i+1
                    break

        num_ctx = np.random.randint(15, max_num_points - 15)
        num_tar = np.random.randint(15, max_num_points - num_ctx)
        num_points = num_ctx + num_tar
        min_length = length.min()

        while num_points > min_length:
            num_ctx = np.random.randint(15, max_num_points - 15)
            num_tar = np.random.randint(15, max_num_points - num_ctx)
            num_points = num_ctx + num_tar

        x = np.zeros((batch_size, num_points, 1))
        y = np.zeros((batch_size, num_points, 2))

        for b in range(batch_size):
            idxs = np.arange(int(length[b]))
            np.random.shuffle(idxs)
            for j in range(num_points):
                x[b,j,0] = time[b, idxs[j], 0]
                y[b,j,0] = pop[b, idxs[j], 0]
                y[b,j,1] = pop[b, idxs[j], 1]

        return x, y, num_ctx, num_tar


class LotkaVolterraSimulator:
    def __init__(
        self,
        x0 = 50,
        y0 = 100,
        theta0 = 0.01,
        theta1 = 0.5,
        theta2 = 1.0,
        theta3 = 0.01,
    ):
        self.x0 = x0
        self.y0 = y0
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

    def simulate(self, num_batches, batch_size, num_steps=20000, max_num_points=100):
        batches = []

        for _ in trange(num_batches):
            _x, _y, num_ctx, num_tar = _simulate(
                batch_size, num_steps, max_num_points,
                self.x0, self.y0, self.theta0, self.theta1, self.theta2, self.theta3,
            )

            padding_size = max_num_points - (num_ctx + num_tar)

            x = np.concatenate(
                (_x, np.zeros((batch_size, padding_size, _x.shape[-1]), dtype=_x.dtype)),
                axis=1, dtype=np.float32,
            )

            y = np.concatenate(
                (_y, np.zeros((batch_size, padding_size, _y.shape[-1]), dtype=_y.dtype)),
                axis=1, dtype=np.float32,
            )

            mask_ctx = np.concatenate((
                np.ones((batch_size, num_ctx), dtype=bool),
                np.zeros((batch_size, max_num_points - num_ctx), dtype=bool),
            ), axis=1)

            mask_tar = np.concatenate((
                np.zeros((batch_size, num_ctx), dtype=bool),
                np.ones((batch_size, num_tar), dtype=bool),
                np.zeros((batch_size, max_num_points - (num_ctx + num_tar)), dtype=bool),
            ), axis=1)

            batch = (x, y, mask_ctx, mask_tar)
            batches.append(batch)

        return batches


def save_image_dataset(args):
    to_numpy = lambda d: np.asarray(d, dtype=np.float32)
    normalize = lambda d: (d / 255) - 0.5

    gray_transform = transforms.Compose([
        to_numpy,
        lambda d: np.expand_dims(d, axis=2),
        normalize,
    ])

    color_transform = transforms.Compose([
        to_numpy,
        normalize,
    ])

    crop_color_transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(32),
        to_numpy,
        normalize,
    ])

    if args.dataset == "mnist":
        train_dataset = datasets.MNIST(root=args.root, train=True,  transform=gray_transform, download=True)
        test_dataset  = datasets.MNIST(root=args.root, train=False, transform=gray_transform, download=True)

        train_image = np.stack([train_dataset[i][0] for i in range(len(train_dataset))], axis=0)
        test_image  = np.stack([test_dataset[i][0] for i in range(len(test_dataset))], axis=0)

        train_data = train_image[:50000]
        valid_data = train_image[50000:]
        test_data  = test_image

        dir_name = "MNIST"

    elif args.dataset == "emnist":  # https://www.tensorflow.org/datasets/catalog/emnist?hl=en#emnistbalanced
        train_dataset = datasets.EMNIST(root=args.root, split="balanced", train=True,  transform=gray_transform, download=True)
        test_dataset  = datasets.EMNIST(root=args.root, split="balanced", train=False, transform=gray_transform, download=True)

        train_image = np.stack([train_dataset[i][0] for i in range(len(train_dataset))], axis=0)
        test_image  = np.stack([test_dataset[i][0] for i in range(len(test_dataset))], axis=0)

        train_data = train_image[:90000]
        valid_data = train_image[90000:]
        test_data  = test_image

        dir_name = "EMNIST"

    elif args.dataset == "svhn":
        train_dataset = datasets.SVHN(root=args.root, split="train", transform=color_transform, download=True)
        test_dataset  = datasets.SVHN(root=args.root, split="test",  transform=color_transform, download=True)

        train_image = np.stack([train_dataset[i][0] for i in range(len(train_dataset))], axis=0)
        test_image  = np.stack([test_dataset[i][0] for i in range(len(test_dataset))], axis=0)

        train_data = train_image[:58600]
        valid_data = train_image[58600:]
        test_data  = test_image

        dir_name = "SVHN"

    elif args.dataset == "celeba":
        train_dataset = datasets.CelebA(root=args.root, split="train", transform=crop_color_transform, download=True)
        valid_dataset = datasets.CelebA(root=args.root, split="valid", transform=crop_color_transform, download=True)
        test_dataset  = datasets.CelebA(root=args.root, split="test",  transform=crop_color_transform, download=True)

        train_image = np.stack([train_dataset[i][0] for i in trange(len(train_dataset))], axis=0)
        valid_image = np.stack([valid_dataset[i][0] for i in trange(len(valid_dataset))], axis=0)
        test_image  = np.stack([test_dataset[i][0] for i in trange(len(test_dataset))], axis=0)

        train_data = train_image
        valid_data = valid_image
        test_data  = test_image

        dir_name = "celeba"

    elif args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root=args.root, train=True,  transform=color_transform, download=True)
        test_dataset  = datasets.CIFAR10(root=args.root, train=False, transform=color_transform, download=True)

        train_image = np.stack([train_dataset[i][0] for i in range(len(train_dataset))], axis=0)
        test_image  = np.stack([test_dataset[i][0] for i in range(len(test_dataset))], axis=0)

        train_data = train_image[:45000]
        valid_data = train_image[45000:]
        test_data  = test_image

        dir_name = "cifar-10-batches-py"

    elif args.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root=args.root, train=True,  transform=color_transform, download=True)
        test_dataset  = datasets.CIFAR100(root=args.root, train=False, transform=color_transform, download=True)

        train_image = np.stack([train_dataset[i][0] for i in range(len(train_dataset))], axis=0)
        test_image  = np.stack([test_dataset[i][0] for i in range(len(test_dataset))], axis=0)

        train_data = train_image[:45000]
        valid_data = train_image[45000:]
        test_data  = test_image

        dir_name = "cifar-100-python"

    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")

    os.makedirs(os.path.join(args.root, dir_name), exist_ok=True)
    np.save(os.path.join(args.root, dir_name, "train.npy"), train_data)
    np.save(os.path.join(args.root, dir_name, "valid.npy"), valid_data)
    np.save(os.path.join(args.root, dir_name, "test.npy"), test_data)


def process_batch(batch, max_num_point):
    x, y = batch["x"], batch["y"]
    xc, xt = batch["xc"], batch["xt"]

    num_batch = x.size(0)
    num_ctx, num_tar = xc.size(1), xt.size(1)
    num_pad = max_num_point - (num_ctx + num_tar)
    x_dim, y_dim = x.size(2), y.size(2)

    x = torch.cat((x, torch.zeros((num_batch, num_pad, x_dim))), dim=1)
    y = torch.cat((y, torch.zeros((num_batch, num_pad, y_dim))), dim=1)
    mask_ctx = torch.cat((
        torch.ones((num_batch, num_ctx), dtype=torch.bool),
        torch.zeros((num_batch, num_tar), dtype=torch.bool),
        torch.zeros((num_batch, num_pad), dtype=torch.bool),
    ), dim=1)
    mask_tar = torch.cat((
        torch.zeros((num_batch, num_ctx), dtype=torch.bool),
        torch.ones((num_batch, num_tar), dtype=torch.bool),
        torch.zeros((num_batch, num_pad), dtype=torch.bool),
    ), dim=1)
    return x, y, mask_ctx, mask_tar


def save_lotka_volterra(args, unknown_args):
    assert numba_enabled, "Numba should be installed to use this function."

    subparser = ArgumentParser()
    subparser.add_argument("--seed",      type=int, default=0)
    subparser.add_argument("--num-batch", type=int, default=10000)
    subparser.add_argument("--batch-size", type=int, default=50)
    subparser.add_argument("--filename", type=str, default="default")
    subparser.add_argument("--x0",     type=float, default=50)
    subparser.add_argument("--y0",     type=float, default=100)
    subparser.add_argument("--theta0", type=float, default=0.01)
    subparser.add_argument("--theta1", type=float, default=0.5)
    subparser.add_argument("--theta2", type=float, default=1.0)
    subparser.add_argument("--theta3", type=float, default=0.01)
    subparser.add_argument("--num-steps", type=int, default=20000)
    subparser.add_argument("--split",  choices=["sim", "real"], default="sim")
    subargs = subparser.parse_args(unknown_args)

    np.random.seed(subargs.seed)
    pyrandom.seed(subargs.seed)
    os.environ["PYTHONHASHSEED"] = str(subargs.seed)

    if subargs.split == "sim":
        simulator = LotkaVolterraSimulator(
            x0=subargs.x0, y0=subargs.y0,
            theta0=subargs.theta0, theta1=subargs.theta1, theta2=subargs.theta2, theta3=subargs.theta3,
        )
        batches = simulator.simulate(subargs.num_batch, subargs.batch_size, num_steps=subargs.num_steps)

        x        = np.concatenate([batch[0] for batch in batches], axis=0)
        y        = np.concatenate([batch[1] for batch in batches], axis=0)
        mask_ctx = np.concatenate([batch[2] for batch in batches], axis=0)
        mask_tar = np.concatenate([batch[3] for batch in batches], axis=0)

    else:
        filename = os.path.join(args.root, "lotka_volterra", "real", "LynxHare.txt")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            import wget
            wget.download(
                "http://people.whitman.edu/~hundledr/courses/M250F03/LynxHare.txt",
                out=os.path.join(args.root, "lotka_volterra", "real", "LynxHare.txt"),
            )
        subargs.filename = "real"

        tb = np.loadtxt(filename)
        x = np.expand_dims(tb[:, 0], -1)
        y = np.stack((tb[:, 2], tb[:, 1]), axis=-1)

        batches = []
        N = x.shape[0]

        for _ in trange(subargs.num_batch):
            num_ctx = np.random.randint(15, N - 15, size=(subargs.batch_size,))
            idxs = np.expand_dims(np.arange(N), axis=0).repeat(subargs.batch_size, axis=0)
            mask_ctx = np.stack([np.random.permutation(v) for v in (idxs < num_ctx[:, None])], axis=0)
            mask_tar = ~mask_ctx

            batch = (mask_ctx, mask_tar)
            batches.append(batch)

        x        = np.expand_dims(x, axis=0).repeat(subargs.num_batch * subargs.batch_size, axis=0)
        y        = np.expand_dims(y, axis=0).repeat(subargs.num_batch * subargs.batch_size, axis=0)
        mask_ctx = np.concatenate([batch[0] for batch in batches], axis=0)
        mask_tar = np.concatenate([batch[1] for batch in batches], axis=0)

    os.makedirs(os.path.join(args.root, "lotka_volterra", subargs.filename), exist_ok=True)
    np.save(os.path.join(args.root, "lotka_volterra", subargs.filename, "x.npy"), x)
    np.save(os.path.join(args.root, "lotka_volterra", subargs.filename, "y.npy"), y)
    np.save(os.path.join(args.root, "lotka_volterra", subargs.filename, "mask_ctx.npy"), mask_ctx)
    np.save(os.path.join(args.root, "lotka_volterra", subargs.filename, "mask_tar.npy"), mask_tar)
    with open(os.path.join(args.root, "lotka_volterra", subargs.filename, "config.json"), "w") as f:
        json.dump(vars(subargs), f, indent=2)


    for split in ("train", "valid", "test"):
        batches = torch.load(f"regression/datasets/lotka_volterra/{split}.tar")
        max_num_point = max([batch.x.size(1) for batch in batches])
        processed_batches = [process_batch(batch, max_num_point) for batch in batches]

        x        = torch.cat([x        for x, _, _,        _        in processed_batches], dim=0)
        y        = torch.cat([y        for _, y, _,        _        in processed_batches], dim=0)
        mask_ctx = torch.cat([mask_ctx for _, _, mask_ctx, _        in processed_batches], dim=0)
        mask_tar = torch.cat([mask_tar for _, _, _,        mask_tar in processed_batches], dim=0)

        np.save(f"datasets/lotka_volterra/{split}/x.npy", x.numpy())
        np.save(f"datasets/lotka_volterra/{split}/y.npy", y.numpy())
        np.save(f"datasets/lotka_volterra/{split}/mask_ctx.npy", mask_ctx.numpy())
        np.save(f"datasets/lotka_volterra/{split}/mask_tar.npy", mask_tar.numpy())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="./datasets")
    parser.add_argument("--dataset", type=str)
    args, unknown_args = parser.parse_known_args()

    if args.dataset.lower() in ["mnist", "emnist", "svhn", "celeba", "cifar10", "cifar100"]:
        if len(unknown_args) > 0:
            raise ValueError(f"Unknown arguments: {unknown_args}")
        save_image_dataset(args)
    elif args.dataset.lower() in ["lotka_volterra", "lotkavolterra"]:
        save_lotka_volterra(args, unknown_args)
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")
