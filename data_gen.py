import os
import random
import argparse

import torch
import numpy as np

from tqdm import trange

from utils import data1 as data
from stheno.torch import EQ, Delta, Matern52


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data generator")
    parser.add_argument("dataset", choices=["sawtooth", "eq", "matern", "noisy-mixture", "weakly-periodic"])
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--batch-train", type=int, default=256)
    parser.add_argument("--batch-val", type=int, default=60)
    parser.add_argument("--batch-test", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    if args.dataset == "sawtooth":
        gen = data.SawtoothGenerator(num_tasks=args.batch_train)
        gen_val = data.SawtoothGenerator(num_tasks=args.batch_val)
        gen_test = data.SawtoothGenerator(num_tasks=args.batch_test)
    else:
        if args.dataset == "eq":
            kernel = EQ().stretch(0.25)
        elif args.dataset == "matern":
            kernel = Matern52().stretch(0.25)
        elif args.dataset == "noisy-mixture":
            kernel = EQ().stretch(1.) + \
                    EQ().stretch(.25) + \
                    0.001 * Delta()
        elif args.dataset == "weakly-periodic":
            kernel = EQ().stretch(0.5) * EQ().periodic(period=0.25)
        else:
            raise ValueError(f"Unknown data '{args.dataset}'.")

        if args.noise:
            kernel = kernel + 0.05 ** 2 * Delta()

        gen = data.GPGenerator(kernel=kernel, num_tasks=args.batch_train)
        gen_val = data.GPGenerator(kernel=kernel, num_tasks=args.batch_val)
        gen_test = data.GPGenerator(kernel=kernel, num_tasks=args.batch_test)

    dataset_dirname = args.dataset + ('-noise' if args.noise else '')
    root_dir = os.path.expanduser(os.path.expandvars(args.data_dir))
    save_dir = os.path.join(root_dir, dataset_dirname, str(args.seed))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train = [[task for task in gen] for _ in trange(args.epoch, ncols=50, disable=args.no_progress)]
    torch.save(train, os.path.join(save_dir, "train.pt"))

    val = [[task for task in gen_val] for _ in trange(args.epoch, ncols=50, disable=args.no_progress)]
    torch.save(val, os.path.join(save_dir, "val.pt"))

    test = [[task for task in gen_test]]
    torch.save(test, os.path.join(save_dir, "test.pt"))
