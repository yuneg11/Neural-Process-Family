import os
from os import path
import argparse

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange

from npf import models
from utils.data import *


def train(model, dataloader, optimizer, device=None):
    model.eval()

    losses = []

    for x_context, y_context, x_target, y_target in dataloader:
        if device:
            x_context = x_context.to(device)
            y_context = y_context.to(device)
            x_target = x_target.to(device)
            y_target = y_target.to(device)

        optimizer.zero_grad()

        loss = model.loss(x_context, y_context, x_target, y_target)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    train_loss = sum(losses) / len(losses)
    return train_loss


def test(model, dataloader, device=None):
    model.train()

    log_likelihoods = []

    with torch.no_grad():
        for x_context, y_context, x_target, y_target in dataloader:
            if device:
                x_context = x_context.to(device)
                y_context = y_context.to(device)
                x_target = x_target.to(device)
                y_target = y_target.to(device)

            ll = model.log_likelihood(x_context, y_context, x_target, y_target)
            log_likelihoods.append(ll.item())

    test_log_likelihood = sum(log_likelihoods) / len(log_likelihoods)
    return test_log_likelihood


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NPF Trainer")
    parser.add_argument("model", choices=["cnp", "np", "flownp"])
    parser.add_argument("data_dir")
    parser.add_argument("-dn", "--dataname")  #! TODO: Temporary data name
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-ti", "--test-interval", type=int, default=10)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("-wd", "--weight-decay", type=float, default=1e-5)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-log", "--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    # Temporary GPU allocation
    if args.device.startswith("cuda"):
        k = args.device.split(":")
        gpu_idx = (int(k[1]) if len(k) > 1 else 0)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        args.device = torch.device("cuda:0")

    # Model
    if args.model == "cnp":
        model = models.CNP(
            x_dim=1, y_dim=1, r_dim=128,
            encoder_dims=[128, 128],
            decoder_dims=[128, 128],
        )
    elif args.model == "np":
        model = models.NP(x_dim=1, y_dim=1)
    else:
        raise ValueError(f"Unsupported model: '{args.model}'")

    model.to(args.device)

    # Data
    train_loader = CachedDataLoader(path.join(args.data_dir, "train.pt"), device=args.device, reuse=False)
    val_loader   = CachedDataLoader(path.join(args.data_dir,   "val.pt"), device=args.device, reuse=False)
    test_loader  = CachedDataLoader(path.join(args.data_dir,  "test.pt"), device=args.device, reuse=True)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.dataname:
        print(f"Logging to TensorBoard: {args.log_dir}/{args.model}/{args.dataname}")
        tb_writer = SummaryWriter(path.join(args.log_dir, args.model, args.dataname))
    else:
        tb_writer = None

    if tb_writer:
        tb_writer.add_graph(model, input_to_model=[
            torch.ones(1, 1, 1, device=args.device),
            torch.ones(1, 1, 1, device=args.device),
            torch.ones(1, 1, 1, device=args.device),
        ])

    # Train
    for epoch in trange(1, args.epochs + 1):
        tqdm.write(f"Epoch {epoch:3d}")

        train_loss = train(model, train_loader, optimizer, args.device)
        tqdm.write(f"Train loss: {train_loss:.4f}")
        if tb_writer:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)

        val_ll = test(model, val_loader, args.device)
        tqdm.write(f"Valid log-likelihood: {val_ll:.4f}")
        if tb_writer:
            tb_writer.add_scalar("Log-Likelihood/validation", val_ll, epoch)

        if epoch % args.test_interval == 0 or epoch == args.epochs:
            test_ll = test(model, test_loader, args.device)
            tqdm.write(f"Test log-likelihood: {test_ll:.4f}")

            if tb_writer:
                tb_writer.add_scalar("Log-Likelihood/test", test_ll, epoch)

        if tb_writer:
            tb_writer.flush()

    with open(path.join(args.data_dir, f"{args.model}-{args.epochs}.txt"), "w") as f:
        f.write(f"Test log-likelihood: {test_ll:.4f}")
