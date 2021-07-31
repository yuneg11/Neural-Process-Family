import os
from os import path
import argparse
import warnings

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange

from npf import models
from utils.data import *


warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train(model, dataloader, optimizer, device=None):
    model.train()

    losses = []

    for x_context, y_context, x_target, y_target in dataloader:
        if device:
            x_context = x_context.to(device)
            y_context = y_context.to(device)
            x_target = x_target.to(device)
            y_target = y_target.to(device)

        optimizer.zero_grad()

        if model.is_latent_model:
            loss = model.loss(x_context, y_context, x_target, y_target, num_latents=20)
        else:
            loss = model.loss(x_context, y_context, x_target, y_target)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    train_loss = sum(losses) / len(losses)
    return train_loss


def test(model, dataloader, device=None):
    model.eval()

    log_likelihoods = []

    with torch.no_grad():
        for x_context, y_context, x_target, y_target in dataloader:
            if device:
                x_context = x_context.to(device)
                y_context = y_context.to(device)
                x_target = x_target.to(device)
                y_target = y_target.to(device)

            if model.is_latent_model:
                ll = model.log_likelihood(x_context, y_context, x_target, y_target, num_latents=20)
            else:
                ll = model.log_likelihood(x_context, y_context, x_target, y_target)
            log_likelihoods.append(ll.item())

    test_log_likelihood = sum(log_likelihoods) / len(log_likelihoods)
    return test_log_likelihood


def main():
    parser = argparse.ArgumentParser("NPF Trainer")
    # parser.add_argument("model", choices=["cnp", "np", "flownp", "flowmixnp", "flowleakynp"])
    parser.add_argument("model")
    parser.add_argument("data_dir")
    parser.add_argument("-dn", "--dataname")  # TODO: Temporary data name
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("-lt", "--loss-type", choices=["vi", "ml"])
    parser.add_argument("-ti", "--test-interval", type=int, default=10)
    parser.add_argument("-lr", "--learning-rate", type=float, default=3e-4)
    parser.add_argument("-wd", "--weight-decay", type=float, default=1e-5)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-l", "--log-dir", type=str, default="./logs")
    parser.add_argument("-q", "--quite", action="store_true")
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
        model = models.NP(
            x_dim=1, y_dim=1, r_dim=128, z_dim=128,
            determ_encoder_dims=[128, 128],
            latent_encoder_dims=[128, 128],
            decoder_dims=[128, 128],
            loss_type=args.loss_type,
        )
    elif args.model == "attncnp":
        model = models.AttnCNP(
            x_dim=1, y_dim=1, r_dim=128,
            encoder_dims=[128, 128],
            decoder_dims=[128, 128],
        )
    elif args.model == "attnnp":
        model = models.AttnNP(
            x_dim=1, y_dim=1, r_dim=128, z_dim=128,
            determ_encoder_dims=[128, 128],
            latent_encoder_dims=[128, 128],
            decoder_dims=[128, 128],
            loss_type=args.loss_type,
        )
    elif args.model == "convcnp":
        model = models.ConvCNP(
            y_dim=1,
            cnn_xl=False,
        )
    elif args.model == "convcnpxl":
        model = models.ConvCNP(
            y_dim=1,
            cnn_xl=True,
        )
    elif args.model == "convnp":
        model = models.ConvNP(
            y_dim=1, z_dim=8,
            determ_cnn_xl=False,
            latent_cnn_xl=False,
            loss_type=args.loss_type,
        )
    elif args.model == "convnpxl":
        model = models.ConvNP(
            y_dim=1, z_dim=8,
            determ_cnn_xl=True,
            latent_cnn_xl=True,
            loss_type=args.loss_type,
        )
    elif args.model == "gnp":
        model = models.GNP(
            y_dim=1,
            mean_cnn_xl=False,
            kernel_cnn_xl=False,
        )
    elif args.model == "gnpxl":
        model = models.GNP(
            y_dim=1,
            mean_cnn_xl=True,
            kernel_cnn_xl=True,
        )
    else:
        raise ValueError(f"Unsupported model: '{args.model}'")

    model.to(args.device)

    if not args.quite:
        print(f"Model params: {model.num_params}")

    # Data
    train_loader = CachedDataLoader(path.join(args.data_dir, "train.pt"), device=args.device, reuse=False)
    val_loader   = CachedDataLoader(path.join(args.data_dir,   "val.pt"), device=args.device, reuse=False)
    test_loader  = CachedDataLoader(path.join(args.data_dir,  "test.pt"), device=args.device, reuse=True)

    # Optimizer
    optimizer = optim.Adam(
    # optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

    if args.dataname:
        if not args.quite:
            print(f"Logging to TensorBoard: {args.log_dir}/{args.model}/{args.dataname}")
        log_root = path.join(args.log_dir, args.model, args.dataname)
        tb_writer = SummaryWriter(log_root)
        logging = True
    else:
        logging = False

    if logging:
        tb_writer.add_graph(model, input_to_model=[
            torch.ones(16, 10, 1, device=args.device),
            torch.ones(16, 10, 1, device=args.device),
            torch.ones(16, 15, 1, device=args.device),
        ])

    # Train
    for epoch in trange(1, args.epochs + 1, desc=f"{args.model}/{args.dataname}", ncols=0):
        if not args.quite:
            tqdm.write(f"Epoch {epoch:3d}", end=" │ ", nolock=True)

        train_loss = train(model, train_loader, optimizer, args.device)
        if not args.quite:
            tqdm.write(f"Train loss: {train_loss:.4f}", end="", nolock=True)
        if logging:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)

        val_ll = test(model, val_loader, args.device)
        if not args.quite:
            tqdm.write(f" │ Valid ll: {val_ll:.4f}", end="", nolock=True)
        if logging:
            tb_writer.add_scalar("Log-Likelihood/validation", val_ll, epoch)

        if epoch % args.test_interval == 0 or epoch == args.epochs:
            test_ll = test(model, test_loader, args.device)
            if not args.quite:
                tqdm.write(f" │ Test ll: {test_ll:.4f}", end="", nolock=True)

            if logging:
                tb_writer.add_scalar("Log-Likelihood/test", test_ll, epoch)
                with open(path.join(log_root, f"epoch{args.epochs}.txt"), "w") as f:
                    f.write(f"Test log-likelihood: {test_ll:.4f}")

        if not args.quite:
            tqdm.write("")

        if logging:
            tb_writer.flush()

        # scheduler.step()

    if logging:
        tb_writer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
