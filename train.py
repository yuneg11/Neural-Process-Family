import os
from os import path
import argparse
import warnings

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange

from utils import (
    get_data,
    get_model,
)


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


def main(
    model_name, data, data_root, comment,
    epochs, test_interval, learning_rate, weight_decay, device,
    log_root, no_logging, quite, **kwargs,
):
    # Temporary GPU allocation
    if device.startswith("cuda"):
        k = device.split(":")
        gpu_idx = (int(k[1]) if len(k) > 1 else 0)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        device = torch.device("cuda:0")
    else:
        device = torch.device(device)

    # Model
    model = get_model(model_name, **kwargs)
    model.to(device)

    if not quite:
        print(f"Model params: {model.num_params}")

    # Data
    train_loader, val_loader, test_loader = get_data(data, data_root, device, **kwargs)

    # Optimizer
    optimizer_class = optim.Adam
    optimizer = optimizer_class(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

    exp_name = f"{data}/{model_name}" + (f"/{comment}" if comment else "")

    if not no_logging:
        if not quite:
            print(f"Logging to TensorBoard: {log_root}/{exp_name}")
        if comment:
            log_root = path.join(log_root, data, model_name, comment)
        else:
            log_root = path.join(log_root, data, model_name)
        tb_writer = SummaryWriter(log_root)

    if not no_logging:
        tb_writer.add_graph(model, input_to_model=[
            torch.ones(16, 10, 1, device=device),
            torch.ones(16, 10, 1, device=device),
            torch.ones(16, 15, 1, device=device),
        ])

    # Train
    for epoch in trange(1, epochs + 1, desc=exp_name, ncols=0):
        if not quite:
            tqdm.write(f"Epoch {epoch:3d}", end=" │ ")

        train_loss = train(model, train_loader, optimizer, device)
        if not quite:
            tqdm.write(f"Train loss: {train_loss:.4f}", end="")
        if not no_logging:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)

        val_ll = test(model, val_loader, device)
        if not quite:
            tqdm.write(f" │ Valid ll: {val_ll:.4f}", end="")
        if not no_logging:
            tb_writer.add_scalar("Log-Likelihood/validation", val_ll, epoch)

        if epoch % test_interval == 0 or epoch == epochs:
            test_ll = test(model, test_loader, device)
            if not quite:
                tqdm.write(f" │ Test ll: {test_ll:.4f}", end="")

            if not no_logging:
                tb_writer.add_scalar("Log-Likelihood/test", test_ll, epoch)
                with open(path.join(log_root, f"epoch{epochs}.txt"), "w") as f:
                    f.write(f"Test log-likelihood: {test_ll:.4f}")

        if not quite:
            tqdm.write("")

        if not no_logging:
            tb_writer.flush()

        # scheduler.step()

    if not no_logging:
        tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NPF Trainer")
    # parser.add_argument("model", choices=["cnp", "np", "flownp", "flowmixnp", "flowleakynp"])
    parser.add_argument("model_name")
    parser.add_argument("data", type=str)
    parser.add_argument("-dr", "--data-root", type=str, default="./data")
    parser.add_argument("-c", "--comment", type=str)
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("-llt", "--likelihood-type", choices=["univariate", "multivariate"])
    parser.add_argument("-lst", "--loss-type", choices=["vi", "ml", "univariate", "multivariate"])
    parser.add_argument("-ti", "--test-interval", type=int, default=10)
    parser.add_argument("-lr", "--learning-rate", type=float, default=3e-4)
    parser.add_argument("-wd", "--weight-decay", type=float, default=1e-5)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-ld", "--log-root", type=str, default="./logs")
    parser.add_argument("-nl", "--no-logging", action="store_true")
    parser.add_argument("-q", "--quite", action="store_true")
    kwargs = parser.parse_args()

    try:
        main(**vars(kwargs))
    except KeyboardInterrupt:
        pass
