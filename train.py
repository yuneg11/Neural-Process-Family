import os
from os import path
import argparse
import warnings

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange

import utils


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


def plot(model, plot_data, plot_fn, device=None, **kwargs):
    model.eval()

    results = []

    with torch.no_grad():
        for x_context, y_context, x_target, y_target in plot_data:
            if device:
                x_context = x_context.to(device)
                y_context = y_context.to(device)
                x_target = x_target.to(device)
                y_target = y_target.to(device)

            if model.is_multivariate_model:
                mu, sigma = model(x_context, y_context, x_target, as_univariate=True)
            else:
                mu, sigma = model(x_context, y_context, x_target)

            results.append((x_context, y_context, x_target, y_target, mu, sigma))

    figs = plot_fn(results, **kwargs)

    return figs


def main(
    model_name, data, data_root, comment,
    epochs, test_interval, learning_rate, weight_decay, device,
    log_root, no_logging, no_graph, no_plotting, quite, **kwargs,
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
    model = utils.get_model(model_name, **kwargs)
    model.to(device)

    if not quite:
        print(f"Model params: {model.num_params}")

    # Data
    train_loader, val_loader, test_loader = utils.get_data(data, data_root)
    plot_data, plot_fn = utils.get_plot(data)

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

    if not no_logging and not no_graph:
        tb_writer.add_graph(model, input_to_model=[
            torch.ones(16, 10, 1, device=device),
            torch.ones(16, 10, 1, device=device),
            torch.ones(16, 15, 1, device=device),
        ])

    # Train
    # FIXME: tqdm write overlapped when first launching dataloader workers
    for epoch in trange(1, epochs + 1, desc=exp_name, ncols=0):
        if not quite:
            tqdm.write(f"Epoch {epoch:3d}", end=" │ ", nolock=True)

        train_loss = train(model, train_loader, optimizer, device)
        if not quite:
            tqdm.write(f"Train loss: {train_loss:.4f}", end="", nolock=True)
        if not no_logging:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)

        val_ll = test(model, val_loader, device)
        if not quite:
            tqdm.write(f" │ Valid ll: {val_ll:.4f}", end="", nolock=True)
        if not no_logging:
            tb_writer.add_scalar("Log-Likelihood/validation", val_ll, epoch)

        if epoch % test_interval == 0 or epoch == epochs:
            test_ll = test(model, test_loader, device)
            if not quite:
                tqdm.write(f" │ Test ll: {test_ll:.4f}", end="", nolock=True)

            if not no_logging:
                tb_writer.add_scalar("Log-Likelihood/test", test_ll, epoch)

                if not no_plotting:
                    figs = plot(model, plot_data, plot_fn, device)
                    tb_writer.add_figure("Test", figs, epoch)

        if not quite:
            tqdm.write("")

        if not no_logging:
            tb_writer.flush()

        # scheduler.step()

    if not no_logging:
        tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NPF Trainer")
    parser.add_argument("model_name", type=str)
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
    parser.add_argument("-ng", "--no-graph", action="store_true")
    parser.add_argument("-np", "--no-plotting", action="store_true")
    parser.add_argument("-q", "--quite", action="store_true")
    kwargs = parser.parse_args()

    try:
        main(**vars(kwargs))
    except KeyboardInterrupt:
        pass
