import os
import argparse

import torch

from npf import models


MAX_T_LEN = 32


def printw(*args, end=""):
    print(*args, end=end, flush=True)


def test(msg, fn):
    try:
        printw(msg)
        out = fn()
        printw("pass")
        return True, out
    except:
        printw("fail")
        return False, None


def get_model(model_name, device):
    if model_name == "cnp":
        model = models.CNP(
            x_dim=1, y_dim=1, r_dim=128,
            encoder_dims=[128, 128],
            decoder_dims=[128, 128],
        )
    elif model_name == "np":
        model = models.NP(
            x_dim=1, y_dim=1, r_dim=128, z_dim=128,
            determ_encoder_dims=[128, 128],
            latent_encoder_dims=[128, 128],
            decoder_dims=[128, 128],
        )
    elif model_name == "attncnp":
        model = models.AttnCNP(
            x_dim=1, y_dim=1, r_dim=128,
            encoder_dims=[128, 128],
            decoder_dims=[128, 128],
        )
    elif model_name == "attnnp":
        model = models.AttnNP(
            x_dim=1, y_dim=1, r_dim=128, z_dim=128,
            determ_encoder_dims=[128, 128],
            latent_encoder_dims=[128, 128],
            decoder_dims=[128, 128],
        )
    elif model_name == "convcnp":
        model = models.ConvCNP(
            y_dim=1,
            cnn_xl=False,
        )
    elif model_name == "convcnpxl":
        model = models.ConvCNP(
            y_dim=1,
            cnn_xl=True,
        )
    elif model_name == "convnp":
        model = models.ConvNP(
            y_dim=1, z_dim=8,
            determ_cnn_xl=False,
            latent_cnn_xl=False,
        )
    elif model_name == "convnpxl":
        model = models.ConvNP(
            y_dim=1, z_dim=8,
            determ_cnn_xl=True,
            latent_cnn_xl=True,
        )

    model.to(device)

    return model


def check_conditional_model(model, data, end):
    model.eval()

    pass_test, _ = test("Forward: ", lambda: model(*data[:3]))
    pass_test, _ = test(" / Likelihood: ", lambda: model.log_likelihood(*data))
    printw(end=end)

    model.train()

    pass_test, loss = test("Loss:    ", lambda: model.loss(*data))
    if pass_test:
        pass_test, _ = test(" / Backward:   ", lambda: loss.backward())
    else:
        printw(" " * (MAX_T_LEN - 13))

    return pass_test

def check_latent_model(model, data, num_latents, end):
    model.eval()

    pass_test, _ = test("Forward: ", lambda: model(*data[:3], num_latents))
    pass_test, _ = test(" / Likelihood: ", lambda: model.log_likelihood(*data, num_latents))
    printw(end=end)

    model.train()

    pass_test, vi_loss = test("VI Loss: ", lambda: model.vi_loss(*data, num_latents))
    if pass_test:
        pass_test, _ = test(" / Backward:   ", lambda: vi_loss.backward())
    else:
        printw(" " * (MAX_T_LEN - 13))
    printw(end=end)

    pass_test, ml_loss = test("ML Loss: ", lambda: model.ml_loss(*data, num_latents))
    if pass_test:
        pass_test, _ = test(" / Backward:   ", lambda: ml_loss.backward())
    else:
        printw(" " * (MAX_T_LEN - 13))

    return pass_test


def main(
    device,
):
    x_dim = 1
    y_dim = 1

    num_batches  = 8
    num_contexts = 10
    num_targets  = 20
    num_latents  = 5

    x_context = torch.randn(num_batches, num_contexts, x_dim, device=device)
    y_context = torch.randn(num_batches, num_contexts, y_dim, device=device)
    x_target  = torch.randn(num_batches, num_targets, x_dim, device=device)
    y_target  = torch.randn(num_batches, num_targets, y_dim, device=device)
    data = (x_context, y_context, x_target, y_target)

    models = dict(
        # Conditional models
        CNP=(get_model("cnp", device), False),
        AttnCNP=(get_model("attncnp", device), False),
        ConvCNP=(get_model("convcnp", device), False),
        ConvCNPXL=(get_model("convcnpxl", device), False),
        # Latent models
        NP=(get_model("np", device), True),
        AttnNP=(get_model("attnnp", device), True),
        ConvNP=(get_model("convnp", device), True),
        ConvNPXL=(get_model("convnpxl", device), True),
    )

    max_m_len = max(map(len, models.keys()))

    num_pass = 0
    num_fail = 0

    top_border = "┌─" + "─" * max_m_len + "─┬─" + "─" * MAX_T_LEN + "─┐"
    middle_border1 = "├─" + "─" * max_m_len + "─┼─" + "─" * MAX_T_LEN + "─┤"
    middle_border2 = "├─" + "─" * max_m_len + "─┴─" + "─" * MAX_T_LEN + "─┤"
    bottom_border = "└─" + "─" * (max_m_len + MAX_T_LEN + 3) + "─┘"

    print(top_border)
    printw(f"│ {'Model':{max_m_len}s} │ {'Test':{MAX_T_LEN}s}")

    for model_name, (model, latent_model) in models.items():
        print(" │\n" + middle_border1)
        printw("│", model_name.ljust(max_m_len), "│ ")
        end = " │\n│ " + " " * max_m_len + " │ "

        if latent_model:
            pass_test = check_latent_model(model, data, num_latents, end=end)
        else:
            pass_test = check_conditional_model(model, data, end=end)

        num_pass += (1 if pass_test else 0)
        num_fail += (0 if pass_test else 1)

    print(" │\n" +middle_border2)
    pass_fail_str = f"Pass: {num_pass} / Fail: {num_fail}"
    print(f"│ {pass_fail_str:{max_m_len + MAX_T_LEN + 3}s} │")
    print(bottom_border)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Temporary GPU allocation
    if args.device.startswith("cuda"):
        k = args.device.split(":")
        gpu_idx = (int(k[1]) if len(k) > 1 else 0)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        args.device = torch.device("cuda:0")

    main(
        device=torch.device(args.device),
    )
