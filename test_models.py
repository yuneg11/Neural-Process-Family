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
    elif model_name == "gnp":
        model = models.GNP(
            y_dim=1,
            mean_cnn_xl=False,
            kernel_cnn_xl=False,
        )
    elif model_name == "gnpxl":
        model = models.GNP(
            y_dim=1,
            mean_cnn_xl=True,
            kernel_cnn_xl=True,
        )

    model.to(device)

    return model


def check_conditional_model(model, data, end):
    model.eval()

    pass_forward, _ = test("Forward: ", lambda: model(*data[:3]))
    pass_likelihood, _ = test(" / Likelihood: ", lambda: model.log_likelihood(*data))
    printw(end=end)

    model.train()

    pass_loss, loss = test("Loss:    ", lambda: model.loss(*data))
    if pass_loss:
        pass_loss_back, _ = test(" / Backward:   ", lambda: loss.backward())
    else:
        pass_loss_back = False
        printw(" " * (MAX_T_LEN - 13))

    pass_test = pass_forward and pass_likelihood \
                and pass_loss and pass_loss_back

    return pass_test

def check_latent_model(model, data, num_latents, end):
    model.eval()

    pass_forward, _ = test("Forward: ", lambda: model(*data[:3], num_latents))
    pass_likelihood, _ = test(" / Likelihood: ", lambda: model.log_likelihood(*data, num_latents))
    printw(end=end)

    model.train()

    pass_viloss, vi_loss = test("VI Loss: ", lambda: model.vi_loss(*data, num_latents))
    if pass_viloss:
        pass_viloss_back, _ = test(" / Backward:   ", lambda: vi_loss.backward())
    else:
        pass_viloss_back = False
        printw(" " * (MAX_T_LEN - 13))
    printw(end=end)

    pass_mlloss, ml_loss = test("ML Loss: ", lambda: model.ml_loss(*data, num_latents))
    if pass_mlloss:
        pass_mlloss_back, _ = test(" / Backward:   ", lambda: ml_loss.backward())
    else:
        pass_mlloss_back = False
        printw(" " * (MAX_T_LEN - 13))

    pass_test = pass_forward and pass_likelihood \
                and pass_viloss and pass_viloss_back \
                and pass_mlloss and pass_mlloss_back

    return pass_test


def main(
    models,
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

    model_lambdas = dict(
        # Conditional models
        CNP=(lambda: get_model("cnp", device), False),
        AttnCNP=(lambda: get_model("attncnp", device), False),
        ConvCNP=(lambda: get_model("convcnp", device), False),
        ConvCNPXL=(lambda: get_model("convcnpxl", device), False),
        # Latent models
        NP=(lambda: get_model("np", device), True),
        AttnNP=(lambda: get_model("attnnp", device), True),
        ConvNP=(lambda: get_model("convnp", device), True),
        ConvNPXL=(lambda: get_model("convnpxl", device), True),
        # Multivariate models
        GNP=(lambda: get_model("gnp", device), False),
        GNPXL=(lambda: get_model("gnpxl", device), False),
    )

    max_m_len = max(map(len, model_lambdas.keys()))

    num_pass = 0
    num_fail = 0

    top_border = "┌─" + "─" * max_m_len + "─┬─" + "─" * MAX_T_LEN + "─┐"
    middle_border1 = "├─" + "─" * max_m_len + "─┼─" + "─" * MAX_T_LEN + "─┤"
    middle_border2 = "├─" + "─" * max_m_len + "─┴─" + "─" * MAX_T_LEN + "─┤"
    bottom_border = "└─" + "─" * (max_m_len + MAX_T_LEN + 3) + "─┘"

    print(top_border)
    printw(f"│ {'Model':{max_m_len}s} │ {'Test':{MAX_T_LEN}s}")

    for model_name, (model_lambda, latent_model) in model_lambdas.items():
        if len(models) > 0 and model_name.lower() not in models:
            continue

        model = model_lambda()

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
    parser.add_argument("models", type=str, nargs="*")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Temporary GPU allocation
    if args.device.startswith("cuda"):
        k = args.device.split(":")
        gpu_idx = (int(k[1]) if len(k) > 1 else 0)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        args.device = torch.device("cuda:0")

    main(
        models=args.models,
        device=torch.device(args.device),
    )
