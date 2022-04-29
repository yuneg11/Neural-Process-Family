import sys
sys.path.append(".")

import os
import logging

import jax
from jax import random
from jax import numpy as jnp

import flax
from flax.training import checkpoints
from flax.training.train_state import TrainState

import optax

from nxcl.rich import track
from nxcl.config import load_config, save_config, add_config_arguments, ConfigDict
from nxcl.experimental.utils import get_experiment_name, setup_logger

from npf.jax.models import *
from npf.jax.data import GPSampler, RBFKernel, PeriodicKernel, Matern52Kernel


def get_train_step(model, **kwargs):
    @jax.jit
    def train_step(state, rngs, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar):
        def loss_fn(params):
            loss = model.apply(
                params, x_ctx=x_ctx, y_ctx=y_ctx, x_tar=x_tar, y_tar=y_tar,
                mask_ctx=mask_ctx, mask_tar=mask_tar, **kwargs,
                method=model.loss, rngs=rngs,
            )
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
    return train_step


def get_eval_step(model, **kwargs):
    @jax.jit
    def eval_step(state, rngs, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar):
        log_likelihood = model.apply(
            state.params, x_ctx=x_ctx, y_ctx=y_ctx, x_tar=x_tar, y_tar=y_tar,
            mask_ctx=mask_ctx, mask_tar=mask_tar, **kwargs,
            method=model.log_likelihood, rngs=rngs,
        )
        return log_likelihood
    return eval_step


def init_model(key, model, *args, **kwargs):
    params_init_key, sample_init_key = random.split(key)

    params = model.init(dict(
        params=params_init_key,
        sample=sample_init_key,
    ), *args, **kwargs)

    tx = optax.adam(learning_rate=5e-4)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


def main(args, output_dir):

    logger = logging.getLogger(__name__)

    key = random.PRNGKey(args.seed)
    init_data = dict(
        x_ctx    = jnp.ones((2, 3, 1)),
        y_ctx    = jnp.ones((2, 3, 1)),
        x_tar    = jnp.ones((2, 4, 1)),
        mask_ctx = jnp.ones((2, 3)),
        mask_tar = jnp.ones((2, 4)),
    )

    if args.model.lower() == "CNP".lower():
        args.model = "CNP"
        model = CNP(y_dim=1)
        state = init_model(key, model, **init_data)
        kwargs = {}
    elif args.model.lower() == "NP".lower():
        args.model = "NP"
        model = NP(y_dim=1, loss_type=args.loss_type if args.loss_type else "vi")
        state = init_model(key, model, **init_data)
        kwargs = dict(num_latents=args.num_latents)
    elif args.model.lower() == "AttnCNP".lower():
        args.model = "AttnCNP"
        model = AttnCNP(y_dim=1)
        state = init_model(key, model, **init_data)
        kwargs = {}
    elif args.model.lower() == "AttnNP".lower():
        args.model = "AttnNP"
        model = AttnNP(y_dim=1)
        state = init_model(key, model, **init_data)
        kwargs = dict(num_latents=args.num_latents)
    elif args.model.lower() == "ConvCNP".lower():
        args.model = "ConvCNP"
        model = ConvCNP(y_dim=1, x_min=-2., x_max=2.)
        state = init_model(key, model, **init_data)
        kwargs = {}
    # elif args.model.lower() == "ConvNP".lower():
    #     args.model = "ConvNP"
    #     model = ConvNP(y_dim=1, x_min=-2., x_max=2.)
    #     state = init_model(key, model, **init_data)
    #     kwargs = dict(num_latents=args.num_latents)
    elif args.model.lower() == "BNP".lower():
        args.model = "BNP"
        model = BNP(y_dim=1)
        state = init_model(key, model, **init_data)
        kwargs = dict(num_samples=args.num_samples)
    elif args.model.lower() == "AttnBNP".lower():
        args.model = "AttnBNP"
        model = AttnBNP(y_dim=1)
        state = init_model(key, model, **init_data)
        kwargs = dict(num_samples=args.num_samples)
    # elif args.model.lower() == "ConvBNP".lower():
    #     args.model = "ConvBNP"
    #     model = ConvBNP(y_dim=1, x_min=-2., x_max=2.)
    #     state = init_model(key, model, **init_data)
    #     kwargs = dict(num_samples=args.num_samples)
    # elif args.model.lower() == "NeuBNP".lower():
    #     args.model = "NeuBNP"
    #     model = NeuBNP(y_dim=1)
    #     state = init_model(key, model, **init_data)
    #     kwargs = dict(num_samples=args.num_samples)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    sampler = GPSampler(RBFKernel())
    if args.eval_dataset == "RBF":
        eval_sampler = GPSampler(RBFKernel())
    elif args.eval_dataset == "Matern":
        eval_sampler = GPSampler(Matern52Kernel())
    elif args.eval_dataset == "Periodic":
        eval_sampler = GPSampler(PeriodicKernel())
    else:
        raise ValueError(f"Unknown eval dataset: {args.eval_dataset}")

    exp_dir = os.path.join("outs", args.model, "1d", args.eval_dataset, os.path.basename(output_dir))
    os.makedirs(os.path.dirname(exp_dir), exist_ok=True)
    os.symlink(os.path.join(*([".."] * 3), "_", os.path.basename(output_dir)), exp_dir)

    train_step = get_train_step(model, **kwargs)
    eval_step  = get_eval_step(model, **kwargs)

    eval_batch = eval_sampler.sample(random.PRNGKey(19), batch_size=5000)

    for i in track(range(args.num_steps), description=args.model):
        key, model_key, data_key = random.split(key, 3)
        batch = sampler.sample(data_key, batch_size=256)
        state, _ = train_step(
            state, dict(sample=model_key),
            x_ctx=batch.x_ctx, y_ctx=batch.y_ctx, x_tar=batch.x, y_tar=batch.y,
            mask_ctx=batch.mask_ctx, mask_tar=batch.mask,
        )

        if i % args.eval_every == 0:
            ll_ctx = eval_step(
                state, dict(sample=model_key),
                x_ctx=eval_batch.x_ctx, y_ctx=eval_batch.y_ctx,
                x_tar=eval_batch.x_ctx, y_tar=eval_batch.y_ctx,
                mask_ctx=eval_batch.mask_ctx, mask_tar=eval_batch.mask_ctx,
            )
            ll_tar = eval_step(
                state, dict(sample=model_key),
                x_ctx=eval_batch.x_ctx, y_ctx=eval_batch.y_ctx,
                x_tar=eval_batch.x_tar, y_tar=eval_batch.y_tar,
                mask_ctx=eval_batch.mask_ctx, mask_tar=eval_batch.mask_tar,
            )
            ll = eval_step(
                state, dict(sample=model_key),
                x_ctx=eval_batch.x_ctx, y_ctx=eval_batch.y_ctx,
                x_tar=eval_batch.x, y_tar=eval_batch.y,
                mask_ctx=eval_batch.mask_ctx, mask_tar=eval_batch.mask,
            )
            logger.info(
                f"Step {i:5d} / {args.num_steps}   CTX LL: {ll_ctx:8.4f}   TAR LL: {ll_tar:8.4f}   LL: {ll:8.4f}"
            )

        if i % args.save_every == 0:
            checkpoints.save_checkpoint(output_dir, state, i, keep=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=False, conflict_handler="resolve")
    parser.add_argument("-f", "--config-file", type=str, required=True)
    args, rest_args = parser.parse_known_args()

    config: ConfigDict = load_config(args.config_file).lock()
    add_config_arguments(parser, config, aliases={
        "train.seed":              ["-s",   "--seed"],
        "train.num_epochs":        ["-e",   "--epochs"],
        "dataset.name":            ["-d",   "--dataset"],
        "dataset.batch_size":      ["-bs",  "--batch-size"],
        "model.name":              ["-m",   "--model"],
        "optimizer.name":          ["-opt", "--optimizer"],
        "optimizer.learning_rate": ["-lr",  "--learning-rate"],
    })
    parser.add_argument("-f", "--config-file", default=argparse.SUPPRESS)
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS)
    args = parser.parse_args(rest_args)

    config.update(vars(args))

    # Logger
    log_name = get_experiment_name()
    output_dir = os.path.join("outs", "_", log_name)
    latest_link = os.path.join("outs", "_", "_latest")

    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(log_name, latest_link)
    save_config(config, os.path.join(output_dir, "config.yaml"))

    logger = setup_logger(__name__, output_dir, suppress=[jax, flax])
    logger.debug("python " + " ".join(sys.argv))

    args_str = "Configs:"
    for k, v in config.items(flatten=True):
        args_str += f"\n    {k:<25}: {v}"
    logger.info(args_str)

    logger.info(f"Output directory: \"{output_dir}\"")

    try:
        main(config, output_dir)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--model",  type=str, required=True)
    parser.add_argument("-s", "--seed",   type=int, default=0)
    parser.add_argument("--num-steps",    type=int, default=10000)
    parser.add_argument("--eval-every",   type=int, default=100)
    parser.add_argument("--save-every",   type=int, default=1000)
    parser.add_argument("--num-latents",  type=int, default=20)
    parser.add_argument("--num-samples",  type=int, default=5)
    parser.add_argument("--loss-type",    type=str, default=None)
    parser.add_argument("--eval-dataset", type=str, default="RBF")
    args = parser.parse_args()

    # Logger
    log_name = get_experiment_name()

    output_dir = os.path.join("outs", "_", log_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(__name__, output_dir, suppress=[jax, flax])

    logger.debug("python " + " ".join(sys.argv))

    args_str = "\n  Arguments:"
    for k, v in vars(args).items():
        args_str += f"\n    {k:<15}: {v}"
    logger.info(args_str + "\n")

    try:
        main(args, output_dir)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.exception(e)
