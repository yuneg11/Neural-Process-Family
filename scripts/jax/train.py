import sys
sys.path.append(".")

import os
import logging
import random as pyrandom
from functools import partial

import jax
from jax import random
from jax import numpy as jnp

import flax
from flax import jax_utils
from flax.training import checkpoints
from flax.training.train_state import TrainState

import optax

from nxcl.rich import Progress
from nxcl.config import load_config, save_config, add_config_arguments, ConfigDict
from nxcl.experimental.utils import get_experiment_name, setup_logger, link_output_dir

from npf.jax.models import *
from npf.jax.data import GPSampler, RBFKernel, PeriodicKernel, Matern52Kernel


@jax.jit
def sync_metric(metric):
    return jax.tree_util.tree_map(lambda x: jnp.mean(x), metric)


@partial(jax.jit, static_argnames="num_replicas")
def shard_batch(batch, num_replicas: int):
    def _shard_batch(d):
        batch_size = d.shape[0]
        if batch_size % num_replicas != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be divisible by number of shards ({num_replicas})"
            )
        return jnp.reshape(d, (num_replicas, batch_size // num_replicas, *d.shape[1:]))
    return jax.tree_util.tree_map(_shard_batch, batch)


def get_train_step(model, **kwargs):
    @partial(jax.pmap, axis_name="batch")
    def _train_step(state, rngs, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar):
        def loss_fn(params):
            loss = model.apply(
                params, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar,
                method=model.loss, rngs=rngs, **kwargs,
            )
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train_step(state, rngs, *, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar):
        state, metric = _train_step(state, rngs, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        return state, sync_metric(metric)

    return train_step


def get_valid_step(model, **kwargs):
    @partial(jax.pmap, axis_name="batch")
    def _valid_step(state, rngs, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar):
        ll = model.apply(
            state.params, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar,
            method=model.log_likelihood, rngs=rngs, **kwargs,
        )
        return ll

    def valid_step(state, rngs, *, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar):
        metric = _valid_step(state, rngs, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        return sync_metric(metric)

    return valid_step


def main(config, output_dir):
    num_devices = jax.local_device_count()

    # Logging
    logger = logging.getLogger(__name__)

    # Random seed
    pyrandom.seed(config.train.seed)
    os.environ["PYTHONHASHSEED"] = str(config.train.seed)

    key = random.PRNGKey(config.train.seed)
    key, params_key, sample_key = random.split(key, 3)
    init_rngs = dict(params=params_key, sample=sample_key)

    # Create model
    models = {
        "CNP":     CNP,
        "NP":      NP,
        "AttnCNP": AttnCNP,
        "AttnNP":  AttnNP,
        "ConvCNP": ConvCNP,
        "BNP":     BNP,
        "AttnBNP": AttnBNP,
    }

    if config.model.name not in models:
        raise ValueError(f"Unknown model: {config.model.name}")

    model = models[config.model.name](
        y_dim=config.datasets.shapes.y_ctx[-1],
        **config.model.get("kwargs", {}),
    )
    params = model.init(
        init_rngs,
        x_ctx=jnp.zeros((num_devices, *config.datasets.shapes.x_ctx[1:])),
        y_ctx=jnp.zeros((num_devices, *config.datasets.shapes.y_ctx[1:])),
        x_tar=jnp.zeros((num_devices, *config.datasets.shapes.x_tar[1:])),
        mask_ctx=jnp.zeros((num_devices, *config.datasets.shapes.mask_ctx[1:])),
        mask_tar=jnp.zeros((num_devices, *config.datasets.shapes.mask_tar[1:])),
        **config.model.get("init_kwargs", {}),
    )

    if config.optimizer.name == "adam":
        tx = optax.adam(learning_rate=float(config.optimizer.learning_rate))
    elif config.optimizer.name == "sgd":
        tx = optax.sgd(learning_rate=float(config.optimizer.learning_rate))
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer.name}")

    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = jax_utils.replicate(state)

    # Create dataset
    if config.datasets.train.name == "RBF":
        train_sampler = GPSampler(RBFKernel())
    elif config.datasets.train.name == "Matern":
        train_sampler = GPSampler(Matern52Kernel())
    elif config.datasets.train.name == "Periodic":
        train_sampler = GPSampler(PeriodicKernel())
    else:
        raise ValueError(f"Unknown train dataset: {config.datasets.train.name}")

    if config.datasets.valid.name == "RBF":
        valid_sampler = GPSampler(RBFKernel())
    elif config.datasets.valid.name == "Matern":
        valid_sampler = GPSampler(Matern52Kernel())
    elif config.datasets.valid.name == "Periodic":
        valid_sampler = GPSampler(PeriodicKernel())
    else:
        raise ValueError(f"Unknown valid dataset: {config.datasets.valid.name}")

    train_batch_size = config.datasets.train.batch_size
    valid_batch_size = config.datasets.valid.batch_size

    valid_batch = shard_batch(valid_sampler.sample(random.PRNGKey(19), batch_size=valid_batch_size), num_devices)

    # Setup output directory
    link_output_dir(output_dir, subnames=(config.model.name, config.datasets.train.name, config.datasets.valid.name))

    # Build steps
    train_step = get_train_step(model, **config.model.get("train_kwargs", {}))
    valid_step = get_valid_step(model, **config.model.get("valid_kwargs", {}))

    # Train
    with Progress() as p:
        for i in p.trange(1, config.train.num_steps + 1, description=config.model.name):
            key, model_key, data_key = random.split(key, 3)
            batch = shard_batch(train_sampler.sample(data_key, batch_size=train_batch_size), num_devices)
            state, _ = train_step(
                state, jax_utils.replicate(dict(sample=model_key)),
                x_ctx=batch.x_ctx,       x_tar=batch.x,
                y_ctx=batch.y_ctx,       y_tar=batch.y,
                mask_ctx=batch.mask_ctx, mask_tar=batch.mask,
            )

            if i % config.train.valid_every == 0:
                ll_ctx = valid_step(
                    state, jax_utils.replicate(dict(sample=model_key)),
                    x_ctx=valid_batch.x_ctx,       x_tar=valid_batch.x_ctx,
                    y_ctx=valid_batch.y_ctx,       y_tar=valid_batch.y_ctx,
                    mask_ctx=valid_batch.mask_ctx, mask_tar=valid_batch.mask_ctx,
                )
                ll_tar = valid_step(
                    state, jax_utils.replicate(dict(sample=model_key)),
                    x_ctx=valid_batch.x_ctx,       x_tar=valid_batch.x_tar,
                    y_ctx=valid_batch.y_ctx,       y_tar=valid_batch.y_tar,
                    mask_ctx=valid_batch.mask_ctx, mask_tar=valid_batch.mask_tar,
                )
                ll = valid_step(
                    state, jax_utils.replicate(dict(sample=model_key)),
                    x_ctx=valid_batch.x_ctx,       x_tar=valid_batch.x,
                    y_ctx=valid_batch.y_ctx,       y_tar=valid_batch.y,
                    mask_ctx=valid_batch.mask_ctx, mask_tar=valid_batch.mask,
                )
                logger.info(
                    f"Step {i:5d} / {config.train.num_steps}   "
                    f"CTX LL: {ll_ctx:8.4f}   TAR LL: {ll_tar:8.4f}   LL: {ll:8.4f}"
                )

            if i % config.train.save_every == 0 and jax.process_index() == 0:
                checkpoints.save_checkpoint(
                    output_dir, jax_utils.unreplicate(state),
                    step=i, prefix="checkpoint_", keep=1,
                )

    logger.info("Finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=False, conflict_handler="resolve")
    parser.add_argument("-f", "--config-file", type=str, required=True)
    args, rest_args = parser.parse_known_args()

    config: ConfigDict = load_config(args.config_file).lock()
    add_config_arguments(parser, config, aliases={
        "train.seed":               ["-s",   "--seed"],
        "train.num_steps":          ["-e",   "--num-steps"],
        "dataset.train.name":       ["-td",  "--train-dataset"],
        "dataset.valid.name":       ["-vd",  "--valid-dataset"],
        "dataset.train.batch_size": ["-tbs", "--train-batch-size"],
        "dataset.valid.batch_size": ["-vbs", "--valid-batch-size"],
        "model.name":               ["-m",   "--model"],
        "model.num_latents":        ["-nl",  "--num-latents"],
        "model.num_samples":        ["-ns",  "--num-samples"],
        "model.loss_type":          ["-lt",  "--loss-type"],
        "optimizer.name":           ["-opt", "--optimizer"],
        "optimizer.learning_rate":  ["-lr",  "--learning-rate"],
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
