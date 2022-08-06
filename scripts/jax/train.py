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

import nxcl
from nxcl.rich import Progress
from nxcl.config import load_config, save_config, add_config_arguments, ConfigDict
from nxcl.experimental.utils import get_experiment_name, setup_logger, AverageMeter

from npf.jax import models, NPData
from npf.jax.data import get_shard_collate, build_dataloader


def link_output_dir(output_dir: str, subnames):
    link_dir = os.path.join("outs", *subnames, os.path.basename(output_dir))
    os.makedirs(os.path.dirname(link_dir), exist_ok=True)
    relpath = os.path.relpath(output_dir, os.path.dirname(link_dir))
    os.symlink(relpath, link_dir)


@jax.jit
def sync_metric(metric):
    return jax.tree_util.tree_map(lambda x: jnp.mean(x), metric)


def get_train_step(model, **kwargs):
    @partial(jax.pmap, axis_name="batch")
    def _train_step(state, rngs, batch):
        def loss_fn(params):
            outs = model.apply(params, batch, method=model.loss, rngs=rngs, **kwargs)
            if isinstance(outs, tuple):
                loss, aux = outs
            else:
                loss, aux = outs, None
            return loss, aux

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        state = state.apply_gradients(grads=grads)

        if aux is None:
            return state, dict(loss=loss)
        else:
            return state, dict(loss=loss, aux=aux)

    def train_step(state, rngs, batch):
        state, metric = _train_step(state, rngs, batch)
        return state, sync_metric(metric)

    return train_step


def get_valid_step(model, **kwargs):
    @partial(jax.pmap, axis_name="batch")
    def _valid_step(state, rngs, batch):
        ll = model.apply(state.params, batch, method=model.log_likelihood, rngs=rngs, **kwargs)
        return ll

    def valid_step(state, rngs, batch):
        metric = _valid_step(state, rngs, batch)
        return sync_metric(metric)

    return valid_step


def main(config, output_dir):
    num_devices = jax.local_device_count()

    # Logging
    logger = logging.getLogger(__name__)
    logger.info(f"Number of devices: {num_devices}")

    # Random seed
    pyrandom.seed(config.train.seed)
    os.environ["PYTHONHASHSEED"] = str(config.train.seed)

    key = random.PRNGKey(config.train.seed)
    key, params_key, sample_key = random.split(key, 3)
    init_rngs = dict(params=params_key, sample=sample_key)

    # Create model
    if not hasattr(models, config.model.name):
        raise ValueError(f"Unknown model: {config.model.name}")

    model = getattr(models, config.model.name)(
        y_dim=config.datasets.shapes.y[-1],
        **config.model.get("kwargs", {}),
    )

    try:
        params = model.init(
            init_rngs,
            NPData(
                x=jnp.zeros((num_devices, *config.datasets.shapes.x)),
                y=jnp.zeros((num_devices, *config.datasets.shapes.y)),
                mask_ctx=jnp.zeros((num_devices, *config.datasets.shapes.mask_ctx)),
                mask_tar=jnp.zeros((num_devices, *config.datasets.shapes.mask_tar)),
            ),
            **config.model.get("init_kwargs", {}),
            training=True,
        )
    except:
        params = model.init(
            init_rngs,
            NPData(
                x=jnp.zeros((num_devices, *config.datasets.shapes.x)),
                y=jnp.zeros((num_devices, *config.datasets.shapes.y)),
                mask_ctx=jnp.zeros((num_devices, *config.datasets.shapes.mask_ctx)),
                mask_tar=jnp.zeros((num_devices, *config.datasets.shapes.mask_tar)),
            ),
            **config.model.get("init_kwargs", {}),
        )

    param_shapes = jax.tree_util.tree_map(lambda v: () if isinstance(v, float) else v.shape, params)
    num_params = jax.tree_util.tree_reduce(lambda a, v: a + (1 if isinstance(v, float) else v.size), params, 0)

    logger.debug(f"Parameter shapes: {param_shapes}")
    logger.info(f"Number of parameters: {num_params}")

    # Create dataset
    key, train_key = random.split(key)
    valid_key = random.PRNGKey(42)

    shard_collate = get_shard_collate(num_replicas=num_devices, jit=True)
    train_loader = build_dataloader(config.datasets.train, train_key, shard_collate)
    valid_loader = build_dataloader(config.datasets.valid, valid_key, shard_collate)

    if train_loader.is_map_dataset:
        num_step_per_epoch = config.train.get("num_step_per_epoch", len(train_loader))
    else:
        num_step_per_epoch = config.train.num_step_per_epoch
        train_iterator = iter(train_loader)

    # Create optimizer
    if config.optimizer.use_scheduler:
        lr = schedule = optax.cosine_decay_schedule(
            config.optimizer.learning_rate,
            num_step_per_epoch * config.train.num_epochs,
        )
    else:
        lr = float(config.optimizer.learning_rate)
        schedule = lambda step: lr  # For logging

    if config.optimizer.name == "adam":
        tx = optax.adam(learning_rate=lr)
    elif config.optimizer.name == "sgd":
        tx = optax.sgd(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer.name}")

    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = jax_utils.replicate(state)

    # Setup output directory
    link_output_dir(output_dir, subnames=(config.model.name, "Train", config.datasets.train.name))

    # Build steps
    train_step = get_train_step(model, **config.model.get("train_kwargs", {}))
    valid_step = get_valid_step(model, **config.model.get("valid_kwargs", {}))

    # Train
    train_meter = AverageMeter("loss")
    valid_meter = AverageMeter("ll_ctx", "ll_tar", "ll")
    best_ll, best_epoch, best_state = -jnp.inf, 0, None
    aux_meter = None

    with Progress(speed_estimate_period=300) as p:
        for i in p.trange(1, config.train.num_epochs + 1, description=config.model.name):
            train_meter.reset()

            if train_loader.is_map_dataset:
                iter_loader = (v for v in train_loader)
            else:
                iter_loader = (next(train_iterator) for _ in range(num_step_per_epoch))

            for batch in p.track(iter_loader, description="Train", remove=True, total=num_step_per_epoch):
                key, model_key = random.split(key)

                state, train_metric = train_step(
                    state=state, rngs=jax_utils.replicate(dict(sample=model_key)), batch=batch,
                )

                train_meter.update(loss=train_metric["loss"], n=len(batch.x))

                if "aux" in train_metric:
                    if aux_meter is None:
                        aux_meter = AverageMeter(*train_metric["aux"].keys())
                    aux_meter.update(train_metric["aux"])

            logger.info(
                f"Epoch {i:3d} / {config.train.num_epochs:4d} | Train Loss: {train_meter.loss:7.4f}"
                + (f" | LR: {schedule(state.step[0]):.4e}" if config.optimizer.use_scheduler else "")
                + ("" if aux_meter is None else " | " + "  ".join([f"{k}: {v:7.4f}" for k, v in aux_meter.value.items()]))
            )

            if aux_meter is not None:
                aux_meter.reset()

            if i % config.train.valid_every == 0:
                valid_meter.reset()

                for batch in p.track(valid_loader, description="Valid", remove=True):
                    key, model_key = random.split(key)
                    replicated_rngs = jax_utils.replicate(dict(sample=model_key))

                    ll_ctx = valid_step(state=state, rngs=replicated_rngs, batch=batch)
                    ll_tar = valid_step(state=state, rngs=replicated_rngs, batch=batch)
                    ll     = valid_step(state=state, rngs=replicated_rngs, batch=batch)

                    valid_meter.update(ll_ctx=ll_ctx, ll_tar=ll_tar, ll=ll, n=len(batch.x))

                ll_ctx, ll_tar, ll = valid_meter.ll_ctx, valid_meter.ll_tar, valid_meter.ll

                logger.info(
                    f"                 | "
                    f"Valid CTX LL: {ll_ctx:7.4f}  TAR LL: {ll_tar:7.4f}  LL: {ll:7.4f}"
                    f"{'  (Best LL)' if ll > best_ll else ''}"
                )

                if ll > best_ll and jax.process_index() == 0:
                    best_ll, best_epoch, best_state = ll, i, state

            if i % config.train.save_every == 0 and jax.process_index() == 0:
                checkpoints.save_checkpoint(
                    output_dir, jax_utils.unreplicate(state), step=i, prefix="ckpt_epoch_",
                )

                if best_state is not None:
                    checkpoints.save_checkpoint(
                        output_dir, jax_utils.unreplicate(best_state), step=best_epoch, prefix="ckpt_best_ll_",
                    )
                    best_state = None

    logger.info("Finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=False, conflict_handler="resolve")
    parser.add_argument("-f", "--config-file", type=str, required=True)
    args, rest_args = parser.parse_known_args()

    config: ConfigDict = load_config(args.config_file)
    # You can add alias to config here.
    # ex) In command line, you can use '-e 100' or '--num-epochs 100' instead of '--train.num_epochs 100'.
    #     But the config key 'train.num_epochs' should exist in the config file (at least with a dummy value).
    add_config_arguments(parser, config, aliases={
        "train.seed":               ["-s",   "--seed"],
        "train.num_epochs":         ["-e",   "--num-epochs"],
        "dataset.train.name":       ["-td",  "--train-dataset"],
        "dataset.valid.name":       ["-vd",  "--valid-dataset"],
        "dataset.train.batch_size": ["-tbs", "--train-batch-size"],
        "dataset.valid.batch_size": ["-vbs", "--valid-batch-size"],
        "model.name":               ["-m",   "--model"],
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
    try:
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(log_name, latest_link)
    except:
        pass
    save_config(config, os.path.join(output_dir, "config.yaml"))

    logger = setup_logger(__name__, output_dir, suppress=[jax, flax, nxcl])
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
