import sys
sys.path.append(".")

import os
import shutil
import logging
import random as pyrandom
from functools import partial

import jax
from jax import random
from jax import numpy as jnp

import flax
from flax import jax_utils
from flax.training import checkpoints

import nxcl
from nxcl.rich import Progress
from nxcl.config import load_config, save_config, add_config_arguments, ConfigDict
from nxcl.experimental.utils import get_experiment_name, setup_logger, link_output_dir, AverageMeter

from npf.jax.models import (
    CNP, CANP,
    NP, ANP,
    BNP, BANP,
    NeuBNP, NeuBANP,
    ConvCNP, ConvNP,
)
from npf.jax.data import get_shard_collate, build_dataloader


def link_output_dir(output_dir: str, subnames):
    link_dir = os.path.join("outs", *subnames, os.path.basename(output_dir))
    os.makedirs(os.path.dirname(link_dir), exist_ok=True)
    relpath = os.path.relpath(output_dir, os.path.dirname(link_dir))
    os.symlink(relpath, link_dir)


@jax.jit
def sync_metric(metric):
    return jax.tree_util.tree_map(lambda x: jnp.mean(x), metric)


def get_test_step(model, **kwargs):
    @partial(jax.pmap, axis_name="batch")
    def _test_step(state, rngs, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar):
        ll = model.apply(
            state["params"], x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar,
            method=model.log_likelihood, rngs=rngs, **kwargs,
        )
        return ll

    def test_step(state, rngs, *, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar):
        metric = _test_step(state, rngs, x_ctx, y_ctx, x_tar, y_tar, mask_ctx, mask_tar)
        return sync_metric(metric)

    return test_step


def main(config, output_dir):
    num_devices = jax.local_device_count()

    # Logging
    logger = logging.getLogger(__name__)
    logger.info(f"Number of devices: {num_devices}")

    # Random seed
    pyrandom.seed(config.test.seed)
    os.environ["PYTHONHASHSEED"] = str(config.test.seed)

    key = random.PRNGKey(config.test.seed)

    # Create model
    models = dict(
        CNP=CNP,
        CANP=CANP,
        NP=NP,
        ANP=ANP,
        BNP=BNP,
        BANP=BANP,
        NeuBNP=NeuBNP,
        NeuBANP=NeuBANP,
        ConvCNP=ConvCNP,
        ConvNP=ConvNP,
    )

    if config.model.name not in models:
        raise ValueError(f"Unknown model: {config.model.name}")

    model = models[config.model.name](
        y_dim=config.datasets.shapes.y_ctx[-1],
        **config.model.get("kwargs", {}),
    )

    state = checkpoints.restore_checkpoint(config.test.checkpoint, target=None)
    state = jax_utils.replicate(state)

    # Create dataset
    test_key = random.PRNGKey(43)
    shard_collate = get_shard_collate(num_replicas=num_devices, jit=True)
    test_loader = build_dataloader(config.datasets.test, test_key, shard_collate)

    # Setup output directory
    link_output_dir(output_dir, subnames=(config.model.name, "Test", config.datasets.test.name))

    # Copy checkpoint to output directory
    # shutil.copyfile(config.test.checkpoint, os.path.join(output_dir, "ckpt"))
    os.link(config.test.checkpoint, os.path.join(output_dir, "ckpt"))

    # Build steps
    test_step = get_test_step(model, **config.model.get("test_kwargs", {}))

    # Test
    test_meter = AverageMeter("ll_ctx", "ll_tar", "ll")

    if test_loader.is_map_dataset:
        num_step_per_epoch = config.test.get("num_step_per_epoch", len(test_loader))
        iter_loader = (v for v in test_loader)
    else:
        num_step_per_epoch = config.test.num_step_per_epoch
        test_iterator = iter(test_loader)
        iter_loader = (next(test_iterator) for _ in range(num_step_per_epoch))

    with Progress() as p:
        for batch in p.track(iter_loader, description="Test", total=num_step_per_epoch):
            key, model_key = random.split(key)
            replicated_rngs = jax_utils.replicate(dict(sample=model_key))

            ll_ctx = test_step(
                state, replicated_rngs,
                x_ctx=batch.x_ctx, y_ctx=batch.y_ctx, mask_ctx=batch.mask_ctx,
                x_tar=batch.x_ctx, y_tar=batch.y_ctx, mask_tar=batch.mask_ctx,
            )
            ll_tar = test_step(
                state, replicated_rngs,
                x_ctx=batch.x_ctx, y_ctx=batch.y_ctx, mask_ctx=batch.mask_ctx,
                x_tar=batch.x_tar, y_tar=batch.y_tar, mask_tar=batch.mask_tar,
            )
            ll = test_step(
                state, replicated_rngs,
                x_ctx=batch.x_ctx, y_ctx=batch.y_ctx, mask_ctx=batch.mask_ctx,
                x_tar=batch.x,     y_tar=batch.y,     mask_tar=batch.mask,
            )

            test_meter.update(ll_ctx=ll_ctx, ll_tar=ll_tar, ll=ll, n=len(batch.x))

        ll_ctx, ll_tar, ll = test_meter.ll_ctx, test_meter.ll_tar, test_meter.ll
        logger.info(f"CTX LL: {ll_ctx:8.4f}   TAR LL: {ll_tar:8.4f}   LL: {ll:8.4f}")

    logger.info("Finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--train-dir", type=str)
    group.add_argument("-f", "--config-file", type=str)
    parser.add_argument("-c", "--checkpoint", "--test.checkpoint", type=str, dest="checkpoint")
    parser.add_argument("-tf", "--test-config-file", type=str)
    args, rest_args = parser.parse_known_args()

    if args.train_dir is not None:
        args.config_file = os.path.join(args.train_dir, "config.yaml")
        if args.checkpoint is None:
            args.checkpoint = checkpoints.latest_checkpoint(args.train_dir, "ckpt_best_ll_")
    else:
        if args.checkpoint is None:
            raise ValueError("Must specify checkpoint if --config-file is specified")

    rest_args.extend(["--test.checkpoint", args.checkpoint])

    config: ConfigDict = load_config(args.config_file)
    if args.test_config_file is not None:
        test_config: ConfigDict = load_config(args.test_config_file)
        config.update(test_config.to_dict(flatten=True))

    parser = argparse.ArgumentParser(conflict_handler="resolve")
    add_config_arguments(parser, config, aliases={
        "test.seed":                ["-s",   "--seed"],
        "dataset.test.name":        ["-td",  "--test-dataset"],
        "dataset.test.batch_size":  ["-tbs", "--test-batch-size"],
        "model.name":               ["-m",   "--model"],
    })
    parser.add_argument("-c", "--checkpoint", "--test.checkpoint", type=str, dest="test.checkpoint")
    args = parser.parse_args(rest_args)

    config.update(vars(args))
    config.setdefault("test.seed", 42)
    # config.lock()

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
