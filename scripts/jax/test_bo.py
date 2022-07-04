import sys
sys.path.append(".")

import os
import shutil
import logging
import random as pyrandom

import numpy as np

import jax
from jax import random
from jax import numpy as jnp

import flax
from flax.training import checkpoints

import nxcl
from nxcl.rich import Progress
from nxcl.config import load_config, save_config, add_config_arguments, ConfigDict
from nxcl.experimental.utils import get_experiment_name, setup_logger, link_output_dir

from npf.jax import functional as F
from npf.jax.models import (
    CNP, CANP,
    NP, ANP,
    BNP, BANP,
    NeuBNP, NeuBANP,
    ConvCNP,
)
from npf.jax.data import build_gp_prior_dataset

from bayeso.gp import gp_kernel
from bayeso.utils import utils_gp
from bayeso import covariance as bocov
from bayeso import acquisition as boacq


def link_output_dir(output_dir: str, subnames):
    link_dir = os.path.join("outs", *subnames, os.path.basename(output_dir))
    os.makedirs(os.path.dirname(link_dir), exist_ok=True)
    relpath = os.path.relpath(output_dir, os.path.dirname(link_dir))
    os.symlink(relpath, link_dir)


class Oracle:
    def __init__(self, str_cov="se"):
        self.str_cov = str_cov

    def __call__(self, rngs, x0_ctx, y0_ctx, x0_tar, mask0_ctx):
        x0_ctx, y0_ctx = x0_ctx[mask0_ctx], y0_ctx[mask0_ctx]

        cov_x_x, inv_cov_X_X, hyps = gp_kernel.get_optimized_kernel(
            x0_ctx, y0_ctx, prior_mu=None, str_cov=self.str_cov, fix_noise=False,
        )

        prior_mu_train = utils_gp.get_prior_mu(None, x0_ctx)
        prior_mu_test = utils_gp.get_prior_mu(None, x0_tar)
        cov_X_Xs = bocov.cov_main(self.str_cov, x0_ctx, x0_tar, hyps, False)
        cov_Xs_Xs = bocov.cov_main(self.str_cov, x0_tar, x0_tar, hyps, True)
        cov_Xs_Xs = (cov_Xs_Xs + cov_Xs_Xs.T) / 2.0

        mu_ = np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), y0_ctx - prior_mu_train) + prior_mu_test
        Sigma_ = cov_Xs_Xs - np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), cov_X_Xs)
        sigma_ = np.expand_dims(np.sqrt(np.maximum(np.diag(Sigma_), 0.0)), axis=1)

        return mu_, sigma_


def get_forward_step(model, state, **kwargs):
    @jax.jit
    def forward_step(rngs, x0_ctx, y0_ctx, x0_tar, mask0_ctx):
        mu, sigma = model.apply(
            state["params"],
            x_ctx=jnp.expand_dims(x0_ctx, axis=0),
            y_ctx=jnp.expand_dims(y0_ctx, axis=0),
            x_tar=jnp.expand_dims(x0_tar, axis=0),
            mask_ctx=jnp.expand_dims(mask0_ctx, axis=0),
            mask_tar=np.ones((1, mask0_ctx.shape[0])),
            rngs=rngs, **kwargs,
        )
        return mu[0], sigma[0]
    return forward_step


def main(config, output_dir):
    # Logging
    logger = logging.getLogger(__name__)

    # Random seed
    pyrandom.seed(config.test_bo.seed)
    os.environ["PYTHONHASHSEED"] = str(config.test_bo.seed)

    key = random.PRNGKey(config.test_bo.seed)

    # Create model
    models = dict(
        Oracle=Oracle,
        CNP=CNP,
        CANP=CANP,
        NP=NP,
        ANP=ANP,
        BNP=BNP,
        BANP=BANP,
        NeuBNP=NeuBNP,
        NeuBANP=NeuBANP,
        ConvCNP=ConvCNP,
    )

    if config.model.name not in models:
        raise ValueError(f"Unknown model: {config.model.name}")

    model = models[config.model.name](
        y_dim=config.datasets.shapes.y_ctx[-1],
        **config.model.get("kwargs", {}),
    )

    if config.model.name != "Oracle":
        state = checkpoints.restore_checkpoint(config.test_bo.checkpoint, target=None)

    # Create dataset
    test_key = random.PRNGKey(0)
    test_key, _ = random.split(test_key, 2)
    test_dataset = build_gp_prior_dataset(config.datasets.test_bo.gp, test_key)

    # Setup output directory
    link_output_dir(output_dir, subnames=(config.model.name, "Test", config.datasets.test_bo.name))

    # Copy checkpoint to output directory
    # shutil.copyfile(config.test_bo.checkpoint, os.path.join(output_dir, "ckpt"))
    os.link(config.test_bo.checkpoint, os.path.join(output_dir, "ckpt"))

    # Build step
    forward = get_forward_step(model, state, **config.model.get("test_kwargs", {}))

    with Progress() as p:
        regret_list = []
        regret_matrix = []

        for i, batch in enumerate(p.track(test_dataset, description=config.model.name)):
            key, sample_key = random.split(key)
            x0, y0, _, _, _, mask0_ctx, _, _, _ = batch

            min_ctx = F.masked_min(y0[:, 0], mask0_ctx)
            min_tar = jnp.min(y0[:, 0])
            min_list = [min_ctx]
            scale_factor = min_ctx - min_tar

            if scale_factor < 1e-4:
                continue

            for j in range(1, config.test_bo.num_steps + 1):
                mu, sigma = forward(dict(sample=sample_key), x0, y0, x0, mask0_ctx)

                if mu.ndim == 3 and sigma.ndim == 3:
                    # Law of total variance
                    var = np.mean(sigma ** 2, axis=0) + np.mean(mu ** 2, axis=0) - np.mean(mu, axis=0) ** 2
                    mu = np.ravel(np.mean(mu, axis=0))
                    sigma = np.ravel(np.sqrt(var))
                else:
                    mu = np.ravel(mu)
                    sigma = np.ravel(sigma)

                neg_acq_vals = boacq.ei(mu, sigma, y0)
                idx = np.argmax(neg_acq_vals).item()

                cur_val = y0[idx].item()
                if cur_val < min_ctx:
                    min_ctx = cur_val

                mask0_ctx = mask0_ctx.at[idx].set(True)
                min_list.append(min_ctx)

            regrets = (jnp.asarray(min_list) - min_tar) / scale_factor
            cum_regrets = np.cumsum(regrets)
            total_regret = cum_regrets[-1]
            regret_list.append(total_regret.item())
            regret_matrix.append(regrets)

            logger.debug(f"Try {i:3d}: Simple regrets:     [{[f'{v:.4f}' for v in regrets]}]")
            logger.debug(f"Try {i:3d}: Cumulative regrets: [{[f'{v:.4f}' for v in cum_regrets]}]")
            logger.info(f"Try {i:3d}: Total regret: {total_regret:.4f}")

    regret_matrix = np.stack(regret_matrix, axis=0)
    np.save(os.path.join(output_dir, "regrets.npy"), regret_matrix)

    logger.info(f"Regret: {np.mean(regret_list):.4f} ± {np.std(regret_list):.4f}")
    logger.info("Finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--train-dir", type=str)
    group.add_argument("-f", "--config-file", type=str)
    parser.add_argument("-c", "--checkpoint", "--test_bo.checkpoint", type=str, dest="checkpoint")
    parser.add_argument("-bf", "--bo-config-file", type=str)
    args, rest_args = parser.parse_known_args()

    if args.train_dir is not None:
        args.config_file = os.path.join(args.train_dir, "config.yaml")
        if args.checkpoint is None:
            args.checkpoint = checkpoints.latest_checkpoint(args.train_dir, "ckpt_best_ll_")
    else:
        if args.checkpoint is None:
            raise ValueError("Must specify checkpoint if --config-file is specified")

    rest_args.extend(["--test_bo.checkpoint", args.checkpoint])

    config: ConfigDict = load_config(args.config_file)
    if args.bo_config_file is not None:
        bo_config: ConfigDict = load_config(args.bo_config_file)
        config.update(bo_config.to_dict(flatten=True))

    parser = argparse.ArgumentParser(conflict_handler="resolve")
    add_config_arguments(parser, config, aliases={
        "test_bo.seed":         ["-s",   "--seed"],
        "dataset.test_bo.name": ["-td",  "--test-dataset"],
        "model.name":           ["-m",   "--model"],
    })
    parser.add_argument("-c", "--checkpoint", "--test_bo.checkpoint", type=str, dest="test_bo.checkpoint")
    args = parser.parse_args(rest_args)

    config.update(vars(args))
    config.setdefault("test_bo.seed", 42)
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

    # logger = setup_logger(__name__, output_dir, suppress=[jax, flax, nxcl])
    logger = setup_logger(__name__, output_dir, suppress=[])
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
