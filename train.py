import argparse

from torch import optim

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from neural_processes.models import (
    ConditionalNeuralProcess,
    AttentiveConditionalNeuralProcess,
    NeuralProcess,
    AttentiveNeuralProcess,
)

import plot
import dataset


class LightningBase(pl.LightningModule):
    def training_step(self, batch, _):
        (x_context, y_context, x_target), y_target = batch

        loss = self.loss(x_context, y_context, x_target, y_target)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        (x_context, y_context, x_target), y_target = batch

        if batch_idx == 0:
            mu, sigma = self(x_context, y_context, x_target)

            img = self.plotter(x_context, y_context, x_target, y_target, mu, sigma)
            self.logger.experiment.add_image("test_images", img, self.current_epoch + 1)

        predictive_ll = self.predictive_log_likelihood(x_context, y_context, x_target, y_target)
        self.log('predictive_ll', predictive_ll)

        # if hasattr(self, "joint_log_likelihood"):
        #     joint_ll = self.joint_log_likelihood(x_target, y_target)
        #     self.log('joint_ll', joint_ll)

    def configure_optimizers(self):
        return self.optimizer

    def set_plotter(self, plotter):
        self.plotter = plotter

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


class CNP(ConditionalNeuralProcess, LightningBase): pass
class ACNP(AttentiveConditionalNeuralProcess, LightningBase): pass
class NP(NeuralProcess, LightningBase): pass
class ANP(AttentiveNeuralProcess, LightningBase): pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset")
    parser.add_argument("-m", "--model")
    parser.add_argument("-g", "--gpu")
    args = parser.parse_args()

    if args.dataset == "sine":
        train_loader = dataset.sine(train=True)
        test_loader = dataset.sine(train=False)
        check_val_every_n_epoch = 100

        models = {
            "cnp": CNP(
                x_dim=1, y_dim=1, r_dim=128,
                encoder_dims=[128, 128, 128],
                decoder_dims=[128, 128, 128],
            ),
            "acnp": ACNP(
                x_dim=1, y_dim=1, r_dim=128,
                encoder_dims=[128, 128, 128],
                decoder_dims=[128, 128, 128],
                attention_dims=[32],
                attention="dot_product",
            ),
            "np": NP(
                x_dim=1, y_dim=1, r_dim=64, s_dim=128, z_dim=64,
                deterministic_dims=[128, 128, 128],
                latent_dims=[128, 128, 128],
                sampler_dims=[96],
                decoder_dims=[128, 128, 128],
            ),
            "anp": ANP(
                x_dim=1, y_dim=1, r_dim=64, s_dim=128, z_dim=64,
                deterministic_dims=[128, 128, 128],
                latent_dims=[128, 128, 128],
                sampler_dims=[96],
                decoder_dims=[128, 128, 128],
                attention_dims=[32],
                attention="dot_product",
            )
        }

        plotter = plot.plot_function

    elif args.dataset == "celeba":
        train_loader = dataset.celeba(train=True, root="~/data")
        test_loader = dataset.celeba(train=False, root="~/data")
        check_val_every_n_epoch = 1

        models = {
            "cnp": CNP(
                x_dim=2, y_dim=3, r_dim=512,
                encoder_dims=[512, 512, 512, 512, 512],
                decoder_dims=[512, 512, 512],
            ),
            "acnp": ACNP(
                x_dim=2, y_dim=3, r_dim=512,
                encoder_dims=[512, 512, 512, 512, 512],
                decoder_dims=[512, 512, 512],
                attention_dims=[128],
                attention="dot_product",
            ),
            "np": NP(
                x_dim=2, y_dim=3, r_dim=256, s_dim=512, z_dim=256,
                deterministic_dims=[512, 512, 512, 512, 512],
                latent_dims=[512, 512, 512, 512, 512],
                sampler_dims=[384],
                decoder_dims=[512, 512, 512],
            ),
            "anp": ANP(
                x_dim=2, y_dim=3, r_dim=64, s_dim=128, z_dim=64,
                deterministic_dims=[512, 512, 512, 512, 512],
                latent_dims=[512, 512, 512, 512, 512],
                sampler_dims=[384],
                decoder_dims=[512, 512, 512],
                attention_dims=[128],
                attention="dot_product",
            )
        }

        plotter = plot.plot_image

    else:
        raise ValueError


    model = models[args.model]
    model.set_plotter(plotter)
    model.set_optimizer(optim.Adam(model.parameters(), lr=5e-5))

    # DEBUG
    check_val_every_n_epoch = 10

    logger = TensorBoardLogger(save_dir=f"logs/{args.dataset}", name=args.model)
    trainer = pl.Trainer(gpus=[int(args.gpu)], logger=logger, check_val_every_n_epoch=check_val_every_n_epoch, max_epochs=int(1e+5))
    trainer.fit(model, train_loader, test_loader)
