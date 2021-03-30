import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import dataset
import plot
from np import ConditionalNeuralProcessModel, NeuralProcessModel


def init(dataset_name, model_name):
    if dataset_name == "sine":
        train_loader = dataset.sine(train=True)
        test_loader = dataset.sine(train=False)

        check_val_every_n_epoch = 100

        if model_name == "cnp":
            model = ConditionalNeuralProcessModel(x_dim=1, y_dim=1, r_dim=128,
                                                  encoder_dims=[128, 128, 128],
                                                  decoder_dims=[128, 128])
        elif model_name == "np":
            model = NeuralProcessModel(x_dim=1, y_dim=1, r_dim=64, z_dim=64,
                                       deterministic_encoder_dims=[128, 128, 128],
                                       latent_encoder_dims=[128, 128, 128],
                                       decoder_dims=[128, 128])
        else:
            raise NameError(f"{model_name} is not supported")

        model.plotter = plot.plot_function

    elif dataset_name == "celeba":
        train_loader = dataset.celeba(train=True, root="~/data")
        test_loader = dataset.celeba(train=False, root="~/data")
        check_val_every_n_epoch = 1

        if model_name == "cnp":
            model = ConditionalNeuralProcessModel(x_dim=2, y_dim=3, r_dim=512,
                                                  encoder_dims=[512, 512, 512, 512, 512],
                                                  decoder_dims=[512, 512, 512])
        elif model_name == "np":
            model = NeuralProcessModel(x_dim=2, y_dim=3, r_dim=256, z_dim=256,
                                       deterministic_encoder_dims=[512, 512, 512, 512, 512],
                                       latent_encoder_dims=[512, 512, 512, 512, 512],
                                       decoder_dims=[512, 512, 512, 512])
        else:
            raise NameError(f"{model_name} is not supported")

        model.plotter = plot.plot_image

    else:
        raise NameError(f"{dataset_name} is not supported")

    return model, train_loader, test_loader, check_val_every_n_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset")
    parser.add_argument("-m", "--model")
    parser.add_argument("-g", "--gpu")
    args = parser.parse_args()

    model, train_loader, test_loader, check_val_every_n_epoch = init(args.dataset, args.model)

    logger = TensorBoardLogger(save_dir=f"./logs/{args.dataset}", name=args.model)
    trainer = pl.Trainer(gpus=[int(args.gpu)], logger=logger, check_val_every_n_epoch=check_val_every_n_epoch, max_epochs=30000)
    trainer.fit(model, train_loader, test_loader)
