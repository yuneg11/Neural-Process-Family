from npf import models


__all__ = [
    "get_model",
]


def get_model(model_name, **model_kwargs):
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
            loss_type=model_kwargs["loss_type"],
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
            loss_type=model_kwargs["loss_type"],
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
            loss_type=model_kwargs["loss_type"],
        )
    elif model_name == "convnpxl":
        model = models.ConvNP(
            y_dim=1, z_dim=8,
            determ_cnn_xl=True,
            latent_cnn_xl=True,
            loss_type=model_kwargs["loss_type"],
        )
    elif model_name == "gnp":
        model = models.GNP(
            y_dim=1,
            mean_cnn_xl=False,
            kernel_cnn_xl=False,
            likelihood_type=model_kwargs["likelihood_type"],
            loss_type=model_kwargs["loss_type"],
        )
    elif model_name == "gnpxl":
        model = models.GNP(
            y_dim=1,
            mean_cnn_xl=True,
            kernel_cnn_xl=True,
            likelihood_type=model_kwargs["likelihood_type"],
            loss_type=model_kwargs["loss_type"],
        )
    else:
        raise ValueError(f"Unsupported model: '{model_name}'")

    return model
