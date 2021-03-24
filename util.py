import torch
from matplotlib import pyplot as plt


default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device(device=None):
    if device is None:
        return default_device
    else:
        if isinstance(device, str):
            return torch.device(device)
        else:
            return device


def plot_function(filename, context_x, context_y, target_x, target_y, mu, sigma, layout=(2, 2), figsize=None):
    nrows, ncols = layout

    if figsize is None:
        figsize = (3 * ncols, 3 * nrows)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=figsize)
    idxs = torch.randint(low=0, high=context_x.shape[0], size=(nrows, ncols))

    for row in range(nrows):
        for col in range(ncols):
            idx = idxs[row][col]

            context_x_sample = context_x[idx].cpu().detach()
            context_y_sample = context_y[idx].cpu().detach()

            target_x_sample = target_x[idx].cpu().detach()
            target_y_sample = target_y[idx].cpu().detach()

            mu_sample = mu[idx].cpu().detach()
            sigma_sample = sigma[idx].cpu().detach()

            xlim = target_x_sample.min(), target_x_sample.max()
            ylim = min(target_y_sample.min(), mu_sample.min()), max(target_y_sample.max(), mu_sample.max())

            ax[row][col].plot(context_x_sample, context_y_sample, "ko", markersize=10)
            ax[row][col].plot(target_x_sample, target_y_sample, "k:", linewidth=2)
            ax[row][col].plot(target_x_sample, mu_sample, "b", linewidth=2)
            ax[row][col].fill_between(target_x_sample[:, 0],
                                      mu_sample[:, 0] - sigma_sample[:, 0],
                                      mu_sample[:, 0] + sigma_sample[:, 0],
                                      alpha=0.2,
                                      facecolor='#65c9f7',
                                      interpolate=True)

            ax[row][col].set_xlim(xlim)
            ax[row][col].set_ylim(ylim)

    fig.tight_layout()

    fig.savefig(filename)

    plt.close(fig=fig)
    # return fig
