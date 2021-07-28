from matplotlib import pyplot as plt
import torch


def plot_function(context_x, context_y, target_x, target_y, mu, sigma, layout=(2, 5), figsize=None, filename=None):
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

            if sigma_sample.dim() == 2:
                sigma_sample = torch.sqrt(torch.diagonal(sigma_sample))

            context_x_idx = torch.argsort(context_x_sample.flatten())
            context_x_sample = context_x_sample[context_x_idx]
            context_y_sample = context_y_sample[context_x_idx]

            target_x_idx = torch.argsort(target_x_sample.flatten())
            target_x_sample = target_x_sample[target_x_idx]
            target_y_sample = target_y_sample[target_x_idx]

            mu_sample = mu_sample[target_x_idx]
            sigma_sample = sigma_sample[target_x_idx]

            xlim = target_x_sample.min(), target_x_sample.max()
            ylim = min(target_y_sample.min(), mu_sample.min()), max(target_y_sample.max(), mu_sample.max())

            ax[row][col].plot(context_x_sample, context_y_sample, "ko", markersize=5)
            ax[row][col].plot(target_x_sample, target_y_sample, "k:", linewidth=2)
            ax[row][col].plot(target_x_sample, mu_sample, "b", linewidth=2)
            ax[row][col].fill_between(target_x_sample[:, 0],
                                      mu_sample - sigma_sample,
                                      mu_sample + sigma_sample,
                                      alpha=0.2,
                                      facecolor='#65c9f7',
                                      interpolate=True)

            ax[row][col].set_xlim(xlim)
            ax[row][col].set_ylim(ylim)

    fig.tight_layout()
    plt.close(fig=fig)

    if filename is None:
        filename = "plot.png"

    return fig.savefig(filename)


def plot_from_dataset(gen, model, filename):
    task = next(iter(gen))

    mu, sigma = model(task['x_context'].cuda(), task['y_context'].cuda(), task['x_target'].cuda())
    mu, sigma = mu.cpu(), sigma.cpu()

    plot_function(task['x_context'], task['y_context'], task['x_target'], task['y_target'], mu, sigma, filename=filename)


# if __name__ == '__main__':
#     import stheno.torch as stheno

#     from neural_process.utils import data
#     from neural_process import models

#     kernel = stheno.EQ().stretch(0.25)
#     gen = data.GPGenerator(kernel=kernel, num_tasks=1)
#     model = models.GNP().cuda()

#     plot_from_dataset(gen, model, "plot.png")
