import torch
from torch.utils.data import Dataset

from matplotlib import pyplot as plt

PI = 3.1415926535


def get_context_and_target(x_data, y_data, num_context_range, num_target_range=None, device="cpu"):
    num_points = x_data.shape[1]
    random_idx = torch.randperm(num_points)

    num_context = torch.randint(low=num_context_range[0], high=num_context_range[1], size=(1,))[0]
    context_idx = random_idx[:num_context]

    context_x = x_data.index_select(dim=1, index=context_idx)
    context_y = y_data.index_select(dim=1, index=context_idx)

    if num_target_range:
        num_target = torch.randint(low=num_target_range[0], high=num_target_range[1], size=(1,))[0]
        target_idx = random_idx[:num_context + num_target] # target always includes context

        target_x = x_data.index_select(dim=1, index=target_idx)
        target_y = y_data.index_select(dim=1, index=target_idx)
    else:
        target_x = x_data
        target_y = y_data

    return ((context_x.to(device), context_y.to(device)),
            (target_x.to(device), target_y.to(device)))


class CosineDataset(Dataset):
    def __init__(self,
                 x_shift_range = (-2., +2.),
                 x_scale_range = (+0.5, +1.5),
                 y_shift_range = (-1., +1.),
                 y_scale_range = (-1., +1.),
                 num_samples = 1000,
                 num_points = 100,
                 train = False):
        self._x_shift_range = x_shift_range
        self._y_shift_range = y_shift_range
        self._y_scale_range = y_scale_range
        self._num_samples = num_samples
        self._num_points = num_points
        self._train = train

        with torch.no_grad():
            x_shift_vec = torch.rand(num_samples, 1) * (x_shift_range[1] -  x_shift_range[0]) + x_shift_range[0]
            x_scale_vec = torch.rand(num_samples, 1) * (x_scale_range[1] -  x_scale_range[0]) + x_scale_range[0]
            y_shift_vec = torch.rand(num_samples, 1) * (y_shift_range[1] -  y_shift_range[0]) + y_shift_range[0]
            y_scale_vec = torch.rand(num_samples, 1) * (y_scale_range[1] -  y_scale_range[0]) + y_scale_range[0]

            if train:
                x_mat = torch.rand(num_samples, num_points) * (2 * PI) - PI
            else:
                x_mat = torch.linspace(start=-PI, end=+PI, steps=num_points).repeat(num_samples, 1)

            y_mat = y_scale_vec * torch.cos((x_mat - x_shift_vec) * x_scale_vec) - y_shift_vec

            self._x_mat = x_mat.unsqueeze(dim=-1)
            self._y_mat = y_mat.unsqueeze(dim=-1)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, index):
        return self._x_mat[index], self._y_mat[index]

    # def plot(self, index=None):
    #     if index:
    #         x_mat = self._x_mat[index].unsqueeze(dim=0)
    #         y_mat = self._y_mat[index].unsqueeze(dim=0)
    #     else:
    #         x_mat = self._x_mat
    #         y_mat = self._y_mat

    #     fig, ax = plt.subplots(ncols=1, nrows=1)

    #     xlim = x_mat.min(), x_mat.max()
    #     ylim = y_mat.min(), y_mat.max()

    #     order = x_mat.argsort(dim=1)
    #     x_mat = x_mat.gather(index=order, dim=1)
    #     y_mat = y_mat.gather(index=order, dim=1)

    #     for x_vec, y_vec in zip(x_mat, y_mat):
    #         ax.scatter(x=x_vec, y=y_vec, s=2)
    #         ax.plot(x_vec, y_vec)

    #     ax.set_xlim(xlim)
    #     ax.set_ylim(ylim)

    #     return fig


class CurveDataset(Dataset):
    def __init__(self,
                 l1_scale: float = 0.4,
                 sigma_scale: float = 1.0,
                 num_samples = 1000,
                 num_points = 100,
                 train = False):
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._num_samples = num_samples
        self._num_points = num_points
        self._train = train
        self._x_size = 1
        self._y_size = 1

        with torch.no_grad():
            if train:
                x_mat = torch.rand(num_samples, num_points) * 4. - 2.
            else:
                x_mat = torch.linspace(start=-2., end=+2., steps=num_points).repeat(num_samples, 1)

            x_mat = x_mat.unsqueeze(dim=-1)

            l1 = torch.ones(num_samples, self._y_size, self._x_size) * self._l1_scale
            sigma_f = torch.ones(num_samples, self._y_size) * self._sigma_scale

            kernel = self._gaussian_kernel(x_mat, l1, sigma_f)

            cholesky = torch.cholesky(kernel)
            y_mat = torch.matmul(cholesky, torch.rand(num_samples, self._y_size, num_points, 1))
            y_mat = torch.squeeze(y_mat, dim=3).permute(0, 2, 1)

            self._x_mat = x_mat
            self._y_mat = y_mat

    def _gaussian_kernel(self,
                         xdata: torch.Tensor,
                         l1: torch.Tensor,
                         sigma_f: torch.Tensor,
                         sigma_noise: float = 2e-2):
        """
        Args:
            xdata: [batch_size, num_total_points, x_size]
            l1: [batch_size, y_size, x_size]
            sigma_f: [batch_size, y_size]
            sigma_noise: float

        Return:
            kernel: [batch_size, y_size, num_total_points, num_total_points]
        """
        num_total_points = xdata.shape[1]

        xdata1 = torch.unsqueeze(xdata, dim=1)
        xdata2 = torch.unsqueeze(xdata, dim=2)
        diff = xdata1 - xdata2

        norm = torch.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])
        norm = torch.sum(norm, dim=-1)

        kernel = torch.square(sigma_f)[:, :, None, None] * torch.exp(-0.5 * norm)
        kernel += (sigma_noise ** 2) * torch.eye(num_total_points)

        return kernel

    def __len__(self):
        return self._num_samples

    def __getitem__(self, index):
        return self._x_mat[index], self._y_mat[index]
