import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.datasets import CelebA

PI = 3.1415926535


def get_collate_fn(num_context_range, num_target_range=None):
    def get_context_and_target(batch):
        x_data = torch.stack([x for x, _ in batch], dim=0)
        y_data = torch.stack([y for _, y in batch], dim=0)

        num_points = x_data.shape[1]
        random_idx = torch.randperm(num_points)

        num_context = torch.randint(low=num_context_range[0], high=num_context_range[1], size=(1,))[0]
        context_idx = random_idx[:num_context]

        context_x = x_data.index_select(dim=1, index=context_idx)
        context_y = y_data.index_select(dim=1, index=context_idx)

        if num_target_range:
            num_target = torch.randint(low=num_target_range[0], high=num_target_range[1], size=(1,))[0]
            target_idx = random_idx[:num_context + num_target]  # target always includes context

            target_x = x_data.index_select(dim=1, index=target_idx)
            target_y = y_data.index_select(dim=1, index=target_idx)
        else:
            target_x = x_data
            target_y = y_data

        return ((context_x, context_y, target_x), target_y)
    return get_context_and_target


class SineDataset(Dataset):
    def __init__(self,
                 x_shift_range=(-2., +2.),
                 x_scale_range=(+0.5, +1.5),
                 y_shift_range=(-1., +1.),
                 y_scale_range=(-1., +1.),
                 num_samples=2048,
                 num_points=128,
                 train=False):
        self._x_shift_range = x_shift_range
        self._y_shift_range = y_shift_range
        self._y_scale_range = y_scale_range
        self._num_samples = num_samples
        self._num_points = num_points
        self._train = train

        with torch.no_grad():
            x_shift_vec = torch.rand(num_samples, 1) * (x_shift_range[1] - x_shift_range[0]) + x_shift_range[0]
            x_scale_vec = torch.rand(num_samples, 1) * (x_scale_range[1] - x_scale_range[0]) + x_scale_range[0]
            y_shift_vec = torch.rand(num_samples, 1) * (y_shift_range[1] - y_shift_range[0]) + y_shift_range[0]
            y_scale_vec = torch.rand(num_samples, 1) * (y_scale_range[1] - y_scale_range[0]) + y_scale_range[0]

            if train:
                x_mat = torch.rand(num_samples, num_points) * (2 * PI) - PI
            else:
                x_mat = torch.linspace(start=-PI, end=+PI, steps=num_points).repeat(num_samples, 1)

            y_mat = y_scale_vec * x_mat * torch.sin((x_mat - x_shift_vec) * x_scale_vec) - y_shift_vec

            self._x_mat = x_mat.unsqueeze(dim=-1)
            self._y_mat = y_mat.unsqueeze(dim=-1)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, index):
        return self._x_mat[index], self._y_mat[index]


class CurveDataset(Dataset):
    def __init__(self,
                 l1_scale: float = 0.4,
                 sigma_scale: float = 1.0,
                 num_samples=1000,
                 num_points=100,
                 train=False):
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


class CelebADataset(CelebA):
    def __init__(self, size=32, **kwargs):
        super().__init__(**kwargs)

        loc = torch.stack((torch.linspace(start=0., end=1., steps=size).unsqueeze(dim=1).repeat(1, size),
                           torch.linspace(start=0., end=1., steps=size).unsqueeze(dim=0).repeat(size, 1)), dim=2)

        self._loc = loc.reshape(size * size, 2)

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)

        # h == w
        h, w = img.shape[1], img.shape[2]
        if h != w or h * w != self._loc.shape[0]:
            raise ValueError("Given image size is not match with predefined size")

        x = self._loc
        y = img.permute(1, 2, 0).reshape(h * w, -1)

        return x, y


def sine(batch_size=1024,
         num_context_range=(5, 10),
         num_target_range=(5, 10),
         train=False,
         dataset_kwargs={},
         dataloader_kwargs={}):
    if train:
        collate_fn = get_collate_fn(num_context_range, num_target_range)
    else:
        collate_fn = get_collate_fn(num_context_range, None)

    dataset = SineDataset(train=train, **dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train,
                            collate_fn=collate_fn, num_workers=4, **dataloader_kwargs)

    return dataloader


def celeba(batch_size=128,
           num_context_range=(30, 100),
           num_target_range=(100, 300),
           train=False,
           root="./data/",
           size=32,
           crop=150,
           dataset_kwargs={},
           dataloader_kwargs={}):
    if train:
        collate_fn = get_collate_fn(num_context_range, num_target_range)
    else:
        collate_fn = get_collate_fn(num_context_range, None)

    transform = transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    dataset = CelebADataset(root=root, split=("train" if train else "test"),
                            transform=transform, size=size, **dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train,
                            collate_fn=collate_fn, num_workers=4, **dataloader_kwargs)

    return dataloader


datasets = {
    "sine": sine,
    "celeba": celeba,
}


def get_dataset(name, batch_size, train=False, **kwargs):
    if name in datasets:
        return datasets[name](batch_size=batch_size, train=train, **kwargs)
    else:
        return NameError(f"Dataset '{name}' is not supported")
