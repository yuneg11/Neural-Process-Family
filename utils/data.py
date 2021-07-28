import torch
from torch.utils.data import IterableDataset


__all__ = [
    "NPFDataset",
    "CachedDataLoader",
]


class NPFDataset(IterableDataset):
    def __init__(self, data, device=None):
        self.data = data
        self.device = device
        self.batch_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx == len(self.data):
            raise StopIteration

        batch_data = self.data[self.batch_idx]
        x_context = batch_data["x_context"]
        y_context = batch_data["y_context"]
        x_target = batch_data["x_target"]
        y_target = batch_data["y_target"]

        if self.device:
            x_context = x_context.to(self.device)
            y_context = y_context.to(self.device)
            x_target  = x_target.to(self.device)
            y_target  = y_target.to(self.device)

        self.batch_idx += 1

        return x_context, y_context, x_target, y_target


class CachedDataLoader:
    def __init__(self, filepath, device="cpu", reuse=False):
        # self.data = torch.load(filepath, map_location=device)
        self.data = torch.load(filepath)
        self.device = device
        self.reuse = reuse
        self.epoch = 0

    def __iter__(self):
        if self.reuse:
            return NPFDataset(self.data[0])

        else:
            if self.epoch >= len(self.data):
                # raise RuntimeError("epoch is larger than the dataset")
                #! TODO: temporary fix
                self.epoch = 0


            epoch_data = self.data[self.epoch]
            self.epoch += 1

            return NPFDataset(epoch_data, device=self.device)
            # return NPFDataset(epoch_data)
