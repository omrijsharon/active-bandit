import numpy as np
import cv2
import torch
import torchvision
from torch.utils import data
from copy import deepcopy
from tqdm import tqdm


def resize_dataset(dataset, dim):
    new_dataset = deepcopy(dataset)
    N = len(dataset.data)
    if len(dataset.data.size()) == 3:
        new_data = torch.zeros(N, *dim)
        data = dataset.data
    elif len(dataset.data.size()) == 4:
        new_data = torch.zeros(N, dataset.data.size(1), *dim)
        data = dataset.data.permute(1, 2, 0)
    else:
        raise ValueError("Expected input to have 3 or 4 dimension, but got {} dimentions.".format(len(dataset.data[0].size())))
    for i in tqdm(range(N)):
        new_data[i] = torch.tensor(cv2.resize(data[i].numpy(), dim, interpolation=cv2.INTER_CUBIC))
    new_dataset.data = new_data
    return new_dataset


def resize_dataset_and_pad(dataset, small_dim: tuple, large_dim: tuple):
    new_dataset = deepcopy(dataset)
    N = len(dataset.data)
    h = int(large_dim[0] / 2) - int(small_dim[0] / 2)
    w = int(large_dim[1] / 2) - int(small_dim[1] / 2)
    if len(dataset.data.size()) == 3:
        new_data = torch.zeros(N, *large_dim)
        data = new_dataset.data
        for i in tqdm(range(N)):
            small_img = torch.tensor(cv2.resize(data[i].numpy(), small_dim, interpolation=cv2.INTER_CUBIC))
            new_data[i, h:h + small_dim[0], w:w + small_dim[1]] = small_img
    elif len(dataset.data.size()) == 4:
        new_data = torch.zeros(N, dataset.data.size(1), *large_dim)
        data = new_dataset.data.permute(1, 2, 0)
        for i in tqdm(range(N)):
            small_img = torch.tensor(cv2.resize(data[i].numpy(), small_dim, interpolation=cv2.INTER_CUBIC))
            new_data[i, :, h:h + small_dim[0], w:w + small_dim[1]] = small_img
    else:
        raise ValueError("Expected input to have 3 or 4 dimension, but got {} dimentions.".format(len(dataset.data[0].size())))

    new_dataset.data = new_data
    return new_dataset


def bw2rgb_expand_channels(data):
    return data.expand(data.size(0), 3, data.size(2), data.size(3))


def embed(x, y):
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    return torch.cat((x, y), dim=1)


class SoftABS(torch.nn.Module):
    def __init__(self):
        super(SoftABS, self).__init__()

    def forward(self, output, target):
        return torch.log(torch.cosh(output - target)).mean()


def dataset2split_loaders(dataset, batch_size, split=None):
    if split is None:
        split = [0.8, 0.2]
    if np.array(split).sum() == 1:
        split = [int(np.round(i * len(dataset))) for i in split]
    elif np.array(split).dtype == np.dtype('int32') and np.array(split).sum() == len(entire_train_dataset):
        pass
    else:
        raise ValueError("split must be a list that sums to 1 or a list of integers that sums up to the length of the dataset.")

    split_datasets = data.random_split(dataset, split)
    split_dataloaders = [data.DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in split_datasets]
    return split_dataloaders, split_datasets, split


class Oracle(object):
    def __init__(self, entire_dataset):
        self.remaining_dataset = entire_dataset
        self.partial_dataset = None

    def random_query(self, n_queries):
        assert n_queries < len(self.remaining_dataset), "n_queries is larger than the remaining dataset."
        new_dataset, self.remaining_dataset = data.random_split(self.remaining_dataset, [n_queries, len(self.remaining_dataset)-n_queries])
        self.__concat_to_partial_dataset(new_dataset)
        return self.partial_dataset

    def query(self, indices):
        new_data = torch.utils.data.Subset(self.remaining_dataset, indices)
        mask = np.ones(len(self.remaining_dataset), dtype=bool)
        mask[indices] = False
        self.remaining_dataset = data.Subset(self.remaining_dataset, *mask.nonzero())
        self.__concat_to_partial_dataset(new_data)
        return self.partial_dataset

    @staticmethod
    def to_dataloader(dataset, batch_size):
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def __concat_to_partial_dataset(self, new_dataset):
        if self.partial_dataset is None:
            self.partial_dataset = new_dataset
        else:
            self.partial_dataset = torch.utils.data.ConcatDataset((self.partial_dataset, new_dataset))

    def __repr__(self):
        return "Oracle status @ remaining_dataset: {},    partial_dataset: {}".format(len(self.remaining_dataset), len(self.partial_dataset))


def weighted_loss(n_classes, target, loss_each, reduction='mean'):
    classes, counts = torch.unique(target, return_counts=True)
    assert len(classes) == n_classes, "there are {} classes in the target tensor but {} classes in the dataset.".format(len(classes), n_classes)
    classes_weights = counts.sum()/(n_classes * counts)
    for c, w in zip(classes, classes_weights):
        loss_each[target == c] = loss_each[target==c] * w
    if reduction.lower() == "mean":
        return loss_each.mean()
    elif reduction.lower() == "none":
        return loss_each
    elif reduction.lower() == "sum":
        return loss_each.sum()
    else:
        raise ValueError("reduction {} is not supported. Only mean, sum and none".format(reduction))


if __name__ == '__main__':
    entire_train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              (0.1307,), (0.3081,))
                                                          ]))
    oracle = Oracle(entire_train_dataset)
    oracle.random_query(100)
    # dataset = resize_dataset_and_pad(entire_train_dataset, small_dim=(7, 7), large_dim=(28, 28))
    # embedded_data = embed(entire_train_dataset.data, dataset.data)