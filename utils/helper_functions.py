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
    assert np.array(split).sum() == 1, "split sum must be equal to 1."
    split = [int(np.round(i * len(dataset))) for i in split]
    splitted_datasets = data.random_split(dataset, split)
    split_dataloader = [data.DataLoader(splitted_dataset, batch_size=batch_size, shuffle=True) for splitted_dataset in splitted_datasets]
    return split_dataloader, split


if __name__ == '__main__':
    entire_train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              (0.1307,), (0.3081,))
                                                          ]))
    dataset = resize_dataset_and_pad(entire_train_dataset, small_dim=(7, 7), large_dim=(28, 28))
    embedded_data = embed(entire_train_dataset.data, dataset.data)