from pipetorch.experiment import Experiment
import os
import torch
from torch import nn
from torch.utils import data
import torchvision
import torch.optim as optim
from utils.helper_functions import bw2rgb_expand_channels, resize_dataset, dataset2split_loaders
import argparse
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt


def get_classifier(model_name=None, checkpoint_path=None):
    classifier = nn.Sequential(torchvision.models.mobilenet_v3_large(False), nn.Linear(1000, 10))
    filename = 'model.pth'
    if not model_name is None:
        filename = 'model_{}.pth'.format(model_name)
    if not checkpoint_path is None:
        classifier.load_state_dict(torch.load(os.path.join(checkpoint_path, filename)))
    return classifier


def loss_calc_func(models, data, target, loss_functions, data_preprocess_function):
    if data_preprocess_function is not None:
        data = data_preprocess_function(data)
    output = models["classifier"](data)
    loss = loss_functions["CrossEntropy"](output, target)
    return loss, output


def train_classifier(args):
    model_name = "classifier"
    models = {model_name: get_classifier(model_name=model_name, checkpoint_path=args.ckpt_path)}
    # models = {model_name: get_classifier(model_name=model_name, checkpoint_path=None)}
    optimizer = optim.Adam(models["classifier"].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # Weight decay scheduler? increase weight_decay when loss goes down
    # Dropout scheduler? increase dropout rate when loss goes down
    entire_train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              (0.1307,), (0.3081,))
                                                          ]))
    name = "full_classifier"
    if args.watermark:
        entire_train_dataset = resize_dataset(entire_train_dataset, dim=(7,7))
        name = "small_classifier"
    split = [0.6, 0.4]
    data_loaders, split = dataset2split_loaders(entire_train_dataset, args.batch_size, split)
    train_loader = data_loaders[0]
    validation_loader = data_loaders[1]
    """
    test_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                                  ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    """
    data_loaders = {"train": train_loader, "eval": validation_loader}
    path = args.experiments_path
    experiment = Experiment(name, models,
                            data_loaders,
                            {"CrossEntropy": torch.nn.CrossEntropyLoss()}, loss_calc_func,
                            optimizer, path, scheduler=scheduler,
                            data_preprocess_function=bw2rgb_expand_channels)
    mean, std, entropy_mean, entropy_std, losses_mean, losses_std = experiment.dataset_calc(data_loaders["train"],n_labels=10)
    entropy_mean_np = entropy_mean.numpy()
    entropy_std_np = entropy_std.numpy()
    losses_mean_np = losses_mean.numpy()
    losses_std_np = losses_std.numpy()
    mean_np = mean.numpy()
    mean_np = mean.numpy()
    std_np = std.numpy()

    n_bins = 50
    bins = np.linspace(entropy_mean_np.min(), entropy_mean_np.max(), n_bins)
    idx = np.digitize(entropy_mean_np, bins=bins)
    loss_mean = []
    loss_std = []
    entropy_axis = []
    loss_min = []
    for i in range(n_bins):
        if len(losses_mean_np[idx == i]) > 1:
            loss_mean.append(losses_mean_np[idx == i].mean())
            loss_std.append(2 * losses_mean_np[idx == i].std())
            loss_min.append(losses_mean_np[idx == i].min())
            # entropy_axis.append((bins[i-1] + bins[i])/2)
            entropy_axis.append(bins[i])
    loss_std = np.array(loss_std)
    loss_mean = np.array(loss_mean)
    loss_min = np.array(loss_min)
    lower_error = loss_std + np.clip(loss_mean - loss_std - loss_min, a_min=-np.inf, a_max=0)
    upper_error = loss_std
    bins_axis = entropy_mean_np
    y_axis = losses_mean_np
    mean_statistics = binned_statistic(bins_axis, y_axis, statistic='mean', bins=50)
    std_statistics = binned_statistic(bins_axis, y_axis, statistic='std', bins=50)
    plt.errorbar((mean_statistics.bin_edges[:-1] + mean_statistics.bin_edges[1:]) / 2, mean_statistics.statistic, yerr=std_statistics.statistic, linestyle='None', marker='o', alpha=0.5, zorder=3)
    plt.plot(bins_axis, y_axis, 'o', markersize=4, markeredgecolor=None, markeredgewidth=0, alpha=0.2)
    # plt.errorbar(np.array(entropy_axis), loss_mean, yerr=[lower_error, upper_error], linestyle='None', marker='o', alpha=0.8, zorder=3)
    plt.ylabel('Cross-Entropy Loss')
    plt.xlabel('Entropy')
    # plt.plot(entropy_mean.numpy(), losses_mean.numpy(), 'bo', markersize=4, markeredgecolor=None, markeredgewidth=0, alpha=0.3)
    # plt.errorbar(entropy_mean.numpy(), losses_mean.numpy(), xerr=entropy_std.numpy(), yerr=losses_std.numpy(), color='orange', markersize=0, linestyle='None', marker='o', alpha=0.2, zorder=3)
    plt.show()
    # experiment.run(delta_epochs_to_save_checkpoint=10)