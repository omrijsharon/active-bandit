from pipetorch.experiment import Experiment
import os
import torch
from torch import nn
from torch.utils import data
import torchvision
import torch.optim as optim
from utils.helper_functions import bw2rgb_expand_channels, resize_dataset
import argparse


def get_classifier(model_name=None, checkpoint_path=None):
    classifier = nn.Sequential(torchvision.models.mobilenet_v3_large(False), nn.Linear(1000, 10))
    filename = 'model.pth'
    if not model_name:
        filename = 'model_{}.pth'.format(model_name)
    if not checkpoint_path:
        classifier.load_state_dict(torch.load(os.path.join(checkpoint_path, filename)))
    return classifier


def loss_calc_func(models, data, target, loss_functions, data_preprocess_function):
    if data_preprocess_function is not None:
        data = data_preprocess_function(data)
    output = models["classifier"](data)
    loss = loss_functions["CrossEntropy"](output, target)
    return loss, output


def train_classifier(args):
    models = {"classifier": get_classifier()}
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optimizer = optim.Adam(models["classifier"].parameters(), lr=learning_rate, weight_decay=weight_decay)
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
    split_dataset = data.random_split(
        entire_train_dataset,[int(0.8 * len(entire_train_dataset)), int(0.2 * len(entire_train_dataset))]
    )
    train_set = split_dataset[0]
    validation_set = split_dataset[1]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    """
    test_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                                  ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    """

    path = args.experiments_path
    experiment = Experiment(name, models,
                            train_loader, validation_loader,
                            {"CrossEntropy": torch.nn.CrossEntropyLoss()}, loss_calc_func,
                            optimizer, path, scheduler=scheduler,
                            data_preprocess_function=bw2rgb_expand_channels)
    experiment.run(delta_epochs_to_save_checkpoint=10)