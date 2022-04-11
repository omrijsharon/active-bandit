from pipetorch.experiment import Experiment
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from models.vanilla_vae import VanillaVAE
from models.beta_vae import BetaVAE
from models.classifier import get_classifier

from torch.utils import data
import torchvision
import torch.optim as optim
from utils.helper_functions import bw2rgb_expand_channels, resize_dataset, dataset2split_loaders
import argparse
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

entire_train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(
                                                          (0.1307,), (0.3081,))
                                                  ]))


def get_vae(model_name=None, checkpoint_path=None):
    vae = None
    if "vannila" in model_name:
        vae = VanillaVAE(1, latent_dim=2, hidden_dims=[32, 64, 128, 256, 512])
    elif "beta" in model_name.lower():
        vae = BetaVAE(1, latent_dim=2, hidden_dims=[32, 64, 128, 256, 512])
    else:
        ValueError("No such model {}".format(model_name))
    # classifier = Classifier(base_model=torchvision.models.mobilenet_v3_large(False), n_labels=10, is_temp=True)
    filename = 'model.pth'
    if not model_name is None:
        filename = 'model_{}.pth'.format(model_name)
    if not checkpoint_path is None:
        vae.load_state_dict(torch.load(os.path.join(checkpoint_path, filename)))
    return vae


def vae_trainer(args):
    model_name = "beta_vae"
    models = {model_name: get_vae(model_name, checkpoint_path=args.ckpt_path)}
    # models = {model_name: get_classifier(model_name=model_name, checkpoint_path=None)}
    optimizer = optim.Adam(models[model_name].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = optim.RMSprop(models["vae"].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # Weight decay scheduler? increase weight_decay when loss goes down
    # Dropout scheduler? increase dropout rate when loss goes down
    experiment = Experiment(model_name, models,
                            entire_train_dataset, args.batch_size,
                            {"CrossEntropy": torch.nn.CrossEntropyLoss}, models[model_name].loss_function,
                            optimizer, args.experiments_path, scheduler=scheduler,
                            data_preprocess_function=None, mode="self_supervised")
    experiment.run(delta_epochs_to_save_checkpoint=10)


def classifier_trainer(args):

    def loss_calc_func(models, data, target, loss_functions, data_preprocess_function):
        if data_preprocess_function is not None:
            data = data_preprocess_function(data)
        output = models["classifier"](data)
        loss = loss_functions["CrossEntropy"](output, target)
        return loss, output

    model_name = "classifier"
    models = {
        "classifier": get_classifier(model_name=model_name, checkpoint_path=None),
        "vae": get_vae("vae", args.ckpt_path)
    }
    # models = {model_name: get_classifier(model_name=model_name, checkpoint_path=None)}
    optimizer = optim.Adam(models["classifier"].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # Weight decay scheduler? increase weight_decay when loss goes down
    # Dropout scheduler? increase dropout rate when loss goes down
    experiment = Experiment(model_name, models,
                                entire_train_dataset, args.batch_size,
                                {"CrossEntropy": torch.nn.CrossEntropyLoss}, loss_calc_func,
                                optimizer, args.experiments_path, scheduler=scheduler,
                                data_preprocess_function=bw2rgb_expand_channels)
    experiment.run(delta_epochs_to_save_checkpoint=50)
