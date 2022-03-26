import os
import torch
from torch import nn
from torch import optim
from pipetorch.experiment import Experiment
import torchvision
from torch.utils import data
import matplotlib.pyplot as plt
from utils.helper_functions import resize_dataset_and_pad, embed, SoftABS
from models.classifier import get_classifier
from PIL import Image
from utils.helper_functions import bw2rgb_expand_channels


class SelfEncoder(nn.Module):
    def __init__(self, n_blocks, activation_func):
        super(SelfEncoder, self).__init__()
        self.layers = torch.nn.ModuleList([nn.ConvTranspose2d(2, 32, kernel_size=(3, 3)), nn.BatchNorm2d(32), activation_func()])
        for _ in range(n_blocks):
            self.layers.extend([nn.Conv2d(32, 32, kernel_size=(3, 3)), nn.BatchNorm2d(32), activation_func()])
            self.layers.extend([nn.ConvTranspose2d(32, 32, kernel_size=(3, 3)), nn.BatchNorm2d(32), activation_func()])
        self.layers.extend([nn.Conv2d(32, 1, kernel_size=(3, 3)), nn.BatchNorm2d(1), nn.Tanh()])

    def forward(self, x):
        xx = torch.zeros(size=(x.size(0), 32, 30, 30)).cuda()
        for i in range(int(len(self.layers)/3)):
            for j in range(3):
                if j == 0:
                    if i % 2 == 0:
                        xx = self.layers[3 * i + j](x) + xx
                    elif i % 2 == 1:
                        if i > 1 and i < int(len(self.layers)/3) - 1:
                            x = self.layers[3 * i + j](xx) + x
                        else:
                            x = self.layers[3 * i + j](xx)
                elif j == 1:
                    if i % 2 == 0:
                        xx = self.layers[3 * i + j](xx)
                    elif i % 2 == 1:
                        x = self.layers[3 * i + j](x)
                elif j == 2:
                    if i % 2 == 0:
                        xx = self.layers[3 * i + j](xx)
                    elif i % 2 == 1:
                        x = self.layers[3 * i + j](x)
        return x


class Encoder(nn.Module):
    def __init__(self, checkpoint_path):
        super(Encoder, self).__init__()
        classifier = get_classifier(checkpoint_path)
        self.model = torchvision.models.mobilenet_v3_large(False)
        classifier_params = [param for param in classifier.parameters()]
        for i, param in enumerate(self.model.parameters()):
            param.data = classifier_params[i].data

    def forward(self, x):
        if len(x.size()) == 3 or (len(x.size()) == 4 and x.size(1) == 1):
            x = bw2rgb_expand_channels(x)
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_size: int, output_channels: int, activation_func):
        super(Decoder, self).__init__()
        self.layers = torch.nn.ModuleList([nn.Linear(input_size, 2048), nn.BatchNorm1d(num_features=2048), activation_func()])
        self.layers.extend([nn.Linear(2048, 4096), nn.BatchNorm1d(num_features=4096), activation_func()])
        self.layers.extend([nn.ConvTranspose2d(16, 32, kernel_size=(3, 3)), nn.BatchNorm2d(32), activation_func()])
        self.layers.extend([nn.ConvTranspose2d(32, 32, kernel_size=(3, 3)), nn.BatchNorm2d(32), activation_func()])
        self.layers.extend([nn.ConvTranspose2d(32, 64, kernel_size=(3, 3)), nn.BatchNorm2d(64), activation_func()])
        self.layers.extend([nn.ConvTranspose2d(64, 32, kernel_size=(3, 3)), nn.BatchNorm2d(32), activation_func()])
        self.layers.extend([nn.ConvTranspose2d(32, 32, kernel_size=(3, 3)), nn.BatchNorm2d(32), activation_func()])
        self.layers.extend([nn.ConvTranspose2d(32, 16, kernel_size=(3, 3)), nn.BatchNorm2d(16), activation_func()])
        self.layers.extend([nn.ConvTranspose2d(16, 16, kernel_size=(3, 3)), nn.BatchNorm2d(16), activation_func()])
        self.layers.extend([nn.Conv2d(16, output_channels, kernel_size=(3, 3)), nn.BatchNorm2d(1), nn.Tanh()])

    def forward(self, x):
        for i in range(int(len(self.layers)/3)):
            if i == 2:
                x = x.reshape(-1, 16, 16, 16)
            for j in range(3):
                x = self.layers[3 * i + j](x)
        return x


def loss_calc_func(models, data, target, loss_functions, data_preprocess_function):
    if data_preprocess_function is not None:
        data = data_preprocess_function(data)
    with torch.no_grad():
        latent_data = models["encoder"](data)
    output = models["decoder"](latent_data)
    with torch.no_grad():
        latent_output = models["encoder"](output)
    loss_cosim = -loss_functions["CosineSimilarity"](output, data)
    loss_l2 = loss_functions["MSE"](latent_output, latent_data)
    p = 0.8
    loss = p * loss_cosim + (1 - p) * loss_l2
    return loss, output


def train_encoder(args):
    ckpt = r'C:\Users\omrijsharon\Documents\Experiments\2022_03_22-09_36_31\epoch_000040'
    model = {"encoder": Encoder(checkpoint_path=ckpt), "decoder": Decoder(input_size=1000, output_channels=3, activation_func=nn.Mish)}
    loss_functions = {"CosineSimilarity": nn.CosineSimilarity(), "MSE": nn.MSELoss()}
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    entire_train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              (0.1307,), (0.3081,))
                                                      ]))
    # resized_dataset = resize_dataset_and_pad(entire_train_dataset, small_dim=(7, 7), large_dim=(28, 28))
    # embedded_data = embed(entire_train_dataset.data, resized_dataset.data)
    # entire_train_dataset.data = embedded_data
    split_dataset = data.random_split(entire_train_dataset, [int(0.8 * len(entire_train_dataset)), int(0.2 * len(entire_train_dataset))])
    train_set = split_dataset[0]
    validation_set = split_dataset[1]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    path = args.experiments_path
    model_name = "autoencoder"
    experiment = Experiment(model_name, model,
                            train_loader, validation_loader,
                            loss_functions, loss_calc_func,
                            optimizer, path, scheduler=scheduler,
                            data_preprocess_function=None, mode='regression')
    experiment.run(delta_epochs_to_save_checkpoint=10)