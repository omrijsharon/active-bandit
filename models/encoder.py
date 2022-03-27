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
from utils.helper_functions import bw2rgb_expand_channels, SoftABS, dataset2split_loaders


class WatermarkEncoder(nn.Module):
    def __init__(self):
        super(WatermarkEncoder, self).__init__()
        self.w = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, img, watermark):
        return img + self.w * watermark + self.bias


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
        classifier = get_classifier(model_name="classifier", checkpoint_path=checkpoint_path)
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
        self.layers.extend([nn.Conv2d(16, output_channels, kernel_size=(3, 3)), nn.Identity(), nn.Tanh()])

    def forward(self, x):
        for i in range(int(len(self.layers)/3)):
            if i == 2:
                x = x.reshape(-1, 16, 16, 16)
            for j in range(3):
                x = self.layers[3 * i + j](x)
        return x


def get_decoder(model_name=None, checkpoint_path=None, input_size=1000, output_channels=3, activation_func=nn.Mish):
    decoder = Decoder(input_size, output_channels, activation_func)
    filename = 'model.pth'
    if not model_name is None:
        filename = 'model_{}.pth'.format(model_name)
    if not checkpoint_path is None:
        decoder.load_state_dict(torch.load(os.path.join(checkpoint_path, filename)))
    return decoder


# def loss_calc_func(models, data, target, loss_functions, data_preprocess_function):
#     if data_preprocess_function is not None:
#         data = data_preprocess_function(data)
#     with torch.no_grad():
#         latent_data = models["encoder"](data)
#     output = models["decoder"](latent_data)
#     with torch.no_grad():
#         latent_output = models["encoder"](output)
#     loss_softabs = loss_functions["SoftABS"](output, data)
#     loss_l2 = loss_functions["MSE"](latent_output, latent_data) / torch.sqrt(torch.tensor(output.size(1)))
#     p = 0.9
#     loss = p * loss_softabs + (1 - p) * loss_l2
#     return loss, output

def loss_calc_func(models, data, target, loss_functions, data_preprocess_function):
    if data_preprocess_function is not None:
        data = data_preprocess_function(data)
    with torch.no_grad():
        gt = models["classifier"](data)
    output = models["watermark_encoder"](data)
    with torch.no_grad():
        latent_output = models["encoder"](output)
    loss = loss_functions["SoftABS"](output, data)
    return loss, output


def train_encoder(args):
    # encoder_ckpt = r'C:\Users\omrijsharon\Documents\Experiments\full_classifier_2022_03_26-13_51_11\epoch_000100'
    # # decoder_ckpt = r'C:\Users\omrijsharon\Documents\Experiments\autoencoder_2022_03_26-16_36_32\epoch_000230'
    # decoder_ckpt = None
    # model = {
    #     "encoder": Encoder(checkpoint_path=encoder_ckpt),
    #     "decoder": get_decoder(model_name="decoder", checkpoint_path=decoder_ckpt, input_size=1000, output_channels=3, activation_func=nn.Mish)
    # }
    classifier_ckpt = r'C:\Users\omrijsharon\Documents\Experiments\full_classifier_2022_03_26-13_51_11\epoch_000100'
    model = {
        "classifier": get_classifier("classifier", checkpoint_path=classifier_ckpt),
        "watermark_encoder": WatermarkEncoder(),
        "watermark_classifier": get_classifier()
    }
    loss_functions = {"SoftABS": SoftABS(), "MSE": nn.MSELoss()}
    optimizer = optim.Adam(model["watermark_encoder"].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    entire_train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              (0.1307,), (0.3081,))
                                                      ]))
    resized_dataset = resize_dataset_and_pad(entire_train_dataset, small_dim=(7, 7), large_dim=(28, 28))
    # embedded_data = embed(entire_train_dataset.data, resized_dataset.data)
    # entire_train_dataset.data = embedded_data
    split = [0.8, 0.2]
    train_loader, validation_loader, split = dataset2split_loaders(entire_train_dataset, args.batch_size, split)

    train_loader_watermark, validation_loader_watermark, split_watermark = dataset2split_loaders(resized_dataset, args.batch_size, p=0.8)



    data_loaders = {
        "train": zip(train_loader, train_loader_watermark), "eval": zip(validation_loader, validation_loader_watermark)
    }
    path = args.experiments_path
    model_name = "autoencoder"
    experiment = Experiment(model_name, model,
                            data_loaders,
                            loss_functions, loss_calc_func,
                            optimizer, path, scheduler=scheduler,
                            data_preprocess_function=None, mode='decoder')
    experiment.run(delta_epochs_to_save_checkpoint=10)