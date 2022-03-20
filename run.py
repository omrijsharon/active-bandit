from pipetorch.experiment import Experiment
import torch
from torch import nn
import torchvision
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size_train', type=int, default=5000, required=False, help="Batch size during training.")
parser.add_argument('--batch_size_test', type=int, default=1, required=False, help="Batch size during testing.")
parser.add_argument('--learning_rate', type=float, default=1e-3, required=False, help="Learning rate.")
parser.add_argument('--weight_decay', type=float, default=0.2, required=False, help="Weight decay.")
parser.add_argument('--experiments_path', type=str, default=r'C:\Users\omrijsharon\Documents\Experiments', required=False, help="Full path of the experiments directory for TensorBoard and Checkpoints.")


args = parser.parse_args()


def main():
    # model = torchvision.models.mobilenet_v3_large(False)
    model = nn.Sequential(torchvision.models.mobilenet_v3_large(False), nn.Linear(1000, 10))
    # model = nn.Sequential(torchvision.models.mobilenet_v3_small(False), nn.Linear(1000, 10))
    batch_size_train = args.batch_size_train
    batch_size_test = args.batch_size_test
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    def preprocess(data):
        return data.expand(data.size(0), 3, data.size(2), data.size(3))

    path = args.experiments_path
    experiment = Experiment(model, train_loader, torch.nn.CrossEntropyLoss(), optimizer, path, scheduler=scheduler,
                            data_preprocess_function=preprocess)
    experiment.run(delta_epochs_to_save_checkpoint=10)


if __name__ == '__main__':
    main()