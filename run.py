from pipetorch.experiment import Experiment
import torch
from torch import nn
from torch.utils import data
import torchvision
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=5000, required=False, help="Batch size.")
parser.add_argument('--learning_rate', type=float, default=1e-3, required=False, help="Learning rate.")
parser.add_argument('--weight_decay', type=float, default=0.0, required=False, help="Weight decay.")
parser.add_argument('--experiments_path', type=str, default=r'C:\Users\omrijsharon\Documents\Experiments', required=False, help="Full path of the experiments directory for TensorBoard and Checkpoints.")


args = parser.parse_args()


def main():
    # model = torchvision.models.mobilenet_v3_large(False)
    model = nn.Sequential(torchvision.models.mobilenet_v3_large(False), nn.Linear(1000, 10))
    # model = nn.Sequential(torchvision.models.mobilenet_v3_small(False), nn.Linear(1000, 10))
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
    # Weight decay scheduler? increase weight_decay when loss goes down
    # Dropout scheduler? increase dropout rate when loss goes down
    entire_train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              (0.1307,), (0.3081,))
                                                          ]))

    split_dataset = data.random_split(
        entire_train_dataset,[int(0.8 * len(entire_train_dataset)), int(0.2 * len(entire_train_dataset))]
    )
    train_set = split_dataset[0]
    validation_set = split_dataset[1]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                                  ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    def preprocess(data):
        return data.expand(data.size(0), 3, data.size(2), data.size(3))

    path = args.experiments_path
    experiment = Experiment(model, train_loader, validation_loader, torch.nn.CrossEntropyLoss(), optimizer, path, scheduler=scheduler,
                            data_preprocess_function=preprocess)
    experiment.run(delta_epochs_to_save_checkpoint=10)


if __name__ == '__main__':
    main()