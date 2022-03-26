from models.classifier import train_classifier
from models.encoder import train_encoder
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=5000, required=False, help="Batch size.")
parser.add_argument('--learning_rate', type=float, default=1e-3, required=False, help="Learning rate.")
parser.add_argument('--weight_decay', type=float, default=0.05, required=False, help="Weight decay.")
parser.add_argument('--watermark', type=bool, default=False, required=False, help="Reduces dataset images to 7x7.")
parser.add_argument('--experiments_path', type=str, default=r'C:\Users\omrijsharon\Documents\Experiments', required=False, help="Full path of the experiments directory for TensorBoard and Checkpoints.")


args = parser.parse_args()


def main():
    train_classifier(args)
    # train_encoder(args)


if __name__ == '__main__':
    main()