from models.classifier import train_classifier
from models.encoder import train_encoder
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=8000, required=False, help="Batch size.")
parser.add_argument('--learning_rate', type=float, default=1e-3, required=False, help="Learning rate.")
parser.add_argument('--weight_decay', type=float, default=0.1, required=False, help="Weight decay.")
parser.add_argument('--watermark', type=bool, default=False, required=False, help="Reduces dataset images to 7x7.")
parser.add_argument('--experiments_path', type=str, default=r'C:\Users\omrijsharon\Documents\Experiments', required=False, help="Full path of the experiments directory for TensorBoard and Checkpoints.")
# parser.add_argument('--ckpt_path', type=str, default=r'C:\Users\omrijsharon\Documents\Experiments\full_classifier_2022_03_26-13_51_11\epoch_000050', required=False, help="Full path of the experiments directory for TensorBoard and Checkpoints.")
# parser.add_argument('--ckpt_path', type=str, default=r'C:\Users\omrijsharon\Documents\Experiments\full_classifier_2022_03_30-23_46_10\epoch_000050', required=False, help="Full path of the experiments directory for TensorBoard and Checkpoints.")
parser.add_argument('--ckpt_path', type=str, default=None, required=False, help="Full path of the experiments directory for TensorBoard and Checkpoints.")

args = parser.parse_args()


def main():
    train_classifier(args)
    # train_encoder(args)


if __name__ == '__main__':
    main()