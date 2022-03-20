import os
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from pipetorch import callback_functions
from time import sleep


def start_tensorboard(experiment_path):
    os.popen(r'tensorboard --logdir=' + experiment_path)


class Experiment(object):
    def __init__(self, model, data_loader, loss_function, optimizer, experiments_path, scheduler=None, data_preprocess_function=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.experiment_path = os.path.join(experiments_path, datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        if not os.path.exists(self.experiment_path):
            os.mkdir(self.experiment_path)
        self.writer = SummaryWriter(self.experiment_path)
        self.data_preprocess_function = data_preprocess_function

    def run_single_epoch(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with tqdm(total=len(self.data_loader)) as pbar:
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data = data.to(self.device)
                if self.data_preprocess_function is not None:
                    data = self.data_preprocess_function(data)
                target = target.to(self.device)
                output = self.model(data)
                loss = self.loss_function(output, target)
                loss.backward()
                accuracy = callback_functions.accuracy(output, target).item()
                recall, precision, f1_score = callback_functions.recall_precision_f1(output, target)
                pbar.set_description("Epoch: {},  Loss: {:.3f},  Accuracy: {:.3f}".format(self.epoch, loss.item(), accuracy))
                pbar.update(1)
                self.writer.add_scalar('Metric/Training loss', loss.item(), self.epoch * len(self.data_loader) + batch_idx)
                self.writer.add_scalar('Metric/Model Entropy', Categorical(logits=output).entropy().mean().item(), self.epoch * len(self.data_loader) + batch_idx)
                self.writer.add_scalar('Metric/Accuracy', accuracy, self.epoch * len(self.data_loader) + batch_idx)
                self.writer.add_scalar('Metric/Recall', recall, self.epoch * len(self.data_loader) + batch_idx)
                self.writer.add_scalar('Metric/Precision', precision, self.epoch * len(self.data_loader) + batch_idx)
                self.writer.add_scalar('Metric/F1 Score', f1_score, self.epoch * len(self.data_loader) + batch_idx)
                self.writer.add_scalar('Metric/Gradients mean', callback_functions.grad_abs_mean(self.model), self.epoch * len(self.data_loader) + batch_idx)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
        self.epoch += 1

    def run(self):
        os.popen(r'tensorboard --logdir=' + self.experiment_path)
        while True:
            self.run_single_epoch()

    def eval(self, eval_data_loader):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(eval_data_loader)):
                output = self.model(data)
                self.writer.add_scalar('training loss', output.item(), self.epoch * len(self.data_loader) + batch_idx)

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.experiment_path, 'model.pth'))
        torch.save(self.optimizer.state_dict(), 'optimizer.pth')

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pth')))
        self.model.to(self.device)
        self.epoch = 0