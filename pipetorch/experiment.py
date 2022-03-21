import os
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pipetorch import callback_functions
from datetime import datetime, timedelta


class Experiment(object):
    def __init__(self, model, train_data_loader, eval_data_loader, loss_function, optimizer, experiments_path, scheduler=None, data_preprocess_function=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.experiment_path = os.path.join(experiments_path, datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        if not os.path.exists(self.experiment_path):
            os.mkdir(self.experiment_path)
            os.mkdir(os.path.join(self.experiment_path, "train"))
            os.mkdir(os.path.join(self.experiment_path, "eval"))
        self.train_writer = SummaryWriter(os.path.join(self.experiment_path, "train"))
        self.eval_writer = SummaryWriter(os.path.join(self.experiment_path, "eval"))
        self.data_preprocess_function = data_preprocess_function
        self.t0 = datetime.now()

    def run_single_epoch(self):
        self.epoch += 1
        data_loader = self.train_data_loader
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with tqdm(total=len(data_loader)) as pbar:
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device)
                if self.data_preprocess_function is not None:
                    data = self.data_preprocess_function(data)
                target = target.to(self.device)
                output = self.model(data)
                loss = self.loss_function(output, target)
                loss.backward()
                accuracy = callback_functions.accuracy(output, target).item()
                recall, precision, f1_score = callback_functions.recall_precision_f1(output, target)
                now = datetime.now()
                pbar.set_description("{} @ Elapsed time: {} @ Epoch: {}, Train".format(now.strftime("%d/%m/%Y, %H:%M:%S"), str(now-self.t0)[:-7], self.epoch))
                pbar.update(1)
                self.train_writer.add_scalar('Metric/Loss', loss.item(), self.epoch * len(data_loader) + batch_idx)
                self.train_writer.add_scalar('Metric/Model Entropy', Categorical(logits=output).entropy().mean().item(), self.epoch * len(data_loader) + batch_idx)
                self.train_writer.add_scalar('Metric/Accuracy', accuracy, self.epoch * len(data_loader) + batch_idx)
                self.train_writer.add_scalar('Metric/Recall', recall, self.epoch * len(data_loader) + batch_idx)
                self.train_writer.add_scalar('Metric/Precision', precision, self.epoch * len(data_loader) + batch_idx)
                self.train_writer.add_scalar('Metric/F1 Score', f1_score, self.epoch * len(data_loader) + batch_idx)
                self.train_writer.add_scalar('Metric/Gradients mean length', callback_functions.grad_abs_mean(self.model), self.epoch * len(data_loader) + batch_idx)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()

    def run(self, delta_epochs_to_save_checkpoint=100):
        os.popen(r'tensorboard --logdir=' + self.experiment_path)
        while True:
            self.run_single_epoch()
            self.eval()
            if self.epoch % delta_epochs_to_save_checkpoint == 0:
                self.save()

    def eval(self):
        # self.model.eval()
        data_loader = self.train_data_loader
        with torch.no_grad():
            with tqdm(total=len(self.eval_data_loader)) as pbar:
                for batch_idx, (data, target) in enumerate(self.eval_data_loader):
                    data = data.to(self.device)
                    if self.data_preprocess_function is not None:
                        data = self.data_preprocess_function(data)
                    target = target.to(self.device)
                    output = self.model(data)
                    loss = self.loss_function(output, target)
                    accuracy = callback_functions.accuracy(output, target).item()
                    recall, precision, f1_score = callback_functions.recall_precision_f1(output, target)
                    now = datetime.now()
                    pbar.set_description("{} @ Elapsed time: {} @ Epoch: {}, Eval ".format(now.strftime("%d/%m/%Y, %H:%M:%S"), str(now - self.t0)[:-7], self.epoch))
                    pbar.update(1)
                    self.eval_writer.add_scalar('Metric/Loss', loss.item(), self.epoch * len(data_loader) + batch_idx)
                    self.eval_writer.add_scalar('Metric/Model Entropy', Categorical(logits=output).entropy().mean().item(), self.epoch * len(data_loader) + batch_idx)
                    self.eval_writer.add_scalar('Metric/Accuracy', accuracy, self.epoch * len(data_loader) + batch_idx)
                    self.eval_writer.add_scalar('Metric/Recall', recall, self.epoch * len(data_loader) + batch_idx)
                    self.eval_writer.add_scalar('Metric/Precision', precision, self.epoch * len(data_loader) + batch_idx)
                    self.eval_writer.add_scalar('Metric/F1 Score', f1_score, self.epoch * len(data_loader) + batch_idx)

    def save(self):
        epoch_path = os.path.join(self.experiment_path, 'epoch_%06d' % self.epoch)
        if not os.path.exists(epoch_path):
            os.mkdir(epoch_path)
        torch.save(self.model.state_dict(), os.path.join(epoch_path, 'model.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(epoch_path, 'optimizer.pth'))

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pth')))
        self.model.to(self.device)
        self.epoch = 0