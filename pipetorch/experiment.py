import numpy as np
import os
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pipetorch import callback_functions
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from drawnow import drawnow


class Experiment(object):
    def __init__(self, name: str, models: dict, train_data_loader, eval_data_loader, loss_functions: dict, loss_calc_func, optimizer, experiments_path, scheduler=None, data_preprocess_function=None, mode="classifier"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.models = {k: model.to(self.device) for k, model in models.items()}
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.loss_functions = loss_functions
        self.loss_calc_func = loss_calc_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mode = mode
        self.epoch = 0
        self.experiment_path = os.path.join(experiments_path, self.name + "_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        if not os.path.exists(self.experiment_path):
            os.mkdir(self.experiment_path)
            os.mkdir(os.path.join(self.experiment_path, "train"))
            os.mkdir(os.path.join(self.experiment_path, "eval"))
        self.train_writer = SummaryWriter(os.path.join(self.experiment_path, "train"))
        self.eval_writer = SummaryWriter(os.path.join(self.experiment_path, "eval"))
        self.data_preprocess_function = data_preprocess_function
        self.t0 = datetime.now()

    def write2tensorboard(self, writer, output, target, batch_idx, loss, dataloader_length):
        writer.add_scalar('Metric/Loss', loss.item(), self.epoch * dataloader_length + batch_idx)
        if self.mode == "classifier":
            accuracy = callback_functions.accuracy(output, target).item()
            recall, precision, f1_score = callback_functions.recall_precision_f1(output, target)
            writer.add_scalar('Metric/Model Entropy', Categorical(logits=output).entropy().mean().item(), self.epoch * dataloader_length + batch_idx)
            writer.add_scalar('Metric/Accuracy', accuracy, self.epoch * dataloader_length + batch_idx)
            writer.add_scalar('Metric/Recall', recall, self.epoch * dataloader_length + batch_idx)
            writer.add_scalar('Metric/Precision', precision, self.epoch * dataloader_length + batch_idx)
            writer.add_scalar('Metric/F1 Score', f1_score, self.epoch * dataloader_length + batch_idx)
        writer.add_scalar('Metric/Gradients mean length', callback_functions.grad_abs_mean(self.model), self.epoch * dataloader_length + batch_idx)

    def update_tqdm(self, pbar):
        now = datetime.now()
        pbar.set_description("{} @ Elapsed time: {} @ Epoch: {}, Train".format(now.strftime("%d/%m/%Y, %H:%M:%S"), str(now - self.t0)[:-7], self.epoch))
        pbar.update(1)

    def run_single_epoch(self):
        self.epoch += 1
        data_loader = self.train_data_loader
        dataloader_length = len(data_loader)
        {model.train() for model in self.models.values()}
        self.optimizer.zero_grad(set_to_none=True)
        with tqdm(total=dataloader_length) as pbar:
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                loss, output = self.loss_calc_func(self.models, data, target, self.loss_functions, self.data_preprocess_function)
                loss.backward()

                self.write2tensorboard(self.train_writer, output, target, batch_idx, loss, dataloader_length)
                self.update_tqdm(pbar)

                # optimizer updates weights
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
        {model.eval() for model in self.models.values()}
        loss_fn_each = torch.nn.CrossEntropyLoss(reduction='none')
        data_loader = self.train_data_loader
        dataloader_length = len(data_loader)
        with torch.no_grad():
            with tqdm(total=len(self.eval_data_loader)) as pbar:
                for batch_idx, (data, target) in enumerate(self.eval_data_loader):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    loss, output = self.loss_calc_func(self.models, data, target, self.loss_functions, self.data_preprocess_function)
                    if self.mode == "classifier":
                        self.loss_each = loss_fn_each(output, target)
                        self.entropy_each = Categorical(logits=output).entropy()
                        if batch_idx==0:
                            drawnow(self.make_fig)
                    self.write2tensorboard(self.eval_writer, output, target, batch_idx, loss, dataloader_length)
                    self.update_tqdm(pbar)

    def save(self):
        epoch_path = os.path.join(self.experiment_path, 'epoch_%06d' % self.epoch)
        if not os.path.exists(epoch_path):
            os.mkdir(epoch_path)
        {torch.save(model.state_dict(), os.path.join(epoch_path, 'model_{}.pth'.format(name))) for name, model in self.models.items()}
        torch.save(self.optimizer.state_dict(), os.path.join(epoch_path, 'optimizer.pth'))

    def load(self, path):
        self.models = {name: model.load_state_dict(torch.load(os.path.join(path, 'model_{}.pth'.format(name)))) for name, model in self.models.items()}
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pth')))
        self.models = {k: model.to(self.device) for k, model in self.models.items()}
        self.epoch = 0

    def make_fig(self):
        entropy_np = self.entropy_each.cpu().detach().numpy()
        loss_np = self.loss_each.cpu().detach().numpy()
        n_bins = 50
        bins = np.linspace(0, entropy_np.max(), n_bins)
        idx = np.digitize(entropy_np, bins=bins)
        loss_mean = []
        loss_std = []
        entropy_axis = []
        loss_min = []
        for i in range(n_bins):
            if len(loss_np[idx == i]) > 1:
                loss_mean.append(loss_np[idx==i].mean())
                loss_std.append(2 * loss_np[idx==i].std())
                loss_min.append(loss_np[idx==i].min())
                # entropy_axis.append((bins[i-1] + bins[i])/2)
                entropy_axis.append(bins[i])
        loss_std = np.array(loss_std)
        loss_mean = np.array(loss_mean)
        loss_min = np.array(loss_min)
        lower_error = loss_std + np.clip(loss_mean - loss_std - loss_min, a_min=-np.inf, a_max=0)
        upper_error = loss_std
        plt.plot(entropy_np, loss_np, 'o', markersize=4, markeredgecolor=None, markeredgewidth=0, alpha=0.2)
        plt.errorbar(np.array(entropy_axis), loss_mean, yerr=[lower_error, upper_error], linestyle='None', marker='o', alpha=0.8, zorder=3)
        plt.ylabel('Cross-Entropy Loss')
        plt.xlabel('Entropy')
        epoch_path = os.path.join(self.experiment_path, 'epoch_%06d' % self.epoch)
        if not os.path.exists(epoch_path):
            os.mkdir(epoch_path)
        plt.savefig(os.path.join(epoch_path, "loss_vs_entropy_epoch_{}.png".format(self.epoch)))