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
from time import sleep


class Experiment(object):
    def __init__(self, name: str, models: dict, data_loaders, loss_functions: dict, loss_calc_func, optimizer, experiments_path, scheduler=None, data_preprocess_function=None, mode="classifier"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.models = {k: model.to(self.device) for k, model in models.items()}
        self.data_loaders = data_loaders
        self.batch_size = {k: data_loader.batch_size for k, data_loader in self.data_loaders.items()}
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
        print("Experiment dir: {}".format(self.experiment_path))
        self.data_preprocess_function = data_preprocess_function
        self.t0 = datetime.now()

    def write2tensorboard(self, writer, output, target, batch_idx, loss, index):
        writer.add_scalar('Metric/Loss', loss.item(), index)
        if self.mode == "classifier":
            accuracy = callback_functions.accuracy(output, target).item()
            recall, precision, f1_score = callback_functions.recall_precision_f1(output, target)
            writer.add_scalar('Metric/Model Entropy', Categorical(logits=output).entropy().mean().item(), index)
            writer.add_scalar('Metric/Accuracy', accuracy, index)
            writer.add_scalar('Metric/Recall', recall, index)
            writer.add_scalar('Metric/Precision', precision, index)
            writer.add_scalar('Metric/F1 Score', f1_score, index)
        # writer.add_scalar('Metric/Gradients mean length', callback_functions.grad_abs_mean(list(self.models.values())[0]), self.epoch * dataloader_length + batch_idx)

    def update_tqdm(self, pbar, text):
        now = datetime.now()
        # used_gpu_mem = (torch.cuda.mem_get_info(device="cuda:0")[1] - torch.cuda.mem_get_info(device="cuda:0")[0]) * 1e-6
        gpu_mem_info = torch.cuda.mem_get_info(device="cuda:0")
        gpu_usage = 1 - (gpu_mem_info[0] / gpu_mem_info[1])
        pbar.set_description("{} @ Elapsed time: {} @ Epoch: {}, GPU memory usage: {} %, {}".format(now.strftime("%d/%m/%Y, %H:%M:%S"), str(now - self.t0)[:-7], self.epoch, int(gpu_usage*10000)/100, text))
        pbar.update(1)

    @staticmethod
    def calc_idx(epoch, dataset_length, batch_idx, batch_size):
        return epoch * dataset_length + batch_idx * batch_size

    def dataset_calc(self, data_loader, n_labels):
        {model.train() for model in self.models.values()}
        loss_fn_each = {"CrossEntropy": torch.nn.CrossEntropyLoss(reduction='none')}
        dataset_length = len(data_loader.dataset.indices)
        dataloader_length = len(data_loader)
        n_runs = 10
        entropy_mean = torch.zeros(size=(dataset_length,))
        entropy_std = torch.zeros(size=(dataset_length,))
        losses_mean = torch.zeros(size=(dataset_length,))
        losses_std = torch.zeros(size=(dataset_length,))
        outputs = torch.zeros(size=(dataset_length, n_runs, n_labels))
        with torch.no_grad():
            with tqdm(total=dataloader_length) as pbar:
                for batch_idx, (data, target) in enumerate(data_loader):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    batch_size = len(data)
                    entropy_temp = torch.zeros(size=(batch_size, n_runs)).to(self.device)
                    losses_temp = torch.zeros(size=(batch_size, n_runs)).to(self.device)
                    outputs_temp = torch.zeros(size=(batch_size, n_runs, n_labels)).to(self.device)
                    for i in range(n_runs):
                        loss, output = self.loss_calc_func(self.models, data, target, loss_fn_each, self.data_preprocess_function)
                        entropy_temp[:, i] = Categorical(logits=output).entropy()
                        losses_temp[:, i] = loss
                        outputs_temp[:, i, :] = output
                    idx = self.calc_idx(0, dataset_length, batch_idx, batch_size)
                    idx = torch.arange(idx, idx + batch_size)
                    entropy_mean[idx] = entropy_temp.mean(dim=1).cpu()
                    entropy_std[idx] = entropy_temp.std(dim=1).cpu()
                    losses_mean[idx] = losses_temp.mean(dim=1).cpu()
                    losses_std[idx] = losses_temp.std(dim=1).cpu()
                    outputs[idx] = outputs_temp.cpu()
                    self.update_tqdm(pbar, text="Dataset")
                mean = outputs.mean(dim=1)
                std = outputs.std(dim=1)
        return mean, std, entropy_mean, entropy_std, losses_mean, losses_std

    def run_single_epoch(self):
        self.epoch += 1
        key = "train"
        data_loader = self.data_loaders[key]
        dataset_length = len(data_loader.dataset.indices)
        {model.train() for model in self.models.values()}
        self.optimizer.zero_grad(set_to_none=True)
        with tqdm(total=len(data_loader)) as pbar:
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                batch_size = len(data)
                loss, output = self.loss_calc_func(self.models, data, target, self.loss_functions, self.data_preprocess_function)
                loss.backward()

                idx = self.calc_idx(self.epoch, dataset_length, batch_size, batch_idx)
                self.write2tensorboard(self.train_writer, output, target, batch_idx, loss, idx)
                self.update_tqdm(pbar, text="Train")

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
        # {model.eval() for model in self.models.values()}
        loss_fn_each = torch.nn.CrossEntropyLoss(reduction='none')
        key = "eval"
        data_loader = self.data_loaders[key]
        dataset_length = len(data_loader.dataset.indices)
        with torch.no_grad():
            with tqdm(total=len(data_loader)) as pbar:
                for batch_idx, (data, target) in enumerate(self.eval_data_loader):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    batch_size = len(data)
                    loss, output = self.loss_calc_func(self.models, data, target, self.loss_functions, self.data_preprocess_function)
                    if self.mode == "classifier":
                        self.loss_each = loss_fn_each(output, target)
                        self.entropy_each = Categorical(logits=output).entropy()
                        if batch_idx==0 and self.epoch > 0:
                            drawnow(self.make_fig)
                    elif self.mode == "decoder":
                        if batch_idx==0 and self.epoch > 0:
                            self.idx = torch.randperm(8)
                            self.output = output[self.idx]
                            self.data = data
                            drawnow(self.make_fig)
                    idx = self.calc_idx(self.epoch, dataset_length, batch_size, batch_idx)
                    self.write2tensorboard(self.eval_writer, output, target, batch_idx, loss, idx)
                    self.update_tqdm(pbar, text="Eval")

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
        if self.mode == "classifier":
            entropy_np = self.entropy_each.cpu().detach().numpy()
            loss_np = self.loss_each.cpu().detach().numpy()
            n_bins = 50
            bins = np.linspace(entropy_np.min(), entropy_np.max(), n_bins)
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
            plt.savefig(os.path.join(self.experiment_path, "loss_vs_entropy_epoch_{}.png".format(self.epoch)))
        elif self.mode == "decoder":
            output_np = (2 * self.output[self.idx].permute(0, 2, 3, 1).detach().cpu().numpy() - 1).astype(np.uint8)
            data_np = self.data[self.idx].permute(0, 2, 3, 1).detach().cpu().numpy()
            for i in range(0, 4):
                plt.subplot(4, 4, i+1)
                plt.imshow(data_np[i])
                plt.title(str(self.epoch))
                plt.subplot(4, 4, 4+i+1)
                plt.imshow(output_np[i])
            for i in range(4, 8):
                plt.subplot(4, 4, 4+i+1)
                plt.imshow(data_np[i])
                plt.subplot(4, 4, 8+i+1)
                plt.imshow(output_np[i])
            plt.savefig(os.path.join(self.experiment_path, "reconstruction_examples_epoch_{}.png".format(self.epoch)))
