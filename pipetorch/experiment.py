import numpy as np
import os
import torch
from torch.utils import data
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pipetorch import callback_functions
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from drawnow import drawnow
from utils.helper_functions import weighted_loss, Oracle, dataset2split_loaders
from time import sleep


class Experiment(object):
    def __init__(self, name: str,
                 models: dict,
                 dataset, batch_size,
                 loss_classes: dict,
                 loss_calc_func,
                 optimizer,
                 experiments_path,
                 scheduler=None,
                 data_preprocess_function=None,
                 phases=None,
                 mode="classifier"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.models = {k: model.to(self.device) for k, model in models.items()}
        self.dataset = dataset
        self.batch_size = batch_size
        split = [0.9, 0.1]
        data_loaders, split_datasets, split = dataset2split_loaders(dataset, batch_size, split)
        self.n_classes = len(data_loaders[0].dataset.dataset.classes)
        self.oracle = Oracle(split_datasets[0])
        self.n_queries = 100
        partial_dataset = self.oracle.random_query(self.n_queries)
        self.data_loaders = {"train": self.to_dataloader(partial_dataset, self.batch_size), "eval": data_loaders[1]}
        self.loss_functions = {k: v(reduction='none') for k, v in loss_classes.items()}
        self.loss_calc_func = loss_calc_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mode = mode
        self.epoch = 0
        self.experiment_path = os.path.join(experiments_path, self.name + "_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        if phases is None:
            phases = ["train", "eval"]
        if not os.path.exists(self.experiment_path):
            os.mkdir(self.experiment_path)
            for phase in phases:
                os.mkdir(os.path.join(self.experiment_path, phase))
        self.writers = {phase: SummaryWriter(os.path.join(self.experiment_path, phase)) for phase in phases}
        print("Experiment dir: {}".format(self.experiment_path))
        self.data_preprocess_function = data_preprocess_function
        self.t0 = datetime.now()
        self.stopping_criteria = 1-1e-4
        self.is_stop = False

    def write2tensorboard(self, writer, output, target, loss, index):
        writer.add_scalar('Metric/Loss', loss.item(), index)
        if self.mode == "classifier":
            accuracy = callback_functions.accuracy(output, target).item()
            if accuracy > self.stopping_criteria:
                self.is_stop = True
            recall, precision, f1_score = callback_functions.recall_precision_f1(output, target)
            writer.add_scalar('Metric/Model Entropy', Categorical(logits=output).entropy().mean().item(), index)
            # writer.add_scalar('Metric/Model Temperature', 1/output[1].mean().item(), index)
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

    @staticmethod
    def calc_epoch_idx(epoch, batches_in_epoch, batch_idx):
        return epoch * batches_in_epoch + batch_idx

    @staticmethod
    def to_dataloader(dataset, batch_size):
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def dataset_calc(self, data_loader, n_labels, n_runs=20):
        {model.train() for model in self.models.values()}
        dataset_length = len(data_loader.dataset.indices)
        dataloader_length = len(data_loader)
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
                        loss, output = self.loss_calc_func(self.models, data, target, self.loss_functions, self.data_preprocess_function)
                        loss = weighted_loss(self.n_classes, target, loss, reduction='none')
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
        return outputs, mean, std, entropy_mean, entropy_std, losses_mean, losses_std

    def run_single_epoch(self, phase: str):
        assert phase in self.writers.keys(), "Expected one of the phases: {}, but got phase: {}.".format(self.writers.keys(), phase)
        data_loader = self.data_loaders[phase]
        {model.train() for model in self.models.values()}
        if phase.lower() == "train":
            self.optimizer.zero_grad(set_to_none=True)
        with tqdm(total=len(data_loader)) as pbar:
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                batch_size = len(data)
                loss, output = self.loss_calc_func(self.models, data, target, self.loss_functions, self.data_preprocess_function)
                loss = weighted_loss(self.n_classes, target, loss)
                if phase.lower() == "train":
                    loss.backward()
                    # optimizer updates weights
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                if phase.lower() == "train":
                    self.idx = self.calc_epoch_idx(self.epoch, len(data_loader.batch_sampler), batch_idx)
                self.write2tensorboard(self.writers[phase], output, target, loss, self.idx)
                self.update_tqdm(pbar, text=phase)

            if phase.lower() == "train":
                if self.scheduler is not None:
                    self.scheduler.step()

    def run(self, delta_epochs_to_save_checkpoint=100):
        os.popen(r'tensorboard --logdir=' + self.experiment_path)
        while True:
            while not self.is_stop:
                self.epoch += 1
                self.run_single_epoch(phase="train")
                self.eval()
                if self.epoch % delta_epochs_to_save_checkpoint == 0:
                    self.save()
            self.is_stop = False
            self.save()
            outputs, mean, std, entropy_mean, entropy_std, losses_mean, losses_std = self.dataset_calc(self.to_dataloader(self.oracle.remaining_dataset, self.batch_size), n_labels=self.n_classes, n_runs=20)
            score = entropy_std
            indices = torch.topk(score, self.n_queries).indices
            partial_dataset = self.oracle.query(indices)
            self.data_loaders["train"] = self.to_dataloader(partial_dataset, self.batch_size)
            print(self.oracle)

    @torch.no_grad()
    def eval(self):
        # {model.eval() for model in self.models.values()}
        self.run_single_epoch(phase="eval")

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
