from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
from tqdm import trange
import torch.nn.functional as F

from models.svdd import DeepSVDD
from trainer.base import BaseTrainer
from trainer.dataset import TabularDataset


class TrainerSVDD(BaseTrainer):

    def __init__(self, params):
        self.params = params
        self.model = self.params.model
        self.model.to(params.device)
        super(TrainerSVDD, self).__init__(params)
        self.c = params.c
        self.R = params.R

    def train_iter(self, sample):
        outputs = self.model(sample)
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        return torch.mean(dist)

    def score(self, sample: torch.Tensor):
        outputs = self.model(sample)
        return torch.sum((outputs - self.c) ** 2, dim=1)

    def before_training(self, dataset: DataLoader):
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print("Initializing center c...")
            self.c = self.init_center_c(dataset)
            print("Center c initialized.")

    def init_center_c(self, train_loader: DataLoader, eps=0.1):
        n_samples = 0
        c = torch.zeros(self.model.rep_dim, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for sample in train_loader:
                # get the inputs of the batch
                X = sample[0]
                X = X.to(self.device).float()
                outputs = self.model(X)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        if c.isnan().sum() > 0:
            raise Exception("NaN value encountered during init_center_c")

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def get_params(self) -> dict:
        return {'c': self.c, 'R': self.R, **self.model.get_params()}


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


def adjust_learning_rate(epoch, total_epochs, only_ce_epochs, learning_rate, optimizer):
    """Adjust learning rate during training.
    Parameters
    ----------
    epoch: Current training epoch.
    total_epochs: Total number of epochs for training.
    only_ce_epochs: Number of epochs for initial pretraining.
    learning_rate: Initial learning rate for training.
    """
    # We dont want to consider the only ce
    # based epochs for the lr scheduler
    epoch = epoch - only_ce_epochs
    drocc_epochs = total_epochs - only_ce_epochs
    # lr = learning_rate
    if epoch <= drocc_epochs:
        lr = learning_rate * 0.001
    if epoch <= 0.90 * drocc_epochs:
        lr = learning_rate * 0.01
    if epoch <= 0.60 * drocc_epochs:
        lr = learning_rate * 0.1
    if epoch <= 0.30 * drocc_epochs:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


class EdgeMLDROCCTrainer(BaseTrainer):
    """
    Trainer class that implements the DROCC algorithm proposed in
    https://arxiv.org/abs/2002.12718
    !!! Most of the code is copied from https://github.com/microsoft/EdgeML/ !!!
    """

    def __init__(self, mu: float = 1., radius: float = 3., gamma: float = 2., **kwargs):
        """Initialize the DROCC Trainer class

        Parameters
        ----------
        model: Torch neural network object
        optimizer: Total number of epochs for training.
        lamda: Weight given to the adversarial loss
        radius: Radius of hypersphere to sample points from.
        gamma: Parameter to vary projection.
        device: torch.device object for device to use.
        """
        super(EdgeMLDROCCTrainer, self).__init__(**kwargs)
        self.lamb = mu
        self.radius = radius
        self.gamma = gamma
        self.only_ce_epochs = 50
        self.ascent_step_size = 0.01
        self.ascent_num_steps = 50

    def score(self, sample: torch.Tensor):
        logits = self.model(sample)
        logits = torch.squeeze(logits, dim=1)
        scores = logits
        return scores

    def train_iter(self, sample: torch.Tensor, **kwargs):
        pass

    def train(self):
        self.model.train()
        dataset = TabularDataset(self.params.data)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True,
                                 num_workers=self.params.num_workers)

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            # Placeholder for the respective 2 loss values
            epoch_adv_loss = torch.tensor([0]).type(torch.float32).to(self.device)  # AdvLoss
            epoch_ce_loss = 0  # Cross entropy Loss
            adjust_learning_rate(epoch, self.n_epochs, self.only_ce_epochs, self.lr, self.optimizer)
            with trange(len(data_loader)) as t:
                for sample in data_loader:
                    # Data processing
                    X, y = sample
                    X = X.to(self.device).float()
                    y = y.to(self.device).float()

                    if len(X) < self.batch_size:
                        break

                    # Reset gradient
                    self.optimizer.zero_grad()

                    # Extract the logits for cross entropy loss
                    logits = self.model(X)
                    logits = torch.squeeze(logits, dim=1)
                    ce_loss = F.binary_cross_entropy_with_logits(logits, y)
                    # Add to the epoch variable for printing average CE Loss
                    epoch_ce_loss += ce_loss

                    if epoch >= self.only_ce_epochs:
                        data = data[y == 0]
                        # AdvLoss
                        adv_loss = self.one_class_adv_loss(data)
                        epoch_adv_loss += adv_loss
                        loss = ce_loss + adv_loss * self.lamb
                    else:
                        # If only CE based training has to be done
                        loss = ce_loss

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:05.3f}'.format(epoch_loss),
                        epoch=epoch + 1
                    )
                    t.update()

    def one_class_adv_loss(self, x_train_data):
        batch_size = len(x_train_data)
        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():
                new_targets = torch.zeros(batch_size, 1).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)

                logits = self.model(x_adv_sampled)
                logits = torch.squeeze(logits, dim=1)

                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)
                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                grad_norm[grad_norm == 0.0] = 10e-10
                grad_normalized = grad / grad_norm

            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 10 == 0:
                # Project the normal points to the set N_i(r)
                h = x_adv_sampled - x_train_data
                norm_h = torch.sqrt(torch.sum(h ** 2,
                                              dim=tuple(range(1, h.dim()))))
                alpha = torch.clamp(norm_h, self.radius,
                                    self.gamma * self.radius).to(self.device)
                # Make use of broadcast to project h
                proj = (alpha / norm_h).view(-1, *[1] * (h.dim() - 1))
                h = proj * h
                x_adv_sampled = x_train_data + h  # These adv_points are now on the surface of hyper-sphere

        adv_pred = self.model(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets + 1))

        return adv_loss
