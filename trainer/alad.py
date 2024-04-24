from models.alad import ALAD
from .base import BaseTrainer
from .dataset import TabularDataset
import torch
import torch.nn as nn
from tqdm import trange
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

class TrainerALAD(BaseTrainer):
    def __init__(self, params):
        self.params = params
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim_ge, self.optim_d = None, None
        self.model = self.params.model
        self.model.to(params.device)
        super(TrainerALAD, self).__init__(params)


    def train_iter(self, sample):
        pass

    def score(self, sample):
        _, feature_real = self.model.D_xx(sample, sample)
        _, feature_gen = self.model.D_xx(sample, self.model.G(self.model.E(sample)))
        return torch.linalg.norm(feature_real - feature_gen, 2, keepdim=False, dim=1)

    def set_optimizer(self):
        self.optim_ge = optim.Adam(
            list(self.model.G.parameters()) + list(self.model.E.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optim_d = optim.Adam(
            list(self.model.D_xz.parameters()) + list(self.model.D_zz.parameters()) + list(
                self.model.D_xx.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )

    def train_iter_dis(self, X):

        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.model(X)
        # Compute loss
        # Discriminators Losses
        loss_dxz = self.criterion(out_truexz, y_true) + self.criterion(out_fakexz, y_fake)
        loss_dzz = self.criterion(out_truezz, y_true) + self.criterion(out_fakezz, y_fake)
        loss_dxx = self.criterion(out_truexx, y_true) + self.criterion(out_fakexx, y_fake)
        loss_d = loss_dxz + loss_dzz + loss_dxx

        return loss_d

    def train_iter_gen(self, X):
        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.model(X)
        # Generator losses
        loss_gexz = self.criterion(out_fakexz, y_true) + self.criterion(out_truexz, y_fake)
        loss_gezz = self.criterion(out_fakezz, y_true) + self.criterion(out_truezz, y_fake)
        loss_gexx = self.criterion(out_fakexx, y_true) + self.criterion(out_truexx, y_fake)
        cycle_consistency = loss_gexx + loss_gezz
        loss_ge = loss_gexz + cycle_consistency

        return loss_ge

    def train(self):
        dataset = TabularDataset(self.params.data)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True,
                                 num_workers=self.params.num_workers)
        self.model.train()
        for epoch in range(self.n_epochs):
            ge_losses, d_losses = 0, 0
            with trange(len(data_loader)) as t:

                for batch in data_loader:
                    data = batch['data'].to(self.params.device)
                    X_dis, X_gen = data, data.clone().to(self.device).float()
                    # Forward pass

                    # Cleaning gradients
                    self.optim_d.zero_grad()
                    loss_d = self.train_iter_dis(X_dis)
                    # Backward pass
                    loss_d.backward()
                    self.optim_d.step()

                    # Cleaning gradients
                    self.optim_ge.zero_grad()
                    loss_ge = self.train_iter_gen(X_gen)
                    # Backward pass
                    loss_ge.backward()
                    self.optim_ge.step()

                    # Journaling
                    d_losses += loss_d.item()
                    ge_losses += loss_ge.item()
                    t.set_postfix(
                        ep=epoch + 1,
                        loss_d='{:05.4f}'.format(loss_d),
                        loss_ge='{:05.4f}'.format(loss_ge),
                    )
                    t.update()

    def eval(self, dataset: DataLoader):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            for row in dataset:
                X = row['data'].to(self.params.device)
                X = X.to(self.device).float()
                loss_d = self.train_iter_dis(X)
                loss_ge = self.train_iter_gen(X)
                loss += loss_d.item() + loss_ge.item()
        self.model.train()
        return loss