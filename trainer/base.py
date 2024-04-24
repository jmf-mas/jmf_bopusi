import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Union
from torch.utils.data.dataloader import DataLoader
from torch import optim
from tqdm import trange
from trainer.dataset import TabularDataset
from utils.patience import Patience


class BaseTrainer(ABC):

    def __init__(self, params):
        self.params = params
        self.device = params.device
        self.batch_size = params.batch_size
        self.n_jobs_dataloader = params.n_jobs_dataloader
        self.n_epochs = params.epochs
        self.lr = params.learning_rate
        self.weight_decay = params.weight_decay
        self.optimizer = self.set_optimizer()
        self.name = 'deep'
        self.model = self.params.model
        self.model.to(params.device)

        patience = params.patience
        self.early_stopper = Patience(patience=patience, use_train_loss=False, model=self.model)

    @abstractmethod
    def train_iter(self, sample: torch.Tensor, **kwargs):
        pass

    @abstractmethod
    def score(self, sample: torch.Tensor):
        pass

    def after_training(self):
        """
        Perform any action after training is done
        """
        pass

    def before_training(self, dataset: DataLoader):
        """
        Optionally perform pre-training or other operations.
        """
        pass

    def set_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.params.weight_decay)

    def train(self):
        dataset = TabularDataset(self.params.data)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)

        self.model.train()
        for epoch in range(self.params.epochs):
            epoch_loss = 0.0
            counter = 1

            with trange(len(data_loader)) as t:
                for batch in data_loader:
                    data = batch['data'].to(self.params.device)
                    self.optimizer.zero_grad()
                    loss = self.train_iter(data)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:.3f}'.format(epoch_loss / counter),
                        epoch=epoch + 1
                    )
                    t.update()
                    counter += 1

        self.after_training()

    def eval(self, dataset: DataLoader):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            for row in dataset:
                data = row['data'].to(self.params.device)
                loss += self.train_iter(data)
        self.model.train()

        return loss

    def _eval(self, dataset: DataLoader):
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in dataset:
                X = row['data'].to(self.params.device)
                y = row['target'].to(self.params.device)
                X = X.to(self.device).float()
                score = self.score(X)
                y_true.append(y.cpu().numpy())
                scores.append(score.cpu().numpy())
        self.model.train()

        y_true, scores = np.concatenate(y_true, axis=0), np.concatenate(scores, axis=0)
        # _estimate_threshold_metrics

        accuracy, precision, recall, f_score, roc, avgpr = _estimate_threshold_metrics(scores, y_true,
                                                                                       optimal=False)

        return {k: round(v, 3) for k, v in
                dict(accuracy=accuracy,
                     precision=precision, recall=recall, f_score=f_score, avgpr=avgpr, proc1p=roc).items()}

    def test(self, data):
        test_set = TabularDataset(data)
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, shuffle=True,
                                 num_workers=self.params.num_workers)
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in test_loader:
                X = row['data'].to(self.params.device)
                y = row['target'].to(self.params.device)
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())
        self.model.train()

        return np.array(y_true), np.array(scores)

    def test_return_all(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores, xs = [], [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row[0], row[1]
                X = X.to(self.device).float()
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())
                xs.extend(X.cpu().tolist())
        self.model.train()

        return np.array(y_true), np.array(scores), np.array(xs)

    def get_params(self) -> dict:
        return {
            "learning_rate": self.lr,
            "epochs": self.n_epochs,
            "batch_size": self.batch_size,
            **self.model.get_params()
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)

class TrainerBaseShallow(ABC):
    def __init__(self, params):
        self.params = params
        self.name = "shallow"

    def train(self):
        self.params.model.clf.fit(self.params.data[:, :-1])

    def score(self, sample):
        return self.params.model.clf.predict(sample)

    def test(self, X):
        score = self.score(X)
        y_pred = np.where(score == 1, 0, score)
        return np.where(y_pred == -1, 1, y_pred)
    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)
