import torch.nn as nn

from models.base import BaseModel


class DeepSVDD(BaseModel):
    def __init__(self, params):
        self.params = params
        super(DeepSVDD, self).__init__(params)
        self.D = params.in_features
        self.device = params.device
        self.net = self._build_network()
        self.rep_dim = self.D // 4

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.D, self.D // 2),
            nn.ReLU(),
            nn.Linear(self.D // 2, self.D // 4)
        ).to(self.device)

    def forward(self, X):
        return self.net(X)

    def get_params(self) -> dict:
        return {'D': self.D, 'rep_dim': self.rep_dim}
