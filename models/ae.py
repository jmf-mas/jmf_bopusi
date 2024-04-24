import torch.nn as nn
import gzip
import pickle
import torch
import torch.nn.functional as F
from .base import BaseModel, BaseAEModel
from typing import Tuple, List

class AECleaning(BaseAEModel):

    def __init__(self, params):
        super(AECleaning, self).__init__(params)
        if "cifar" in params.dataset_name or "svhn" in params.dataset_name or "mnist" in params.dataset_name:
            self.enc = nn.Sequential(
                nn.Linear(params.in_features, 512),
                nn.Linear(512, 256),
                nn.Dropout(params.dropout),
                nn.Linear(256, 128),
                nn.Dropout(params.dropout),
                nn.Linear(128, 64),
                nn.Linear(64, 32),
                nn.Dropout(params.dropout),
                nn.Linear(32, 16),
                nn.Linear(16, 8)
            )

        else:
            self.enc = nn.Sequential(
                nn.Linear(params.in_features, 64),
                nn.Linear(64, 32),
                nn.Dropout(params.dropout),
                nn.Linear(32, 16),
                nn.Dropout(params.dropout),
                nn.Linear(16, 8)
            )
        self.name = params.dataset_name

        if "cifar" in params.dataset_name or "svhn" in params.dataset_name or "mnist" in params.dataset_name:
            self.dec = nn.Sequential(
                nn.Linear(8, 16),
                nn.Linear(16, 32),
                nn.Dropout(params.dropout),
                nn.Linear(32, 64),
                nn.Linear(64, 128),
                nn.Dropout(params.dropout),
                nn.Linear(128, 256),
                nn.Dropout(params.dropout),
                nn.Linear(256, 512),
                nn.Linear(512, params.in_features)
            )
        else:
            self.dec = nn.Sequential(
                nn.Linear(8, 16),
                nn.Linear(16, 32),
                nn.Dropout(params.dropout),
                nn.Linear(32, 64),
                nn.Dropout(params.dropout),
                nn.Linear(64, params.in_features)
            )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode

    def compute_l2_loss(self):
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        return l2_lambda * l2_norm



class AEDetecting(BaseAEModel):
    """
    Implements a basic Deep Auto Encoder
    """

    def __init__(self, params, enc_layers: list = None, dec_layers: list = None):
        latent_dim = params.ae_latent_dim
        cond = (enc_layers and dec_layers) or (params.dataset_name and params.in_features)
        assert cond, "please provide either the name of the dataset and the number of features or specify the encoder " \
                     "and decoder layers "
        super(AEDetecting, self).__init__(params)
        if not enc_layers or not dec_layers:
            enc_layers, dec_layers = AEDetecting.resolve_layers(params.in_features,
                                                                params.dataset_name,
                                                                latent_dim=latent_dim)
        self.latent_dim = dec_layers[0][0]
        self.in_features = enc_layers[-1][1]
        self.encoder = self._make_linear(enc_layers)
        self.decoder = self._make_linear(dec_layers)
        # Randomly initialize the model center and make it learnable
        self.latent_center = nn.Parameter(torch.randn(1, latent_dim))
        self.name = "AutoEncoder"

        self.params = params

    @staticmethod
    def from_dataset(params):
        enc_layers, dec_layers = AEDetecting.resolve_layers(params.in_features, params.dataset_name)
        return AEDetecting(params, enc_layers, dec_layers)

    @staticmethod
    def resolve_layers(in_features: int, dataset_name: str, latent_dim=1):
        if dataset_name == "Arrhythmia":
            enc_layers = [
                (in_features, 10, nn.Tanh()),
                (10, latent_dim, None)
            ]
            dec_layers = [
                (latent_dim, 10, nn.Tanh()),
                (10, in_features, None)
            ]
        elif dataset_name == "Thyroid":
            enc_layers = [
                (in_features, 12, nn.Tanh()),
                (12, 4, nn.Tanh()),
                (4, latent_dim, None)
            ]
            dec_layers = [
                (latent_dim, 4, nn.Tanh()),
                (4, 12, nn.Tanh()),
                (12, in_features, None)
            ]
        elif "kdd" in dataset_name.lower():
            # enc_layers = [
            #     (in_features, 60, nn.Tanh()),
            #     (60, 30, nn.Tanh()),
            #     (30, 10, nn.Tanh()),
            #     (10, latent_dim, None)
            # ]
            # dec_layers = [
            #     (latent_dim, 10, nn.Tanh()),
            #     (10, 30, nn.Tanh()),
            #     (30, 60, nn.Tanh()),
            #     (60, in_features, None)]

            enc_layers = [
                (in_features, 180, nn.Tanh()),
                (180, 30, nn.Tanh()),
                (30, 20, nn.Tanh()),
                (20, latent_dim, None)
            ]
            dec_layers = [
                (latent_dim, 20, nn.Tanh()),
                (20, 30, nn.Tanh()),
                (30, 180, nn.Tanh()),
                (180, in_features, None)]
        elif "iot2023" in dataset_name.lower():
            #     enc_layers = [
            #         (in_features, 120, nn.Tanh()),
            #         (120, 30, nn.Tanh()),
            #         (30, 20, nn.Tanh()),
            #         (20, latent_dim, None)
            #     ]
            #     dec_layers = [
            #         (latent_dim, 20, nn.Tanh()),
            #         (20, 30, nn.Tanh()),
            #         (30, 120, nn.Tanh()),
            #         (120, in_features, None)]
            #     # enc_layers = [
            #     #     (in_features, 60, nn.Tanh()),
            #     #     (60, 30, nn.Tanh()),
            #     #     (30, 10, nn.Tanh()),
            #     #     (10, latent_dim, None)
            #     # ]
            #     # dec_layers = [
            #     #     (latent_dim, 10, nn.Tanh()),
            #     #     (10, 30, nn.Tanh()),
            #     #     (30, 60, nn.Tanh()),
            #     #     (60, in_features, None)]
            #
            enc_layers = [
                (in_features, 120, nn.ReLU()),
                (120, 64, nn.ReLU()),
                (64, 32, nn.ReLU()),
                (32, latent_dim, None)
            ]
            dec_layers = [
                (latent_dim, 32, nn.ReLU()),
                (32, 64, nn.ReLU()),
                (64, 120, nn.ReLU()),
                (120, in_features, None)]
        elif "kitsune" in dataset_name.lower():

            enc_layers = [
                (in_features, 120, nn.ReLU()),
                (120, 64, nn.ReLU()),
                (64, 32, nn.ReLU()),
                (32, latent_dim, None)
            ]
            dec_layers = [
                (latent_dim, 32, nn.ReLU()),
                (32, 64, nn.ReLU()),
                (64, 120, nn.ReLU()),
                (120, in_features, None)]

        elif "toy" in dataset_name.lower():

            enc_layers = [
                (in_features, in_features * 2, nn.ReLU()),
                (in_features * 2, latent_dim, None)
            ]
            dec_layers = [
                (latent_dim, in_features * 2, nn.ReLU()),
                (in_features * 2, in_features, None)]

        else:
            # enc_layers = [
            #     (in_features, in_features, nn.ReLU()),
            #     (in_features, in_features // 2, nn.ReLU()),
            #     (in_features // 2, in_features // 4, nn.ReLU()),
            #     (in_features // 4, in_features // 6, nn.ReLU()),
            #     (in_features // 6, latent_dim, nn.ReLU())
            # ]
            # dec_layers = [
            #     (latent_dim, in_features // 6, nn.ReLU()),
            #     (in_features // 6, in_features // 4, nn.ReLU()),
            #     (in_features // 4, in_features // 2, nn.ReLU()),
            #     (in_features // 2, in_features, nn.ReLU()),
            #     (in_features, in_features, nn.Sigmoid())]

            # enc_layers = [
            #     (in_features, in_features, nn.ReLU()),
            #     (in_features, in_features // 2, nn.ReLU()),
            #     (in_features // 2, in_features // 4, nn.ReLU()),
            #     (in_features // 4, in_features // 6, nn.ReLU()),
            #     (in_features // 6, latent_dim, None)
            # ]
            # dec_layers = [
            #     (latent_dim, in_features // 6, nn.ReLU()),
            #     (in_features // 6, in_features // 4, nn.ReLU()),
            #     (in_features // 4, in_features // 2, nn.ReLU()),
            #     (in_features // 2, in_features, nn.ReLU()),
            #     (in_features, in_features,None)]

            enc_layers = [
                (in_features, in_features, nn.ReLU()),
                (in_features, int(in_features * .83), nn.ReLU()),
                (int(in_features * .83), int(in_features * .66), nn.ReLU()),
                (int(in_features * .66), int(in_features * .50), nn.ReLU()),
                (int(in_features * .50), latent_dim, None)

            ]
            dec_layers = [
                (latent_dim, int(in_features * .50), nn.ReLU()),
                (int(in_features * .50), int(in_features * .66), nn.ReLU()),
                (int(in_features * .66), int(in_features * .83), nn.ReLU()),
                (int(in_features * .83), in_features, nn.Sigmoid()),
                ]

        return enc_layers, dec_layers

    def _make_linear(self, layers: List[Tuple]):
        """
        This function builds a linear model whose units and layers depend on
        the passed @layers argument
        :param layers: a list of tuples indicating the layers architecture (in_neuron, out_neuron, activation_function)
        :return: a fully connected neural net (Sequentiel object)
        """
        net_layers = []
        for in_neuron, out_neuron, act_fn in layers:
            net_layers.append(nn.Linear(in_neuron, out_neuron))
            if act_fn:
                net_layers.append(act_fn)
        return nn.Sequential(*net_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """
        z = self.encoder(x)
        # z = F.normalize(z, p=2, dim=-1)
        output = self.decoder(F.relu(z))
        return z, output

    def get_params(self) -> dict:
        return {
            "in_features": self.in_features,
            "latent_dim": self.latent_dim,
            'robust': self.kwargs['rob'],
            'reg_n': self.kwargs['reg_n'],
            'reg_a': self.kwargs['reg_a']
        }

    def reset(self):
        self.apply(self.weight_reset)

    def weight_reset(self, m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    @staticmethod
    def load_(filename):
        # Load model from file (.pklz)
        with gzip.open(filename, 'rb') as f:
            model = pickle.load(f)
        assert isinstance(model, BaseModel)
        return model

    def save_(self, filename):
        torch.save(self.state_dict(), filename)

