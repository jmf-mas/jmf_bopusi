
import numpy as np
import torch

from .ae import AEDetecting
from .base import BaseModel
from torch import nn

from .gmm import GMM


class DAGMM(BaseModel):

    def __init__(self, params):
        self.params = params
        self.lambda_1 = params.lambda_1
        self.lambda_2 = params.lambda_2
        self.reg_covar = params.reg_covar
        self.ae = None
        self.gmm = None
        self.K = None
        self.latent_dim = None
        self.name = "DAGMM"
        super(DAGMM, self).__init__(params)
        self.cosim = nn.CosineSimilarity()
        self.softmax = nn.Softmax(dim=-1)

    def resolve_params(self):
        latent_dim = self.latent_dim or 1
        if self.params.dataset_name == 'Arrhythmia':
            K = 2
            gmm_layers = [
                (latent_dim + 2, 10, nn.Tanh()),
                (None, None, nn.Dropout(0.5)),
                (10, K, nn.Softmax(dim=-1))
            ]
        elif self.params.dataset_name == "Thyroid":
            K = 2
            gmm_layers = [
                (latent_dim + 2, 10, nn.Tanh()),
                (None, None, nn.Dropout(0.5)),
                (10, K, nn.Softmax(dim=-1))
            ]
        else:
            K = 4
            gmm_layers = [
                (latent_dim + 2, 10, nn.Tanh()),
                (None, None, nn.Dropout(0.5)),
                (10, K, nn.Softmax(dim=-1))
            ]
        self.latent_dim = latent_dim
        self.K = K
        self.ae = AEDetecting.from_dataset(self.params)
        self.gmm = GMM(gmm_layers)

    def forward(self, x: torch.Tensor):
        code = self.ae.encoder(x)
        x_prime = self.ae.decoder(code)
        rel_euc_dist = self.relative_euclidean_dist(x, x_prime)
        cosim = self.cosim(x, x_prime)
        z_r = torch.cat([code, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)], dim=1)
        gamma_hat = self.gmm.forward(z_r)

        return code, x_prime, cosim, z_r, gamma_hat

    def forward_end_dec(self, x: torch.Tensor):
        code = self.ae.encoder(x)
        x_prime = self.ae.decoder(code)
        rel_euc_dist = self.relative_euclidean_dist(x, x_prime)
        cosim = self.cosim(x, x_prime)
        z_r = torch.cat([code, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)], dim=1)

        return code, x_prime, cosim, z_r

    def forward_estimation_net(self, z_r: torch.Tensor):
        gamma_hat = self.gmm.forward(z_r)

        return gamma_hat

    def relative_euclidean_dist(self, x, x_prime):
        return (x - x_prime).norm(2, dim=1) / x.norm(2, dim=1)

    def compute_params(self, z: torch.Tensor, gamma: torch.Tensor):
        N = z.shape[0]
        K = gamma.shape[1]

        # K
        gamma_sum = torch.sum(gamma, dim=0)
        phi = gamma_sum / N
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_mat = mu_z.unsqueeze(-1) @ mu_z.unsqueeze(-2)
        cov_mat = gamma.unsqueeze(-1).unsqueeze(-1) * cov_mat
        cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        K, N, D = gamma.shape[1], z.shape[0], z.shape[1]
        gamma_sum = torch.sum(gamma, dim=0)
        phi_ = gamma_sum / N
        mu_ = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)

        self.phi = phi.data
        self.mu = mu.data
        self.cov_mat = cov_mat

        return phi, mu, cov_mat

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        # jc_res = self.estimate_sample_energy_js(z, phi, mu)

        # Avoid non-invertible covariance matrix by adding small values (eps)
        d = z.shape[1]
        eps = self.reg_covar
        cov_mat = cov_mat + (torch.eye(d)).to(self.device) * eps
        # N x K x D
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # scaler
        inv_cov_mat = torch.cholesky_inverse(torch.cholesky(cov_mat))
        # inv_cov_mat = torch.linalg.inv(cov_mat)
        det_cov_mat = torch.linalg.cholesky(2 * np.pi * cov_mat)
        det_cov_mat = torch.diagonal(det_cov_mat, dim1=1, dim2=2)
        det_cov_mat = torch.prod(det_cov_mat, dim=1)

        exp_term = torch.matmul(mu_z.unsqueeze(-2), inv_cov_mat)
        exp_term = torch.matmul(exp_term, mu_z.unsqueeze(-1))
        exp_term = - 0.5 * exp_term.squeeze()

        # Applying log-sum-exp stability trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]
        exp_result = torch.exp(exp_term - max_val)

        log_term = phi * exp_result
        log_term /= det_cov_mat
        log_term = log_term.sum(axis=-1)

        # energy computation
        energy_result = - max_val.squeeze() - torch.log(log_term + eps)

        if average_energy:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result, pen_cov_mat

    def compute_loss(self, x, x_prime, energy, pen_cov_mat):
        """

        """
        rec_err = ((x - x_prime) ** 2).mean()
        loss = rec_err + self.lambda_1 * energy + self.lambda_2 * pen_cov_mat

        return loss

    def get_params(self) -> dict:
        return {
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "latent_dim": self.ae.latent_dim,
            "K": self.gmm.K
        }