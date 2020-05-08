import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from radialbnn.distributions import Gaussian


class RadialLayer(nn.Module):

    """Radial Linear Layer.

    Implementation of a Radial Linear Layer as described in 'Radial
    Bayesian Neural Networks: Beyond Discrete Support In Large-Scale
    Bayesian Deep Learning'.
    """

    def __init__(self, in_features: int, out_features: int) -> None:

        """Radial Linear Layer.

        Parameters
        ----------
        in_features : int
            Number of features to feed into the layer.
        out_features : int
            Number of features produced by the layer.
        """

        super().__init__()

        w_mu = torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        w_rho = torch.empty(out_features, in_features).uniform_(-5.0, -4.0)

        self.w_mu = nn.Parameter(w_mu)
        self.w_rho = nn.Parameter(w_rho)

        self.prior = Gaussian(0, 1)
        self.epsilon_normal = torch.distributions.Normal(0, 1)

        self.kl_divergence = 0.0

    def forward(self, x: Tensor) -> Tensor:

        """Calculates the forward pass through the linear layer.

        Parameters
        ----------
        x : Tensor
            Inputs to the Radial Linear Layer.

        Returns
        -------
        Tensor
            Output from the Radial Linear Layer.
        """

        w_std = torch.log(1 + torch.exp(self.w_rho))

        eps_mfvi = self.epsilon_normal.sample(self.w_mu.size())
        eps_norm = torch.norm(eps_mfvi, p=2, dim=1).unsqueeze(1)
        r_mfvi = torch.randn(1)

        w = torch.addcmul(self.w_mu, eps_mfvi / eps_norm, r_mfvi)

        log_posterior = -torch.sum(torch.log(w_std))
        log_prior = self.prior.log_likelihood(w)
        self.kl_divergence = log_posterior - log_prior

        return F.linear(x, w)
