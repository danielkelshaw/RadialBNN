import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class Gaussian(nn.Module):

    def __init__(self, mu: float, sigma: float) -> None:

        """Gaussian Prior.

        Parameters
        ----------
        mu : Tensor
            Mean of the distribution.
        sigma : Tensor
            Std of the distribution.
        """

        super().__init__()

        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

        self.dist = torch.distributions.Normal(mu, sigma)

    def log_likelihood(self, w: Tensor) -> Tensor:

        """Log Likelihood for each weight sampled from the distribution.

        Calculates the Gaussian log likelihood of the sampled weight
        given the the current mean, mu, and standard deviation, sigma:
            LL = -log((2pi * sigma^2)^0.5) - 0.5(w - mu)^2 / sigma^2

        Returns
        -------
        Tensor
            Gaussian log likelihood for the weights sampled.
        """

        log_const = np.log(np.sqrt(2 * np.pi))
        log_exp = ((w - self.mu) ** 2) / (2 * self.sigma ** 2)
        log_posterior = -log_const - torch.log(self.sigma) - log_exp

        return log_posterior.sum()
