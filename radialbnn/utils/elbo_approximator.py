from typing import Any

import torch.nn as nn
from torch import Tensor

from ..radial_layer import RadialLayer


def elbo_approximator(model: nn.Module) -> nn.Module:

    """Adds ability to calculate ELBO to a nn.Module.

    Parameters
    ----------
    model : nn.Module
        Model to calculate the ELBO loss for.

    Returns
    -------
    model : nn.Module
        Model with ELBO calculation functionality.
    """

    def kl_divergence(self) -> Tensor:

        """Calculates the KL Divergence for each RadialLayer.

        The total KL Divergence is calculated by iterating through the
        RadialLayers in the model. KL Divergence for each module is
        calculated as the difference between the log_posterior and the
        log_prior.

        Returns
        -------
        kl : Tensor
            Total KL Divergence.
        """

        kl = 0
        for module in self.modules():
            if isinstance(module, RadialLayer):
                kl += module.kl_divergence

        return kl

    # add `kl_divergence` to the model
    setattr(model, 'kl_divergence', kl_divergence)

    def elbo(self,
             inputs: Tensor,
             targets: Tensor,
             criterion: Any,
             n_samples: int = 1,
             w_complexity: float = 1.0) -> Tensor:

        """Samples the ELBO loss for a given batch of data.

        The ELBO loss for a given batch of data is the sum of the
        complexity cost and a data-driven cost. Monte Carlo sampling is
        used in order to calculate a representative loss.

        Parameters
        ----------
        inputs : Tensor
            Inputs to the model.
        targets : Tensor
            Target outputs of the model.
        criterion : Any
            Loss function used to calculate data-dependant loss.
        n_samples : int
            Number of samples to use
        w_complexity : float
            Weighting for the complexity term of the loss.

        Returns
        -------
        Tensor
            Value of the ELBO loss for the given data.
        """

        loss = 0
        for sample in range(n_samples):
            outputs = self(inputs)
            loss += criterion(outputs, targets)
            loss += w_complexity * self.kl_divergence()

        return loss / n_samples

    # add `elbo` to the model
    setattr(model, 'elbo', elbo)

    return model
