import pytest

import torch
import torch.nn as nn

from radialbnn import RadialLayer, variational_approximator


class TestVariationalApproximator:

    @pytest.fixture
    def basic_model(self):

        @variational_approximator
        class Model(nn.Module):

            def __init__(self):

                super().__init__()

                self.rl1 = RadialLayer(5, 10)
                self.rl2 = RadialLayer(10, 10)
                self.rl3 = RadialLayer(10, 1)

            def forward(self, x):

                x = x.view(-1, 5)

                x = self.rl1(x)
                x = self.rl2(x)
                x = self.rl3(x)

                return x

        model = Model()
        return model

    def test_kl_divergence(self, basic_model):

        to_feed = torch.ones(3, 5)
        output = basic_model(to_feed)

        ret_kld = basic_model.kl_divergence()

        assert isinstance(ret_kld, torch.Tensor)
        assert ret_kld.numel() == 1

    def test_elbo(self, basic_model):

        to_feed = torch.ones(3, 5)
        output = basic_model(to_feed)

        criterion = nn.MSELoss()
        ret_elbo = basic_model.elbo(to_feed, torch.ones(3, 1), criterion, 5)

        assert isinstance(ret_elbo, torch.Tensor)
        assert ret_elbo.numel() == 1
