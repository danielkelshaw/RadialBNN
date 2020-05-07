import pytest

import torch
from radialbnn.distributions import Gaussian


class TestGaussian:

    def test_log_likelihood(self):

        dist = Gaussian(0, 1)
        to_feed = torch.ones(3, 5)

        ret_ll = dist.log_likelihood(to_feed)

        print(ret_ll)

        assert isinstance(ret_ll, torch.Tensor)
