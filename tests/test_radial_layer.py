import pytest
import torch
import torch.nn as nn

from radialbnn import RadialLayer


class TestRadialLayer:

    @pytest.fixture
    def radial_layer(self):
        return RadialLayer(5, 1)

    def test_forward(self, radial_layer):

        to_feed = torch.ones(3, 5)

        ret_tens = radial_layer(to_feed)

        assert isinstance(ret_tens, torch.Tensor)
        assert ret_tens.numel() == 3
