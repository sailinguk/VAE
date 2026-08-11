import torch
from torch import nn

from vae import Encoder


def test_encoder_supports_no_hidden_layers():
    encoder = Encoder([128], latent_dim=40)
    posterior = encoder(torch.randn(3, 128))
    assert isinstance(encoder.network, nn.Sequential)
    assert len(encoder.network) == 0
    assert posterior.loc.shape == (3, 40)
    assert posterior.scale.shape == (3, 40)


def test_existing_hidden_encoder_layout_is_unchanged():
    encoder = Encoder([8, 5], latent_dim=2)
    assert [type(layer) for layer in encoder.network] == [nn.Linear, nn.ReLU]
    posterior = encoder(torch.randn(4, 8))
    assert posterior.loc.shape == (4, 2)
