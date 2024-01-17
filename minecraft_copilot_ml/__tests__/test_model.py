import torch

from minecraft_copilot_ml.model import UNet3D


def test_model() -> None:
    model = UNet3D(10)
    assert model is not None


def test_model_forward() -> None:
    n_channels = 10
    width = 16
    height = 16
    length = 16
    model = UNet3D(n_channels)
    x = torch.randn(1, n_channels, width, height, length, dtype=torch.float32)
    y = model(x)
    assert y.shape == (1, n_channels, width, height, length)
