from .base import VAEDataParallel

__all__ = ["VAEDataParallel", "select_vae_architecture"]

def select_vae_architecture(arch, input_channels, **kwargs):

    if arch.upper() in ["CONV2D", "CONVOLUTIONAL2D"]:
        from .conv2d import Conv2DVAE
        return Conv2DVAE(input_channels=input_channels, **kwargs)

    else:
        raise ValueError