from .base import AutoencoderDataParallel

__all__ = ["AutoencoderDataParallel", "select_ae_architecture"]

def select_ae_architecture(arch, input_channels, **kwargs):
    
    if arch.upper() in ["CONV2D", "CONVOLUTIONAL2D"]:
        from .conv2d import Conv2DAutoencoder
        return Conv2DAutoencoder(input_channels=input_channels, **kwargs)

    else:
        raise ValueError