from .autoencoders import *
from .vaes import *
from .classifier import Classifier

__all__ = [
    "select_architecture", 
    "AutoencoderDataParallel", 
    "VAEDataParallel", 
    "Classifier"
]

def select_architecture(
        pipeline, n_classes, **kwargs
    ):
    
    if pipeline.upper() in ["AE", "AUTOENCODER"]:
        return select_ae_architecture(**kwargs)
    
    elif pipeline.upper() == "VAE":
        return select_vae_architecture(n_classes=n_classes, **kwargs)
    
    else:
        raise ValueError