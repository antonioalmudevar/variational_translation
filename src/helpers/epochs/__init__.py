def train_epoch(pipeline, **kwargs):
    if pipeline.upper() in ["AE", "AUTOENCODER"]:
        from .ae import train_epoch_ae
        train_epoch_ae(**kwargs)
    elif pipeline.upper() == "VAE":
        from .vae import train_epoch_vae
        train_epoch_vae(**kwargs)
    elif pipeline.upper() == "CLASSIFIER":
        from .classifiier import train_epoch_classifier
        train_epoch_classifier(**kwargs)
    

def test_epoch(pipeline, **kwargs):
    if pipeline.upper() in ["AE", "AUTOENCODER"]:
        from .ae import test_epoch_ae
        return test_epoch_ae(**kwargs)
    elif pipeline.upper() == "VAE":
        from .vae import test_epoch_vae
        return test_epoch_vae(**kwargs)
    elif pipeline.upper() == "CLASSIFIER":
        from .classifiier import test_epoch_classifier
        test_epoch_classifier(**kwargs)
    

def translate_epoch(pipeline, logger, **kwargs):
    if pipeline.upper() in ["AE", "AUTOENCODER"]:
        logger.info("Translation cannot be done for autoencoder pipeline")
    elif pipeline.upper() == "VAE":
        from .vae import translate_epoch_vae
        return translate_epoch_vae(**kwargs)
    elif pipeline.upper() == "CLASSIFIER":
        logger.info("Translation cannot be done for classifier pipeline")