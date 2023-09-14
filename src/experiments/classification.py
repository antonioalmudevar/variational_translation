import logging
from PIL import Image

from pathlib import Path

from src.helpers import *
from src.utils import *
from .datasets import get_dataset

def run_classification(
    config_data: str, 
    config_model: str, 
    config_training: str,
    freeze_encoder: bool=False,
):

    root = Path(__file__).resolve().parents[2]

    results_dir = root/"results"/config_data/config_model/config_training
    vaes_dir = results_dir/"models"
    logs_path = "{}/logs/classification_{}_{}_{}.log".format(
        root, config_data, config_model, config_training
    )
    setup_default_logging(log_path=logs_path)
    logger = logging.getLogger('train')

    cfg_data, cfg_model, cfg_training = read_configs_model(
        path=root/"configs", 
        config_data=config_data,
        config_model=config_model,
        config_training='sgd-128-1e-3',
    )

    device = get_device()

    train_dataset, n_channels, size, n_classes, _, _, _ = get_dataset(
        train=True, **cfg_data
    )
    train_loader, n_iters_train = get_train_loader(
        train_dataset, batch_size=cfg_training['batch_size']
    )

    test_dataset, _, _, _, _, _, _ = get_dataset(
        train=False, **cfg_data
    )
    test_loader, n_iters_test = get_test_loader(
        test_dataset, batch_size=cfg_training['batch_size']
    )

    vae = get_model(
        device=device,
        input_channels=n_channels, 
        input_shape=size, 
        n_classes=n_classes, 
        **cfg_model
    )
    if freeze_encoder:
        load_last_epoch_model(
            model_dir=vaes_dir, 
            model=vae
        )

    classifier = get_classifier(device, vae=vae, freeze_encoder=freeze_encoder)
    optimizer, scheduler, n_epochs = get_optimizer_scheduler(
        params=classifier.parameters(), 
        cfg_optimizer=cfg_training['optimizer'], 
        cfg_scheduler=cfg_training['scheduler'],
    )
    logger.info("Number of parameters: {}".format(count_parameters(classifier)))

    for epoch in range(1, n_epochs+1):

        #=====Train Epoch==========
        train_epoch(
            pipeline='classifier',
            logger=logger,
            epoch=epoch,
            n_epochs=n_epochs,
            n_iters_train=n_iters_train,
            device=device,
            train_loader=train_loader,
            model=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        
        #=====Test Epoch==========
        test_epoch(
            pipeline='classifier',
            logger=logger,
            epoch=epoch, 
            n_epochs=n_epochs,
            n_iters_test=n_iters_test,
            device=device,
            test_loader=test_loader,
            model=classifier,
        )

        logger.info("")