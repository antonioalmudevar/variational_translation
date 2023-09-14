import logging

import numpy as np
from pathlib import Path

from src.helpers import *
from src.utils import *
from .datasets import get_dataset


def test_model(
    config_data: str, 
    config_model: str,
    config_training: str,
):

    root = Path(__file__).resolve().parents[2]

    results_dir = root/"results"/config_data/config_model/config_training
    models_dir = results_dir/"models"
    preds_dir = results_dir/"preds"
    Path(preds_dir).mkdir(parents=True, exist_ok=True)

    logs_path = "{}/logs/test_{}_{}_{}.log".format(
        root, config_data, config_model, config_training
    )
    setup_default_logging(log_path=logs_path, restart=True)
    logger = logging.getLogger('test')

    cfg_data, cfg_model, cfg_training = read_configs_model(
        path=root/"configs", 
        config_data=config_data,
        config_model=config_model,
        config_training=config_training,
    )

    device = get_device()

    test_dataset, n_channels, size, n_classes, _, _, is_img = get_dataset(
        train=False, **cfg_data
    )
    test_loader, n_iters_test = get_test_loader(
        test_dataset, batch_size=cfg_training['batch_size']
    )

    model = get_model(
        device=device,
        input_channels=n_channels, 
        input_shape=size, 
        n_classes=n_classes, 
        **cfg_model
    )
    load_epoch_model(
        epoch=cfg_training['scheduler']['epochs'], 
        device=device, 
        model_dir=models_dir, 
        model=model
    )

    #=====Test Epoch==========
    out = test_epoch(
        pipeline=cfg_model['pipeline'],
        logger=logger,
        epoch=1, 
        n_epochs=1,
        n_iters_test=n_iters_test,
        device=device,
        test_loader=test_loader,
        model=model,
        print_iters=-1,
        save_recon=True,
    )

    np.savez(
        preds_dir/"predictions.npz", 
        input=tensor_to_numpy(test_dataset.data),
        recon=out['recon'],
    )