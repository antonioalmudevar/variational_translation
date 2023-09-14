import numpy as np
from pathlib import Path

from src.helpers import *
from src.utils import *
from .datasets import get_dataset


def translate_model(
    config_data: str, 
    config_model: str,
    config_training: str,
):

    root = Path(__file__).resolve().parents[2]

    results_dir = root/"results"/config_data/config_model/config_training
    models_dir = results_dir/"models"
    preds_dir = results_dir/"preds"
    Path(preds_dir).mkdir(parents=True, exist_ok=True)

    logs_path = "{}/logs/translate_{}_{}_{}.log".format(
        root, config_data, config_model, config_training
    )
    setup_default_logging(log_path=logs_path, restart=True)
    logger = logging.getLogger('train')

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
    test_loader, __cached__ = get_test_loader(
        test_dataset, batch_size=4
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
    out = translate_epoch(
        pipeline=cfg_model['pipeline'],
        logger=logger,
        device=device,
        test_loader=test_loader,
        model=model,
    )

    np.savez(
        preds_dir/"translations.npz", 
        translations=out
    )