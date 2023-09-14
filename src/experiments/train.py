import logging
from PIL import Image

from pathlib import Path

from src.helpers import *
from src.utils import *
from .datasets import get_dataset

def train_model(
    config_data: str, 
    config_model: str, 
    config_training: str,
    restart: bool=False,
):

    root = Path(__file__).resolve().parents[2]

    results_dir = root/"results"/config_data/config_model/config_training

    models_dir = results_dir/"models"
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    imgs_dir = results_dir/"images"
    Path(imgs_dir).mkdir(parents=True, exist_ok=True)

    Path(root/"logs").mkdir(parents=True, exist_ok=True)
    logs_path = "{}/logs/train_{}_{}_{}.log".format(
        root, config_data, config_model, config_training
    )
    setup_default_logging(log_path=logs_path, restart=restart)
    logger = logging.getLogger('train')

    cfg_data, cfg_model, cfg_training = read_configs_model(
        path=root/"configs", 
        config_data=config_data,
        config_model=config_model,
        config_training=config_training,
        model_dir=models_dir,
        save=True,
    )

    device = get_device()

    train_dataset, n_channels, size, n_classes, mean, std, is_img = get_dataset(
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

    model = get_model(
        device=device,
        input_channels=n_channels, 
        input_shape=size, 
        n_classes=n_classes, 
        **cfg_model
    )
    logger.info("Number of parameters: {}".format(count_parameters(model)))

    optimizer, scheduler, n_epochs = get_optimizer_scheduler(
        params=model.parameters(), 
        cfg_optimizer=cfg_training['optimizer'], 
        cfg_scheduler=cfg_training['scheduler'],
    )

    ini_epoch = load_last_epoch_model(
        model_dir=models_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        restart=restart,
    )

    model = get_parallel_model(model=model, pipeline=cfg_model['pipeline'])

    pil_image = Image.fromarray(recover_tensor(
        test_dataset.__getitem__(50)[0], is_img=is_img
    ))
    pil_image.save(imgs_dir/("input.png"))
    
    for epoch in range(ini_epoch+1, n_epochs+1):

        #=====Train Epoch==========
        train_epoch(
            pipeline=cfg_model['pipeline'],
            logger=logger,
            epoch=epoch,
            n_epochs=n_epochs,
            n_iters_train=n_iters_train,
            device=device,
            model_dir=models_dir,
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            save=(epoch%cfg_training['save_epochs']==0) or epoch==n_epochs,
        )
        
        #=====Test Epoch==========
        out = test_epoch(
            pipeline=cfg_model['pipeline'],
            logger=logger,
            epoch=epoch, 
            n_epochs=n_epochs,
            n_iters_test=n_iters_test,
            device=device,
            test_loader=test_loader,
            model=model,
            save_recon=(epoch%1==0),
        )

        logger.info("")

        if epoch%cfg_training['save_epochs']==0:
            pil_image = Image.fromarray(recover_tensor(
                out['recon'][50], mean=mean, std=std, is_img=is_img
            ))
            pil_image.save(imgs_dir/("epoch_"+str(epoch)+"_recon.png"))