from typing import Union
import logging

import torch
from torch import nn

from src.models import *
from .training import read_config, save_config, get_models_list

__all__ = [
    "get_model",
    "get_classifier",
    "get_parallel_model",
    "read_configs_model",
    "load_epoch_model", 
    "load_last_epoch_model",
]


def get_model(
    device: torch.device,
    **kwargs,
):
    return select_architecture(**kwargs).to(device=device, dtype=torch.float)


def get_classifier(
    device: torch.device,
    **kwargs,
):
    return Classifier(**kwargs).to(device=device, dtype=torch.float)


def get_parallel_model(
    model: nn.Module,
    pipeline: str,
):
    if pipeline.upper() in ["AE", "AUTOENCODER"]:
        return AutoencoderDataParallel(model)
    elif pipeline.upper() == "VAE":
        return VAEDataParallel(model)


def read_configs_model(
    path,
    config_data, 
    config_model, 
    config_training,
    model_dir: str=None, 
    save: bool=False,
):

    if save and model_dir is None:
        raise ValueError

    cfg_data = read_config(path/"data"/config_data)
    cfg_model = read_config(path/"models"/config_model)
    cfg_training = read_config(path/"training"/config_training)

    if save:
        save_config(cfg_data, model_dir/"config_data")
        save_config(cfg_model, model_dir/"config_model")
        save_config(cfg_training, model_dir/"config_training")

    return cfg_data, cfg_model, cfg_training
   

def load_last_epoch_model(
    model_dir: str,
    model: Union[nn.Module, nn.DataParallel]=None,
    optimizer: torch.optim.Optimizer=None,
    scheduler: torch.optim.lr_scheduler._LRScheduler=None,
    restart: bool=False
):
    previous_models = get_models_list(model_dir, 'epoch_')
    if len(previous_models)>0 and not restart:
        checkpoint = torch.load(model_dir/previous_models[-1])
        if model is not None:
            model.load_state_dict(checkpoint['model'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['epoch']
    else:
        return 0


def load_epoch_model(
    epoch: int,
    device: torch.device,
    model_dir: str,
    model: Union[nn.Module, nn.DataParallel]=None,
    optimizer: torch.optim.Optimizer=None,
    scheduler: torch.optim.lr_scheduler._LRScheduler=None,
):
    epoch_path = model_dir/("epoch_"+str(epoch)+".pt")
    if model is not None:
        model.load_state_dict(
            torch.load(epoch_path, map_location=device)['model']
        )
    if optimizer is not None:
        optimizer.load_state_dict(
            torch.load(epoch_path, map_location=device)['optimizer']
        )
    if scheduler is not None:
        scheduler.load_state_dict(
            torch.load(epoch_path, map_location=device)['scheduler']
        )


def save_epoch_model(
    epoch: int,
    model_dir: str,
    model: Union[nn.Module, nn.DataParallel],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    save: bool=True,
):
    if save:
        epoch_path = model_dir/("epoch_"+str(epoch)+".pt")
        checkpoint = {
            'epoch':        epoch,
            'model':        model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'scheduler':    scheduler.state_dict(),
        }
        torch.save(checkpoint, epoch_path)


def print_loss_ae(
    logger: logging.Logger,
    train: bool,
    epoch: int, 
    n_epochs: int, 
    iter: int, 
    n_iters: int, 
    total_loss: float,
    print_iters: int=1000
):
    if print_iters>0 and (iter%print_iters==0 or iter==n_iters):
        train = 'Train' if train else 'Test'
        logger.info("{} Loss - Epoch {}/{} - Iteration {}/{}\t| Total: {:.4f}".format(
            train, epoch, n_epochs, iter, n_iters, total_loss/iter
        ))


def print_loss_vae(
    logger: logging.Logger,
    train: bool,
    epoch: int, 
    n_epochs: int, 
    iter: int, 
    n_iters: int, 
    kl_loss: float,
    ent_loss: float,
    recons_loss: float,
    total_loss: float,
    print_iters: int=1000
):
    if print_iters>0 and (iter%print_iters==0 or iter==n_iters):
        train = 'Train' if train else 'Test'
        logger.info("{} Loss - Epoch {}/{} - Iteration {}/{}\t| KL: {:.4f}\t| Entropy: {:.4f}\t| Recons: {:.4f}\t| Total: {:.4f}".format(
            train, epoch, n_epochs, iter, n_iters, kl_loss/iter, ent_loss/iter, recons_loss/iter, total_loss/iter
        ))


def print_loss_classifier(
    logger: logging.Logger,
    train: bool,
    epoch: int, 
    n_epochs: int, 
    iter: int, 
    n_iters: int, 
    labeled_loss: float,
    unlabeled_loss: float,
    acc1_labeled: float,
    acc1_unlabeled: float,
    print_iters: int=1000
):
    if print_iters>0 and (iter%print_iters==0 or iter==n_iters):
        train = 'Train' if train else 'Test'
        logger.info("{} - Epoch {}/{}\t| Lab L: {:.4f}\t| Unlab L: {:.4f}\t| Lab Acc: {:.4f}\t| Unlab Acc: {:.4f}".format(
            train, epoch, n_epochs, labeled_loss/iter, unlabeled_loss/iter, acc1_labeled/iter, acc1_unlabeled/iter
        ))