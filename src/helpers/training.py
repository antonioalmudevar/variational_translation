import os
import copy
from typing import List
import logging
import logging.handlers

import yaml
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from timm.scheduler import create_scheduler


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


#======Get Device=========================================================
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')


#======Data Loaders=========================================================
NUM_WORKERS = 4

def get_train_loader(
    train_dataset: Dataset,
    batch_size: int=1, 
    num_workers: int=NUM_WORKERS, 
    pin_memory: bool=True,
):

    train_loader = DataLoader(
        dataset=train_dataset, 
        shuffle=True, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    n_iters_train = int(np.ceil(len(train_dataset) / batch_size))

    return train_loader, n_iters_train


def get_test_loader(
    test_dataset: Dataset,
    batch_size: int=1, 
    num_workers: int=NUM_WORKERS, 
    pin_memory: bool=True,
):

    test_loader = DataLoader(
        dataset=test_dataset, 
        shuffle=False, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    n_iters_test = int(np.ceil(len(test_dataset) / batch_size))

    return test_loader, n_iters_test


#======Optimizer & Scheduler=========================================================
OPTIMIZERS = {
    'SGD':      optim.SGD,
    'ADAM':     optim.Adam,
    'ADAMW':    optim.AdamW,
}


def get_optimizer(params, optimizer, base_lr, base_batch_size, batch_size=None, **kwargs):
    batch_size = batch_size or base_batch_size
    lr = base_lr*batch_size/base_batch_size
    assert optimizer.upper() in OPTIMIZERS, "optimizer is not correct"
    return OPTIMIZERS[optimizer.upper()](params, lr=lr, **kwargs)


def get_scheduler(optimizer, cfg_scheduler, updates_per_epoch=0):
    scheduler, n_epochs = create_scheduler(
        args=Struct(**cfg_scheduler),
        optimizer=optimizer,
        updates_per_epoch=updates_per_epoch,
    )
    return scheduler, n_epochs


def get_optimizer_scheduler(params, cfg_optimizer, cfg_scheduler):
    optimizer = get_optimizer(params, **cfg_optimizer)
    scheduler, n_epochs = get_scheduler(optimizer, cfg_scheduler)
    return optimizer, scheduler, n_epochs


#======Dropout and Stochastic Depth=========================================================
def set_dropout(models, dropouts):
    models = models if isinstance(models, List) else [models]
    dropouts = dropouts if isinstance(dropouts, List) else [dropouts]*len(models)
    assert len(models)==len(dropouts), "Different lengths of models and dropouts"
    for model, dropout in zip(models, dropouts):
        for child in model.children():
            if isinstance(child, torch.nn.Dropout):
                child.p = dropout
            set_dropout(child, dropout)


def set_drop_path(models, drop_probs):
    models = models if isinstance(models, List) else [models]
    drop_probs = drop_probs if isinstance(drop_probs, List) else [drop_probs]*len(models)
    assert len(models)==len(drop_probs), "Different lengths of models and dropouts"
    for model, drop_prob in zip(models, drop_probs):
        for child in model.children():
            if isinstance(child, DropPath):
                child.drop_prob = drop_prob
            set_dropout(child, drop_prob)


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


#======Exponential Moving Average=========================================================
class EMA(nn.Module):

    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def update_decay(self, decay):
        self.decay = decay

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


#======Mixed Precision=========================================================
class NotScaler():

    def __init__(self):
        pass

    def backward(self, loss: torch.Tensor):
        loss.backward()

    def step(self, optimizer: optim):
        optimizer.step()
    
    def update(self):
        pass


class MixedScaler(torch.cuda.amp.GradScaler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def backward(self, loss):
        self.scale(loss).backward()

def get_scaler(mixed_precision=False, **kwargs):
    return MixedScaler(**kwargs) if mixed_precision else NotScaler()


#======Setup Loggers=========================================================
class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO, log_path='', restart=False):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        if restart: 
            try:
                os.remove(log_path)
            except OSError:
                pass
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=(1024 ** 2 * 2), backupCount=3
        )
        file_formatter = logging.Formatter("%(asctime)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)


#======Count paramters=========================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#======Read and save config file=========================================================
def read_config(path):
    with open(str(path)+".yaml", 'r') as f:
        return yaml.load(f, yaml.FullLoader)


def save_config(cfg, path):
    with open(str(path)+".yaml", 'w') as f:
        return yaml.dump(cfg, f)


#======Get list of models=========================================================
def get_models_list(
    dir: str, 
    prefix: str,
):
    models = [epoch for epoch in os.listdir(dir) if epoch.startswith(prefix)]
    models_int = sorted([int(epoch[len(prefix):-3]) for epoch in models])
    return [prefix+str(epoch)+'.pt' for epoch in models_int]