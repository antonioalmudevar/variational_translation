from logging import Logger

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from ..model import print_loss_ae, save_epoch_model

__all__ = [
    "train_epoch_ae", 
    "test_epoch_ae",
]


def train_epoch_ae(
    logger: Logger,
    epoch: int, 
    n_epochs: int,
    n_iters_train: int,
    device: torch.device,
    model_dir: str,
    train_loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    save: bool=False,
    print_iters: int=1000,
):
    
    total_loss = 0
    model.train()

    for iter, (input, _) in enumerate(train_loader, start=1):

        #======Data preparation=======
        input = Variable(input).to(
            device=device, dtype=torch.float, non_blocking=True
        )

        #======Forward=======
        x_hat = model(input)
        iter_loss = model.calc_loss(x=input, x_hat=x_hat)
        total_loss += torch.mean(iter_loss).detach()

        #======Backward=======
        optimizer.zero_grad()
        torch.mean(iter_loss).backward()
        optimizer.step()

        #======Logs=======
        print_loss_ae(
            logger=logger,
            train=True, 
            epoch=epoch, 
            n_epochs=n_epochs, 
            iter=iter, 
            n_iters=n_iters_train, 
            total_loss=total_loss,
            print_iters=print_iters,
        )

    scheduler.step(epoch)

    save_epoch_model(
        epoch, model_dir, model, optimizer, scheduler, save
    )



def test_epoch_ae(
    logger: Logger,
    epoch: int,
    n_epochs: int,
    n_iters_test: int,
    device: torch.device,
    test_loader: DataLoader,
    model: torch.nn.Module,
    print_iters: int=1000,
    save_recon: bool=False,
):
    
    total_loss  = 0
    test_recon = []
    model.eval()

    for iter, (input, l_) in enumerate(test_loader, start=1):

        #======Data preparation=======
        input = Variable(input).to(
            device=device, dtype=torch.float, non_blocking=True
        )

        #======Forward=======
        with torch.no_grad():
            x_hat = model(input)
            iter_loss = model.calc_loss(x=input, x_hat=x_hat)
            total_loss += torch.mean(iter_loss).detach()
            if save_recon:
                test_recon.extend(x_hat.cpu().detach())

        #======Logs=======
        print_loss_ae(
            logger=logger,
            train=False, 
            epoch=epoch, 
            n_epochs=n_epochs, 
            iter=iter, 
            n_iters=n_iters_test, 
            total_loss=total_loss,
            print_iters=print_iters,
        )
    
    if save_recon:
        test_recon = torch.stack(test_recon).numpy()

    return {'recon': test_recon}