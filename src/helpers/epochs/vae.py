from logging import Logger

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from ..model import print_loss_vae, save_epoch_model

__all__ = [
    "train_epoch_vae", 
    "test_epoch_vae",
    "translate_epoch_vae"
]


def train_epoch_vae(
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
    
    total_loss, recons_loss, kl_loss, ent_loss = 0, 0, 0, 0
    model.train()

    for iter, (input, labels) in enumerate(train_loader, start=1):

        #======Data preparation=======
        input = Variable(input).to(
            device=device, dtype=torch.float, non_blocking=True
        )
        labels = labels.to(device=device, non_blocking=True)

        #======Forward=======
        output = model(input)
        iter_loss = model.calc_loss(**output, x=input, labels=labels)
        total_loss += torch.mean(iter_loss['total_loss']).detach()
        kl_loss += torch.mean(iter_loss['kl_loss']).detach()
        ent_loss += torch.mean(iter_loss['entropy_loss']).detach()
        recons_loss += torch.mean(iter_loss['recons_loss']).detach()

        #======Backward=======
        optimizer.zero_grad()
        torch.mean(iter_loss['total_loss']).backward()
        optimizer.step()

        #======Logs=======
        print_loss_vae(
            logger=logger,
            train=True, 
            epoch=epoch, 
            n_epochs=n_epochs, 
            iter=iter, 
            n_iters=n_iters_train, 
            kl_loss=kl_loss,
            ent_loss=ent_loss,
            recons_loss=recons_loss,
            total_loss=total_loss,
            print_iters=print_iters,
        )

    scheduler.step(epoch)

    save_epoch_model(
        epoch, model_dir, model, optimizer, scheduler, save
    )



def test_epoch_vae(
    logger: Logger,
    epoch: int,
    n_epochs: int,
    n_iters_test: int,
    device: torch.device,
    test_loader: DataLoader,
    model: torch.nn.Module,
    print_iters: int=1000,
    save_latent: bool=False,
    save_recon: bool=False,
):
    
    total_loss, recons_loss, kl_loss, ent_loss = 0, 0, 0, 0
    test_mu, test_logvar, test_recon = [], [], []
    model.eval()

    for iter, (input, labels) in enumerate(test_loader, start=1):

        #======Data preparation=======
        input = Variable(input).to(
            device=device, dtype=torch.float, non_blocking=True
        )
        labels = labels.to(device=device, non_blocking=True)

        #======Forward=======
        with torch.no_grad():
            output = model(input, n_samples=0)
            iter_loss = model.calc_loss(**output, x=input, labels=labels)
            total_loss += torch.mean(iter_loss['total_loss']).detach()
            kl_loss += torch.mean(iter_loss['kl_loss']).detach()
            ent_loss += torch.mean(iter_loss['entropy_loss']).detach()
            recons_loss += torch.mean(iter_loss['recons_loss']).detach()
            if save_latent:
                test_mu.extend(output['mu'].cpu().detach())
                test_logvar.extend(output['logvar'].cpu().detach())
            if save_recon:
                test_recon.extend(output['x_hat'].cpu().detach())

        #======Logs=======
        print_loss_vae(
            logger=logger,
            train=False, 
            epoch=epoch, 
            n_epochs=n_epochs, 
            iter=iter, 
            n_iters=n_iters_test, 
            kl_loss=kl_loss,
            ent_loss=ent_loss,
            recons_loss=recons_loss,
            total_loss=total_loss,
            print_iters=print_iters,
        )
    
    if save_latent:
        test_mu = torch.stack(test_mu).numpy()
        test_logvar = torch.stack(test_logvar).numpy()
    if save_recon:
        test_recon = torch.stack(test_recon).numpy()

    return {
        'recon': test_recon, 
        'mu': test_mu, 
        'logvar': test_logvar
    }


def translate_epoch_vae(
    device: torch.device,
    test_loader: DataLoader,
    model: torch.nn.Module,
):
    
    test_translate = []
    model.eval()

    for _, (input, labels) in enumerate(test_loader, start=1):

        #======Data preparation=======
        input = Variable(input).to(
            device=device, dtype=torch.float, non_blocking=True
        )
        labels = labels.to(device=device, non_blocking=True)

        #======Forward=======
        with torch.no_grad():
            output = model.translate(input, labels)
            test_translate.extend(output.cpu())
    
    test_translate = torch.stack(test_translate).numpy()

    return test_translate