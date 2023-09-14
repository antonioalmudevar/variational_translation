from logging import Logger

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.utils import accuracy
from ..model import print_loss_classifier

__all__ = [
    "train_epoch_classifier", 
    "test_epoch_classifier",
]


def train_epoch_classifier(
    logger: Logger,
    epoch: int, 
    n_epochs: int,
    n_iters_train: int,
    device: torch.device,
    train_loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    print_iters: int=1000,
):
    
    labeled_loss, unlabeled_loss, acc1_labeled, acc1_unlabeled = 0, 0, 0, 0
    model.train()

    for iter, (input, labels) in enumerate(train_loader, start=1):

        #======Data preparation=======
        input = Variable(input).to(
            device=device, dtype=torch.float, non_blocking=True
        )
        labels = labels.to(device=device, non_blocking=True)

        #======Forward=======
        output = model(input)
        iter_loss = model.calc_loss(**output, labels=labels)
        labeled_loss += torch.mean(iter_loss['labeled_loss']).detach()
        unlabeled_loss += torch.mean(iter_loss['unlabeled_loss']).detach()
        acc1_labeled += accuracy(output['preds_labeled'].detach(), labels, topk=(1,))[0][0]
        acc1_unlabeled += accuracy(output['preds_unlabeled'].detach(), labels, topk=(1,))[0][0]

        #======Backward=======
        optimizer.zero_grad()
        torch.mean(iter_loss['total_loss']).backward()
        optimizer.step()

        #======Logs=======
        print_loss_classifier(
            logger=logger,
            train=True, 
            epoch=epoch, 
            n_epochs=n_epochs, 
            iter=iter, 
            n_iters=n_iters_train, 
            labeled_loss=labeled_loss,
            unlabeled_loss=unlabeled_loss,
            acc1_labeled=acc1_labeled,
            acc1_unlabeled=acc1_unlabeled,
            print_iters=print_iters,
        )

    scheduler.step(epoch)



def test_epoch_classifier(
    logger: Logger,
    epoch: int,
    n_epochs: int,
    n_iters_test: int,
    device: torch.device,
    test_loader: DataLoader,
    model: torch.nn.Module,
    print_iters: int=1000,
):
    
    labeled_loss, unlabeled_loss, acc1_labeled, acc1_unlabeled = 0, 0, 0, 0
    model.eval()

    for iter, (input, labels) in enumerate(test_loader, start=1):

        #======Data preparation=======
        input = Variable(input).to(
            device=device, dtype=torch.float, non_blocking=True
        )
        labels = labels.to(device=device, non_blocking=True)

        #======Forward=======
        with torch.no_grad():
            output = model(input)
            iter_loss = model.calc_loss(**output, labels=labels)
            labeled_loss += torch.mean(iter_loss['labeled_loss']).detach()
            unlabeled_loss += torch.mean(iter_loss['unlabeled_loss']).detach()
            acc1_labeled += accuracy(output['preds_labeled'].detach(), labels, topk=(1,))[0][0]
            acc1_unlabeled += accuracy(output['preds_unlabeled'].detach(), labels, topk=(1,))[0][0]

        #======Logs=======
        print_loss_classifier(
            logger=logger,
            train=False, 
            epoch=epoch, 
            n_epochs=n_epochs, 
            iter=iter, 
            n_iters=n_iters_test, 
            labeled_loss=labeled_loss,
            unlabeled_loss=unlabeled_loss,
            acc1_labeled=acc1_labeled,
            acc1_unlabeled=acc1_unlabeled,
            print_iters=print_iters,
        )