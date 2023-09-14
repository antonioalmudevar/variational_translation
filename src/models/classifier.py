import torch
import torch.nn as nn

from .vaes.base import BaseVAE


class Classifier(nn.Module):

    def __init__(
            self, 
            vae: BaseVAE,
            freeze_encoder: bool=True,
        ):
        super(Classifier, self).__init__()

        self.vae = vae
        if freeze_encoder:
            for param in self.vae.parameters():
                param.requires_grad = False

        self.latent_dim = vae.latent_dim
        self.latent_dim_class = vae.latent_dim_class
        self.n_classes = vae.n_classes

        self.last_layer_labeled = nn.Linear(
            self.latent_dim_class, vae.n_classes
        )
        self.last_layer_unlabeled = nn.Linear(
            self.latent_dim-self.latent_dim_class, vae.n_classes
        )


    def get_embeds(self, x):
        latent_mu, _ = self.vae.encode(x)
        mu_labeled = latent_mu[:,:self.latent_dim_class]
        mu_unlabeled = latent_mu[:,self.latent_dim_class:]
        return mu_labeled, mu_unlabeled
    

    def get_logits(self, mu_labeled, mu_unlabeled):
        logits_labeled = self.last_layer_labeled(mu_labeled)
        logits_unlabeled = self.last_layer_unlabeled(mu_unlabeled)
        return logits_labeled, logits_unlabeled
    

    def get_preds(self, logits_labeled, logits_unlabeled):
        preds_labeled = logits_labeled.softmax(dim=-1)
        preds_unlabeled = logits_unlabeled.softmax(dim=-1)
        return preds_labeled, preds_unlabeled


    def forward(self, x):
        mu_labeled, mu_unlabeled = self.get_embeds(x)
        logits_labeled, logits_unlabeled = self.get_logits(mu_labeled, mu_unlabeled)
        preds_labeled, preds_unlabeled = self.get_preds(logits_labeled, logits_unlabeled)
        return {'preds_labeled': preds_labeled, 'preds_unlabeled': preds_unlabeled}
    

    def class_loss(self, labels, preds):
        one_hot = torch.zeros((labels.shape[0], self.n_classes)).to(labels.get_device())
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        return -torch.sum(one_hot*torch.log(preds), axis=-1)
    

    def calc_loss(self, preds_labeled, preds_unlabeled, labels):
        labeled_loss = self.class_loss(labels, preds_labeled)
        unlabeled_loss = self.class_loss(labels, preds_unlabeled)
        return {
            'labeled_loss': labeled_loss, 
            'unlabeled_loss': unlabeled_loss,
            'total_loss': labeled_loss + unlabeled_loss,
        }