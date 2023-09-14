import torch
import torch.nn as nn


class BaseAutoencoder(nn.Module):

    def __init__(
            self, 
            latent_dim, 
            **kwargs
        ):
        super(BaseAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        
        
    #==========Build methods====================
    def _build_model(self):
        raise NotImplementedError


    #==========Forward methods====================
    def encode(self):
        raise NotImplementedError
    

    def decode(sel):
        raise NotImplementedError
    

    def forward(self, x, n_samples=1):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    

    def recons_loss(self, x, x_hat):
        return torch.mean((x-x_hat)**2, axis=(1,2,3))
    

    def calc_loss( self, x, x_hat):
        recons_loss = self.recons_loss(x, x_hat)
        return recons_loss
    


class AutoencoderDataParallel(nn.DataParallel):

    def __init__(self, vae_model):
        super(AutoencoderDataParallel, self).__init__(vae_model)

    def calc_loss(self, x, x_hat):
        return self.module.calc_loss(x, x_hat)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.module.load_state_dict(state_dict, *args, **kwargs)