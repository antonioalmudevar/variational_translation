import torch
import torch.nn as nn

from .utils import create_orthogonal_matrix

EPS = 0.001


class BaseVAE(nn.Module):

    def __init__(
            self, 
            n_classes,
            latent_dim, 
            latent_dim_class: int=None,
            hidden_dim: int=None,
            beta_kl: float=1.,
            beta_ent: float=0.,
            mean_norm: float=1.,
            var_norm: float=1.,
        ):
        super(BaseVAE, self).__init__()

        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.latent_dim_class = latent_dim//2 if latent_dim_class is None else latent_dim_class
        self.hidden_dim = hidden_dim if hidden_dim else latent_dim
        self.beta_kl = beta_kl
        self.beta_ent = beta_ent
        if mean_norm:
            self.mean_norm = torch.Tensor([mean_norm])
        else:
            self.mean_norm = nn.Parameter(0.1*torch.randn(1))
        self.logvar_norm = torch.log(torch.Tensor([var_norm]))
        
        self._init_distribution_class(n_classes)
        
        
    #==========Build methods====================
    def _build_model(self):
        raise NotImplementedError
    

    #==========Initialize target latent space====================
    def _create_transform(self):
        transform = torch.eye(self.latent_dim)
        transform[:self.latent_dim_class, :self.latent_dim_class] = \
            create_orthogonal_matrix(self.latent_dim_class)
        return transform
    

    def _init_distribution_class(self, n_classes):
        self.mean_base = torch.cat((
            torch.randn(self.latent_dim_class),
            torch.zeros(self.latent_dim - self.latent_dim_class)
        ))
        self.transforms = torch.stack([
            self._create_transform() for _ in range(n_classes)
        ])
        self.mean_class = torch.stack([
            self.transforms[i]@self.mean_base for i in range(n_classes)
        ])
        self.logvar_class = torch.log(torch.ones((n_classes, self.latent_dim)))


    #==========Forward methods====================
    def encode(self):
        raise NotImplementedError
    

    def decode(sel):
        raise NotImplementedError
    

    def reparameterize(self, mu, logvar, n_samples):
        if n_samples==0:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    

    def forward(self, x, n_samples=1):
        latent_mu, latent_logvar = self.encode(x)
        z = self.reparameterize(latent_mu, latent_logvar, n_samples=n_samples)
        x_hat = self.decode(z)
        #x_hat = torch.sigmoid(x_hat)
        return {'x_hat': x_hat, 'mu': latent_mu, 'logvar': latent_logvar}
    

    def kl_loss(self, mu, logvar, labels):
        one_hot_labels = torch.nn.functional.one_hot(
            labels, self.n_classes
        ).to(dtype=torch.float)
        mu_y = self.mean_norm*torch.matmul(one_hot_labels, self.mean_class)
        var_y = (torch.exp(self.logvar_norm))*torch.matmul(
            one_hot_labels, torch.exp(self.logvar_class)
        )
        
        if len(mu.shape)==3:
            mu_y = mu_y.unsqueeze(2).repeat(1, 1, mu.shape[2])
            var_y = var_y.unsqueeze(2).repeat(1, 1, mu.shape[2])

        kl_loss = -0.5 * torch.sum(
            1 + \
            logvar - torch.log(var_y) - \
            (mu - mu_y)**2 / var_y - \
            logvar.exp() / var_y, 
            axis=1
        )

        if len(mu.shape)==3:
            return kl_loss.mean(axis=1)
        else:
            return kl_loss


    def recons_loss(self, x, x_hat):
        return torch.mean((x-x_hat)**2, axis=(1,2,3))
    

    def entropy_loss(self, logvar):

        ent_loss = - 0.5 * torch.sum(
            1 + torch.log(torch.tensor(2*torch.pi)) + logvar, axis=1
        )

        if len(logvar.shape)==3:
            return ent_loss.mean(axis=1)
        else:
            return ent_loss
    

    def calc_loss( self, x, x_hat, mu, logvar, labels):
        kl_loss = self.kl_loss(mu, logvar, labels)
        entropy_loss = self.entropy_loss(logvar)
        recons_loss = self.recons_loss(x, x_hat)
        total_loss = recons_loss + self.beta_kl*kl_loss + self.beta_ent*entropy_loss
        return {
            'kl_loss': kl_loss,
            'entropy_loss': entropy_loss,
            'recons_loss': recons_loss,
            'total_loss': total_loss
        }


    def generate_samples(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    

    def translate(self, x, labels):
        latent_mu, latent_logvar = self.encode(x)
        mu_origin = torch.stack([
            self._transform_embeds( self.transforms[label.to(torch.int)].T, latent_mu[i]) for i, label in enumerate(labels)
        ])
        mu_translated = torch.stack([
            torch.stack([
                self._transform_embeds(self.transforms[c], mu) for c in range(self.n_classes)
            ]) for mu in mu_origin
        ])
        mu_translated = mu_translated.view(-1, *latent_mu.shape[1:])
        x_translated = self.decode(mu_translated)
        x_translated = x_translated.view(x.shape[0], -1, *x.shape[1:])
        return x_translated
    

    def _transform_embeds(self, transformation, embed):
        if len(embed.shape):
            return transformation @ embed
        elif len(embed.shape)==2:
            new_embed = torch.zeros_like(embed)
            for t in range(embed.shape[1]):
                new_embed[:,t] = transformation @ embed[:,t]
            return new_embed


    #==========Device management methods====================
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mean_norm = self.mean_norm.to(*args, **kwargs)
        self.mean_class = self.mean_class.to(*args, **kwargs)
        self.logvar_norm = self.logvar_norm.to(*args, **kwargs)
        self.logvar_class = self.logvar_class.to(*args, **kwargs)
        return self


    #==========Load & Save model==========
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if not isinstance(self.mean_norm, nn.Parameter):
            state_dict['mean_norm'] = self.mean_norm
        if not isinstance(self.mean_class, nn.Parameter):
            state_dict['mean_class'] = self.mean_class
        if not isinstance(self.logvar_norm, nn.Parameter):
            state_dict['logvar_norm'] = self.logvar_norm
        if not isinstance(self.logvar_class, nn.Parameter):
            state_dict['logvar_class'] = self.logvar_class
        if not isinstance(self.transforms, nn.Parameter):
            state_dict['transforms'] = self.transforms
        return state_dict
    
    
    def load_state_dict(self, state_dict, strict: bool=False):
        if not isinstance(self.mean_norm, nn.Parameter):
            self.mean_norm = state_dict['mean_norm']
            del state_dict['mean_norm']
        if not isinstance(self.mean_class, nn.Parameter):
            self.mean_class = state_dict['mean_class']
            del state_dict['mean_class']
        if not isinstance(self.logvar_norm, nn.Parameter):
            self.logvar_norm = state_dict['logvar_norm']
            del state_dict['logvar_norm']
        if not isinstance(self.logvar_class, nn.Parameter):
            self.logvar_class = state_dict['logvar_class']
            del state_dict['logvar_class']
        if not isinstance(self.transforms, nn.Parameter):
            self.transforms = state_dict['transforms']
            del state_dict['transforms']
        super().load_state_dict(state_dict, strict)
    


class VAEDataParallel(nn.DataParallel):

    def __init__(self, vae_model):
        super(VAEDataParallel, self).__init__(vae_model)

    def calc_loss(self, x, x_hat, mu, logvar, labels):
        return self.module.calc_loss(x, x_hat, mu, logvar, labels)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.module.load_state_dict(state_dict, *args, **kwargs)

    def generate_samples(self, num_samples, device):
        return self.module.generate_samples(num_samples, device)

    def translate(self, x, labels):
        return self.module.translate(x, labels)