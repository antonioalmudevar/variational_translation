import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseVAE
from ..modules import Conv2DBlock, Conv2DTransposeBlock


class Conv2DVAE(BaseVAE):

    def __init__(
            self, 
            input_channels, 
            input_shape, 
            channel_list,
            **kwargs,
        ):
        super(Conv2DVAE, self).__init__(**kwargs)

        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)
        elif isinstance(input_shape, (list, tuple)):
            input_shape = tuple(input_shape)
        else:
            raise ValueError(
                "Invalid input_shape. Expected an integer, list, or tuple."
            )

        self.input_channels = input_channels
        self.input_shape = input_shape
        self.channel_list = channel_list
        self.num_encoder_layers = len(channel_list)
        self.num_decoder_layers = len(channel_list)
        
        self._build_model(input_channels, channel_list)
        
        
    #==========Build methods====================
    def _build_model(self, input_channels, channel_list):

        encoder_layers = []
        in_channels = input_channels
        for _, out_channels in enumerate(channel_list):
            encoder_layers.append(Conv2DBlock(in_channels, out_channels))
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)
        self.conv_output_size = self._calculate_conv_output_size()
        self.encoder_fc = nn.Linear(self.conv_output_size, self.hidden_dim)

        self.latent_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.latent_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        
        decoder_layers = []
        in_channels = channel_list[-1]
        for _, out_channels in reversed(list(enumerate(channel_list[:-1]))):
            decoder_layers.append(Conv2DTransposeBlock(in_channels, out_channels))
            in_channels = out_channels
        self.decoder = nn.Sequential(*decoder_layers)
        self.decoder_fc = nn.Linear(self.latent_dim, self.conv_output_size)
        self.decoder_output = nn.ConvTranspose2d(
            channel_list[0], input_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )


    def _calculate_conv_output_size(self):
        input_tensor = torch.zeros(1, self.input_channels, *self.input_shape)
        conv_output = self.encoder(input_tensor)
        return int(torch.prod(torch.tensor(conv_output.size())))


    #==========Forward methods====================
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.encoder_fc(x))
        latent_mu = self.latent_mu(x)
        latent_logvar = self.latent_logvar(x)
        return latent_mu, latent_logvar
    

    def decode(self, z):
        x = F.relu(self.decoder_fc(z))
        x = x.view(
            -1, 
            self.channel_list[-1], 
            int(self.input_shape[0] / (2 ** self.num_encoder_layers)), 
            int(self.input_shape[1] / (2 ** self.num_encoder_layers))
        )
        x = self.decoder(x)
        x = self.decoder_output(x)
        return x