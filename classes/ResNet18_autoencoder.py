# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 22:22:45 2024

@author: djy41
"""
import torch
import torch.nn as nn

from classes.ResNet18_encoder import ResNet18EncoderFC
from classes.ResNet18_decoder import ResNet18DecoderFC

class ResNet18Autoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(ResNet18Autoencoder, self).__init__()
        self.encoder_c = ResNet18EncoderFC(latent_dim=latent_dim)
        #self.encoder_p = ResNet18EncoderFC(latent_dim=latent_dim)
        self.decoder = ResNet18DecoderFC(latent_dim=latent_dim)   
    
    def forward(self, x):
        c = self.encoder_c(x)
        #u = self.encoder_p(x) 
        #z = torch.concat([c, u], dim=1)
        x_bar = self.decoder(c)
        
        return x_bar, c#, u
#*****************************************************************************

