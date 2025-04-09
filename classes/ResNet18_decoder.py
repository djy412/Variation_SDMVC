# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 22:22:45 2024

@author: djy41
"""
import torch.nn as nn

class ResNet18DecoderFC(nn.Module):
    #def __init__(self, latent_dim=512, initial_shape=(512, 7, 7)):#--- this is for 224x224 images
    #def __init__(self, latent_dim=512, initial_shape=(512, 4, 2)):#---This is for 64x128 images
    def __init__(self, latent_dim=512, initial_shape=(512, 3, 3)):#---This is for 96x96 images
    #def __init__(self, latent_dim=512, initial_shape=(512, 1, 1)):#---This is for 32x32 images
        super(ResNet18DecoderFC, self).__init__()
        self.fc = nn.Linear(latent_dim, initial_shape[0] * initial_shape[1] * initial_shape[2])
        self.initial_shape = initial_shape
    
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1), #---for 64x128
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1), #---for 96x96
            #nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),#--- for 32x32
            nn.Sigmoid()  # Assuming input images are normalized between 0 and 1
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, *self.initial_shape)    
        x = self.decoder(x)

        return x
#*****************************************************************************
