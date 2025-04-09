# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 22:22:45 2024

@author: djy41
"""
import torch.nn as nn
import torch
import torchvision.models as models

class ResNet18EncoderFC(nn.Module):
    def __init__(self, latent_dim=512):
        super(ResNet18EncoderFC, self).__init__()
        resnet18 = models.resnet18(weights = None)
        # Modify the first convolutional layer to accept 1-channel images
        # resnet18.conv1 = nn.Conv2d(
        #     in_channels=1,  # Change from 3 to 1
        #     out_channels=64,  # Keep the same as original
        #     kernel_size=7,
        #     stride=2,
        #     padding=3,
        #     bias=False
        # )
        
        self.features = nn.Sequential(*list(resnet18.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):     
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        z = self.fc(x)

        return z
#*****************************************************************************




