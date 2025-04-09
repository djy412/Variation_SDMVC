# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:44:42 2024

@author: djy41
"""
import torch.nn as nn
from scripts.config import LATENT_DIM, NUM_CLASSES


#************************************************************************
#--- Create the Fully Connected network for classification training
#************************************************************************
class FC_NN(nn.Module):
    def __init__(self):
        super(FC_NN, self).__init__()
        self.linear1 = nn.Linear(LATENT_DIM, NUM_CLASSES) 
        
    def forward(self, x):
        x = self.linear1(x)
        return x
#*****************************************************************************