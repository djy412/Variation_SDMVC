# -*- coding: utf-8 -*-
""" Created on Fri Jan 24 08:50:50 2025
@author: djy41
"""
from scripts.config import NUM_VIEWS, BATCH_SIZE, PATIENCE, PRE_TRAIN_EPOCHS, MODEL_FILENAME, LR, WORKERS, LATENT_DIM, NUM_CLASSES, LAMBDA, GAMMA, FINE_TUNE_EPOCHS, dataset_name , CHANNELS, TOLERANCE, UPDATE_INTERVAL
from classes.ResNet18_autoencoder import ResNet18Autoencoder
from visualizations.Visualization import Show_settings, Show_dataloader_data, Show_Training_Loss, Show_Component_Embeddings, Show_Componet_Reconstructions, Show_Embedding_Space, Show_Complete_Reconstructions, Show_Partial_Embedding_Space, Show_Results, Show_Representation, Show_NMI_By_Epochs, Show_Variance
from scripts.data_loading import load_data
from scripts.utils import cluster_acc, calculate_purity, set_seed 
import torch
import torch.optim as optim
import datetime
from time import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, f1_score, confusion_matrix, precision_score
from PIL import Image
from torchvision import datasets, transforms
#--- For hyperparameter optimization
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.trial import TrialState
from optuna_dashboard import run_server

from torch.optim import Adam
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from skimage.metrics import structural_similarity as ssim
import itertools
import seaborn as sns
import pandas as pd

PreTRAIN = False
MULT_AE = False
SEED = 1
set_seed(SEED)


#************************************************************************
#--- Define Convolutional Auto Encoder
#************************************************************************
class CAE(nn.Module):
    def __init__(self, LATENT_DIM):
        super(CAE, self).__init__()
        # Encoder_Common
        self.encoder_c = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 5, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 32, 64, 64)
            nn.Conv2d(32, 64, 5, padding=1),  # (batch_size, 64, 64, 64)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 64, 32, 32)
            nn.Conv2d(64, 128, 3, padding=1),  # (batch_size, 128, 32, 32)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 128, 16, 16)
            #nn.Conv2d(128, 256, 3, padding=1),  # (batch_size, 256, 16, 16)
            #nn.ReLU(True),
            #nn.MaxPool2d(2, 2),  # (batch_size, 256, 8, 8)
            
            #--- use for 96x96 or larger
            #nn.Conv2d(256, 512, 3, padding=1),  # (batch_size, 512, 16, 16)
            #nn.ReLU(True),
            #nn.MaxPool2d(2, 2),  # (batch_size, 512, 8, 8)
                   
            nn.Flatten(),
            #nn.Linear(512*3*3, LATENT_DIM) #--- 96x96 images
            #nn.Linear(256*6*6, LATENT_DIM) #---  96x96 images
            #nn.Linear(256*4*4, LATENT_DIM) #--- 64x64 images
            nn.Linear(128*4*4, LATENT_DIM) #--- 32x32 images
        )
        # Decoder
        self.decoder = nn.Sequential(
            #nn.Linear(LATENT_DIM, 256*2*2), #--- 32x32 images
            #nn.Linear(LATENT_DIM, 256*4*4), #--- this is for 64x64 images
            #nn.Linear(LATENT_DIM, 256*6*6), #--- this is for 96x96 images
            #nn.Linear(LATENT_DIM, 512*3*3), #--- this is for 96x96 images
            
            nn.Unflatten(1, (128, 4, 4)), #--- 32x32 images
            #nn.Unflatten(1, (512, 4, 2)), #---  
            #nn.Unflatten(1, (256, 4, 4)), #--- 64x64 images
            #nn.Unflatten(1, (256, 6, 6)), #--- 96x96 images
            #nn.Unflatten(1, (512, 3, 3)), #--- 96x96 images
 
            #--- use for 96x96 or larger
            #nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            #nn.ReLU(True),
            
            #nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            #nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, CHANNELS, 5, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
            #nn.Tanh()
        )

    def forward(self, x):
        c = self.encoder_c(x)
        print(c.size())
        #u = self.encoder_p(x) 
        #z = torch.concat([c, u], dim=1)
        x_hat = self.decoder(c)
        return x_hat, c

#************************************************************************
#--- Define IDEC Model
#************************************************************************
class IDEC(nn.Module):
    def __init__(self, n_z, n_clusters):
        
        super(IDEC, self).__init__()
        self.ae = ResNet18Autoencoder(latent_dim=LATENT_DIM)
        #self.ae = CAE(LATENT_DIM).to(device)
        
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path='', view=1):
        pretrain_ae(self.ae, data_loader, view)

    def forward(self, x):
        x_hat, c = self.ae(x)
        q = 1.0 / (1.0 + torch.sum(torch.pow(c.unsqueeze(1) - self.cluster_layer, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()
        return x_hat, c, q

#************************************************************************
#--- Define Sharpening function for target distribution
#************************************************************************
def target_distribution(q):
    weight = q**2 / q.sum(0)
    weight = (weight.t() / weight.sum(1)).t()
    return weight

#************************************************************************
#--- Define Student-T distribution
#************************************************************************
def Student_T_distribution(z, centers):
    q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - centers, 2), 2))
    q = (q.t() / torch.sum(q, 1)).t()
    return q


#--- Define Autoencoder pre-training
#************************************************************************
def pretrain_ae(model, data_loader, view): #- Takes in the model, the data, and which view of the data so use
    print("Pre-training Autoencoder")
    if PreTRAIN == True:
        optimizer = Adam(model.parameters(), lr=LR)
        Loss_histroy = [0]
        
        for epoch in range(PRE_TRAIN_EPOCHS):
            total_loss = 0.0
            if view == 1: #--- Get the first view
                for batch_idx, (x, _, _, y_true, _) in enumerate(data_loader):
                    x = x.to(device)
        
                    optimizer.zero_grad()
                    x_hat, z = model(x)
                    loss = F.mse_loss(x_hat, x)
                    
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    Loss_histroy.append(loss.item())
                print("epoch {} loss={:.4f}".format(epoch, total_loss / (batch_idx + 1)))
                
            if view == 2: #--- Get the second view
                for batch_idx, (_, x, _, y_true, _) in enumerate(data_loader):
                    x = x.to(device)
      
                    optimizer.zero_grad()
                    x_hat, z = model(x)
                    loss = F.mse_loss(x_hat, x)
                  
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    Loss_histroy.append(loss.item())
                print("epoch {} loss={:.4f}".format(epoch, total_loss / (batch_idx + 1)))  

            if view == 3: #--- Get the second view
                for batch_idx, (_, _, x, y_true, _) in enumerate(data_loader):
                    x = x.to(device)
      
                    optimizer.zero_grad()
                    x_hat, z = model(x)
                    loss = F.mse_loss(x_hat, x)
                  
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    Loss_histroy.append(loss.item())
                print("epoch {} loss={:.4f}".format(epoch, total_loss / (batch_idx + 1)))  
               #--- Plot the training loss
        Show_Training_Loss(Loss_histroy)
        
        #--- Save the models weights
        if view == 1:
            if dataset_name == 'MULTI-MNIST':
                torch.save(model.state_dict(), 'weights/Multi-MNIST_ae1.pkl')
            if dataset_name == 'MULTI-FASHION':
                torch.save(model.state_dict(), 'weights/Multi-FASHION_ae1.pkl')
            if dataset_name == 'FASHION-MV':
                torch.save(model.state_dict(), 'weights/FASHION-MV_ae1.pkl')
            if dataset_name == 'MULTI-MVP-N':
                torch.save(model.state_dict(), 'weights/Multi-MVP-N_ae1.pkl') 
            if dataset_name == 'MULTI_STL-10':
                torch.save(model.state_dict(), 'weights/Multi-STL-10_ae1.pkl')             
            if dataset_name == 'MULTI_Eglin':
                torch.save(model.state_dict(), 'weights/MULTI_Eglin_ae1.pkl')  
            if dataset_name == 'MULTI_2V_Market':
                torch.save(model.state_dict(), 'weights/MULTI_2V_Market_ae1.pkl')  
            print("model saved to weights/'Dataset_name'_ae1.pkl")           
        if view == 2:
            if dataset_name == 'MULTI-MNIST':
                torch.save(model.state_dict(), 'weights/Multi-MNIST_ae2.pkl')
            if dataset_name == 'MULTI-FASHION':
                torch.save(model.state_dict(), 'weights/Multi-FASHION_ae2.pkl')
            if dataset_name == 'FASHION-MV':
                torch.save(model.state_dict(), 'weights/FASHION-MV_ae2.pkl')
            if dataset_name == 'MULTI-MVP-N':
                torch.save(model.state_dict(), 'weights/Multi-MVP-N_ae2.pkl') 
            if dataset_name == 'MULTI_STL-10':
                torch.save(model.state_dict(), 'weights/Multi-STL-10_ae2.pkl') 
            if dataset_name == 'MULTI_Eglin':
                torch.save(model.state_dict(), 'weights/MULTI_Eglin_ae2.pkl')
            if dataset_name == 'MULTI_2V_Market':
                torch.save(model.state_dict(), 'weights/MULTI_2V_Market_ae2.pkl') 
            print("model saved to weights/'Dataset_name'_ae2.pkl")
        if view == 3:
            if dataset_name == 'MULTI-MNIST':
                torch.save(model.state_dict(), 'weights/Multi-MNIST_ae3.pkl')
            if dataset_name == 'MULTI-FASHION':
                torch.save(model.state_dict(), 'weights/Multi-FASHION_ae3.pkl')
            if dataset_name == 'FASHION-MV':
                torch.save(model.state_dict(), 'weights/FASHION-MV_ae3.pkl')
            if dataset_name == 'MULTI-MVP-N':
                torch.save(model.state_dict(), 'weights/Multi-MVP-N_ae3.pkl') 
            if dataset_name == 'MULTI_STL-10':
                torch.save(model.state_dict(), 'weights/Multi-STL-10_ae3.pkl') 
            if dataset_name == 'MULTI_Eglin':
                torch.save(model.state_dict(), 'weights/MULTI_Eglin_ae3.pkl')
            if dataset_name == 'MULTI_2V_Market':
                torch.save(model.state_dict(), 'weights/MULTI_2V_Market_ae3.pkl') 
            print("model saved to weights/'Dataset_name'_ae3.pkl")
        
    else: #--- Load models weights
        if view == 1:
            if dataset_name == 'MULTI-MNIST': 
                load_model_path = './weights/Multi-MNIST_ae1.pkl'    
                model.load_state_dict(torch.load(load_model_path))     
            if dataset_name == 'MULTI-FASHION':    
                load_model_path = './weights/Multi-FASHION_ae1.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            if dataset_name == 'FASHION-MV':    
                load_model_path = './weights/FASHION-MV_ae1.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            if dataset_name == 'MULTI-MVP-N':
                load_model_path = './weights/Multi-MVP-N_ae1.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            if dataset_name == 'MULTI_STL-10':   
                load_model_path = './weights/Multi-STL-10_ae1.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            if dataset_name == 'MULTI_Eglin':   
                load_model_path = './weights/MULTI_Eglin_ae1.pkl'
                model.load_state_dict(torch.load(load_model_path))                 
            if dataset_name == 'MULTI_2V_Market':   
                load_model_path = './weights/MULTI_2V_Market_ae1.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            for batch_idx, (x, _, _, labels, _) in enumerate(data_loader):
                with torch.no_grad():
                    x = x.to(device)
                    x_hat, z = model(x)
                break
        if view == 2:
            if dataset_name == 'MULTI-MNIST': 
                load_model_path = './weights/Multi-MNIST_ae2.pkl'    
                model.load_state_dict(torch.load(load_model_path))     
            if dataset_name == 'MULTI-FASHION':    
                load_model_path = './weights/Multi-FASHION_ae2.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            if dataset_name == 'FASHION-MV':    
                load_model_path = './weights/FASHION-MV_ae2.pkl'
                model.load_state_dict(torch.load(load_model_path))             
            if dataset_name == 'MULTI-MVP-N':
                load_model_path = './weights/Multi-MVP-N_ae2.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            if dataset_name == 'MULTI_STL-10':   
                load_model_path = './weights/Multi-STL-10_ae2.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            if dataset_name == 'MULTI_Eglin':   
                load_model_path = './weights/MULTI_Eglin_ae2.pkl'
                model.load_state_dict(torch.load(load_model_path))
            if dataset_name == 'MULTI_2V_Market':   
                load_model_path = './weights/MULTI_2V_Market_ae2.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            for batch_idx, (_, x, _, labels, _) in enumerate(data_loader):
                with torch.no_grad():
                    x = x.to(device)
                    x_hat, z = model(x)
                break    
        if view == 3:
            if dataset_name == 'MULTI-MNIST': 
                load_model_path = './weights/Multi-MNIST_ae3.pkl'    
                model.load_state_dict(torch.load(load_model_path))     
            if dataset_name == 'MULTI-FASHION':    
                load_model_path = './weights/Multi-FASHION_ae3.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            if dataset_name == 'FASHION-MV':    
                load_model_path = './weights/FASHION-MV_ae3.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            if dataset_name == 'MULTI-MVP-N':
                load_model_path = './weights/Multi-MVP-N_ae3.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            if dataset_name == 'MULTI_STL-10':   
                load_model_path = './weights/Multi-STL-10_ae3.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            if dataset_name == 'MULTI_Eglin':   
                load_model_path = './weights/MULTI_Eglin_ae3.pkl'
                model.load_state_dict(torch.load(load_model_path))  
            if dataset_name == 'MULTI_2V_Market':   
                load_model_path = './weights/MULTI_2V_Market_ae3.pkl'
                model.load_state_dict(torch.load(load_model_path)) 
            for batch_idx, (_, _, x, labels, _) in enumerate(data_loader):
                with torch.no_grad():
                    x = x.to(device)
                    x_hat, z = model(x)
                break    
    
    Show_Complete_Reconstructions(x, x_hat)    
    
    #--- Clear everything in memory now that the models are saved and displayed
    x = None
    x_hat = None
    z = None
    torch.cuda.empty_cache()
#----------------------------------------------------------------------------


#*****************************************************************************
#--- Main Function
#*****************************************************************************
if __name__=='__main__': 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    print('Loading data...')
    if dataset_name == 'MULTI-MNIST':
        dataset, dims, view, data_size, class_num = load_data("MULTI-MNIST")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI-FASHION':
        dataset, dims, view, data_size, class_num = load_data("MULTI-FASHION")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'FASHION-MV':
        dataset, dims, view, data_size, class_num = load_data("FASHION-MV")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)    
    elif dataset_name == 'MULTI-MVP-N':
        dataset, dims, view, data_size, class_num = load_data("MULTI-MVP-N")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI_STL-10':
        dataset, dims, view, data_size, class_num = load_data("MULTI-STL-10")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI_Eglin':
        dataset, dims, view, data_size, class_num = load_data("MULTI_Eglin")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI_2V_Market':
        dataset, dims, view, data_size, class_num = load_data("MULTI_2V_Market")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)

    #--- Show all the Settings
    Show_settings() 
    for img1, img2, _, labels, _ in data_loader:
        data = img1
        break
    #--- Show the dataloader images
    Show_dataloader_data(img1, img2, labels) 
    img1 = img2 = lables = data = None #--- Clear data
    torch.cuda.empty_cache()
        
    #--- Define the models
    model1 = IDEC(LATENT_DIM, NUM_CLASSES).to(device)
    if MULT_AE:
        model2 = IDEC(LATENT_DIM, NUM_CLASSES).to(device)
        if view == 3:
            model3 = IDEC(LATENT_DIM, NUM_CLASSES).to(device)

    #--- STEP 1 --- 
    #--- Pre-train deep autoencoder
    model1.pretrain(data_loader, 1)
    if MULT_AE:
        model2.pretrain(data_loader, 2)
        if view == 3:    
            model3.pretrain(data_loader, 3)
            
    # Define an optimizer for models
    optimizer = optim.Adam(model1.parameters(), lr=LR)
    if view == 2 and MULT_AE:
        optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=LR)
    if view == 3 and MULT_AE:
        print("Running 3 views with one AE per view")
        optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()) + list(model3.parameters()), lr=LR)    
            
    #--- STEP 2 ---
    #--- Initialize cluster centers, centers for each view
    data=[]
    y_true = []
    for x, _, _, y, _ in data_loader:
        data.append(x)
        y_true.append(y)
    data = np.concatenate(data)
    y_true = np.concatenate(y_true)
    data = torch.Tensor(data).to(device)
    with torch.no_grad():
        x_hat, z = model1.ae(data)
    kmeans = KMeans(n_clusters=NUM_CLASSES, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())   
    #--- Load the initial cluster centers into model
    #--- 
    model1.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device) 
    if MULT_AE:
        model2.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device) 
    if view == 3 and MULT_AE:
        model3.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device) 
        
    # data=[]
    # y_true = []
    # for _, x, _, y, _ in data_loader:
    #     data.append(x)
    #     y_true.append(y)
    # data = np.concatenate(data)
    # y_true = np.concatenate(y_true)
    # data = torch.Tensor(data).to(device)
    # with torch.no_grad():
    #     x_hat, z = model2.ae(data)
    # kmeans = KMeans(n_clusters=NUM_CLASSES, n_init=20)
    # y_pred = kmeans.fit_predict(z.data.cpu().numpy())   
    # #--- Load the initial cluster centers into model
    # model2.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)   
    # if MULT_AE:
    #     if view == 3:
    #         data=[]
    #         y_true = []
    #         for _, _, x, y, _ in data_loader:
    #             data.append(x)
    #             y_true.append(y)
    #         data = np.concatenate(data)
    #         y_true = np.concatenate(y_true)
    #         data = torch.Tensor(data).to(device)
    #         with torch.no_grad():
    #             x_hat, z = model3.ae(data)
    #         kmeans = KMeans(n_clusters=NUM_CLASSES, n_init=20)
    #         y_pred = kmeans.fit_predict(z.data.cpu().numpy())   
    #         #--- Load the initial cluster centers into model
    #         model3.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device) 
    
    #--- Show the latent space
    Show_Partial_Embedding_Space(z, y_true)
    # Clear all unneeded data
    x_hat = z = data = None
    torch.cuda.empty_cache()
    
    #--- Test starting accuracy, NMI, and purity
    p = [torch.zeros(data_size, NUM_CLASSES, device=device), torch.zeros(data_size, device=device, dtype=torch.long)]
    y_pred_total, y_true = [],[]  # To collect y_pred y_true across all batches
    for x, _, _, y, idx in data_loader:  # Using shuffled data
        x = x.to(device)
        with torch.no_grad():
            _, z, tmp_q = model1(x)     
        # update target distribution p
        tmp_q = tmp_q.data
        p[0][idx] = target_distribution(tmp_q)
        p[1][idx] = idx.to(device).long() 
            
        # evaluate clustering performance
        y_pred = tmp_q.cpu().numpy().argmax(1)
        y_pred_total.extend(y_pred)  # Collect y_pred for accuracy, NMI, ARI calculations
                    
        # Collect y_true for accuracy, NMI, ARI calculations
        y_true.extend(y.cpu().numpy())
    
    x = y = tmp_q = p = None #--- clear variables
                
    acc = cluster_acc(y_true, y_pred_total, NUM_CLASSES)
    nmi = nmi_score(y_true, y_pred_total)
    ari = ari_score(y_true, y_pred_total)
    pur = calculate_purity(y_true, y_pred_total)   
    print('Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', purity {:.4f}'.format(pur))
    
    Start_ACC = acc
    Start_NMI = nmi
    Start_PUR = pur
    
    nmi_view1, nmi_view2, nmi_view3 = [],[],[]
    Rate = []
    
    #--- STEP 3 ---
    #--- Fine-tuning phase
    swap = 1
    t0 = time()
    switch = 1
    NMI_LAST = 0
    previous_value = float('inf')
    stable_count = 0  # Counter for stability
    #--------------------------------------------------------------------------
    for epoch in range(FINE_TUNE_EPOCHS):
        
        #--- Update the target distribution based on the view with highest probability
        if epoch % UPDATE_INTERVAL == 0:  
            p = [torch.zeros(data_size, NUM_CLASSES, device=device), torch.zeros(data_size, device=device, dtype=torch.long)]
            y_pred_total1, y_pred_total2, y_true = [],[],[]  # To collect y_pred y_true across all batches
            if view == 3:
                y_pred_total3 = []
                
            for A, P, N, y, idx in data_loader:  # Using shuffled data
                A = A.to(device)
                P = P.to(device)
                N = N.to(device)
                                
                _, z1, q1 = model1(A)
                _, z2, q2 = model1(P)
                if view == 3:
                    _, z3, q3 = model1(N)
                
                if MULT_AE:
                    _, z2, q2 = model2(P)    
                if view == 3 and MULT_AE:
                    _, z3, q3 = model3(N)
                
                #--- create q based off the variance of q1 and q2 and q3
                q1 = q1.detach() #--- gradients not needed
                q2 = q2.detach()
                if view == 3:
                    q3 = q3.detach()
                    
                positions = torch.arange(q1.size(1)).float().to(device)  # Shape: (N,)      
                # Mean for each batch
                mean_q1 = torch.sum(q1 * positions, dim=1, keepdim=True).to(device)  # Shape: (B, 1)
                mean_q2 = torch.sum(q2 * positions, dim=1, keepdim=True).to(device)  # Shape: (B, 1) 
                if view == 3:
                    mean_q3 = torch.sum(q3 * positions, dim=1, keepdim=True).to(device)  # Shape: (B, 1) 
                # Variance for each batch
                var_q1 = torch.sum(((positions - mean_q1)**2) * q1, dim=1, keepdim=True).to(device)  
                var_q2 = torch.sum(((positions - mean_q2)**2) * q2, dim=1, keepdim=True).to(device)   
                if view == 3:
                    var_q3 = torch.sum(((positions - mean_q3)**2) * q3, dim=1, keepdim=True).to(device)  
                # Weights inversely proportional to variances
                if view == 2:
                    w1 = (1/var_q1) / ((1/var_q1) + (1/var_q2)).to(device) 
                    w2 = (1/var_q2) / ((1/var_q1) + (1/var_q2)).to(device)  
                if view == 3:
                    w1 = (1/var_q1) / ((1/var_q1) + (1/var_q2) + (1/var_q3)).to(device)  
                    w2 = (1/var_q2) / ((1/var_q1) + (1/var_q2) + (1/var_q3)).to(device)   
                    w3 = (1/var_q3) / ((1/var_q1) + (1/var_q2) + (1/var_q3)).to(device)  
                # Combine distributions with broadcasting
                q = w1*q1 + w2*q2  # Shape: (B, N)
                if view == 3:
                    q = w1*q1 + w2*q2 + w3*q3
                # Normalize combined distributions
                q = q / q.sum(dim=1, keepdim=True)    
                    
                # update target distribution p
                p[0][idx] = target_distribution(q)
                p[1][idx] = idx.to(device).long() 
                
                y_pred = q1.cpu().numpy().argmax(1)
                y_pred_total1.extend(y_pred)  #--- Collect y_pred for accuracy, NMI, ARI calculations
                y_pred = q2.cpu().numpy().argmax(1)
                y_pred_total2.extend(y_pred)   
                if view == 3:
                    y_pred = q3.cpu().numpy().argmax(1)
                    y_pred_total3.extend(y_pred)                      
                y_true.extend(y.cpu().numpy()) #--- Collect y_true 
                       
            #--- Get updated metrics        
            acc = cluster_acc(y_true, y_pred_total1, NUM_CLASSES)
            nmi = nmi_score(y_true, y_pred_total1)
            ari = ari_score(y_true, y_pred_total1)
            print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
            nmi_view1.append(nmi) #--- Save the NMI for plotting
            acc = cluster_acc(y_true, y_pred_total2, NUM_CLASSES)
            nmi = nmi_score(y_true, y_pred_total2)
            ari = ari_score(y_true, y_pred_total2)
            print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
            nmi_view2.append(nmi) #--- Save the NMI for plotting
            #Show_Partial_Embedding_Space(z1, y)
            #Show_Partial_Embedding_Space(z2, y)
            if view == 3:
                acc = cluster_acc(y_true, y_pred_total3, NUM_CLASSES)
                nmi = nmi_score(y_true, y_pred_total3)
                ari = ari_score(y_true, y_pred_total3)
                print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
                nmi_view3.append(nmi) #--- Save the NMI for plotting
                #Show_Partial_Embedding_Space(z3, y)
                
            A = P = N = y = idx  = None  # Clear variables for the next iteration
            
            #--- Check for stoping condition
            change = abs(NMI_LAST - nmi)
            print(f"Iteration {epoch}: Change = {change:.5e}")
            if change < TOLERANCE:
                stable_count += 1  # Increment stability counter
                if stable_count >= PATIENCE:  # If stable for `patience` iterations, stop
                    print("Stopping criterion met: Performance measure stabilized.")
                    break
            else:
                stable_count = 0  # Reset counter if change is significant
            NMI_LAST = nmi  # Update previous value            
            
        #--- Train on batches    
        for batch_idx, (x, x2, x3, _, idx) in enumerate(data_loader):
            x = x.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            idx = idx.to(device)
              
            x_hat, _, q1 = model1(x)                   
            reconstr_loss = F.mse_loss(x_hat, x)
            kl_loss = F.kl_div(q1.log(), p[0][idx], reduction='batchmean')  
            loss1 = GAMMA*kl_loss + reconstr_loss
            
            x2_hat, _, q2 = model1(x2)
            if MULT_AE:
                _, z2, q2 = model2(P)
            reconstr_loss2 = F.mse_loss(x2_hat, x2)
            kl_loss2 = F.kl_div(q2.log(), p[0][idx], reduction='batchmean')   
            loss2 = GAMMA*kl_loss2 + reconstr_loss2
            
            if view == 3:
                x3_hat, _, q3 = model1(x3)            
                reconstr_loss3 = F.mse_loss(x3_hat, x3)
                kl_loss3 = F.kl_div(q3.log(), p[0][idx], reduction='batchmean')   
                loss3 = GAMMA*kl_loss3 + reconstr_loss3
            if view == 3 and MULT_AE:
                x3_hat, _, q3 = model3(x3)            
                reconstr_loss3 = F.mse_loss(x3_hat, x3)
                kl_loss3 = F.kl_div(q3.log(), p[0][idx], reduction='batchmean')   
                loss3 = GAMMA*kl_loss3 + reconstr_loss3
                   
            loss = loss1+loss2
            if view == 3:    
                loss += loss3
                
            optimizer.zero_grad()
            loss.backward()       
            optimizer.step()
    #--------------------------------------------------------------------------

    print('Fine Tuning time: ', time() - t0)
    #--- Save the models
    torch.save(model1.state_dict(), './weights/IDEC_Model1.pt')
    print("model saved as {}.".format('IDEC_Model1.pt'))
    # torch.save(model2.state_dict(), './weights/IDEC_Model2.pt')
    # print("model saved as {}.".format('IDEC_Model2.pt'))
 
    #--- Plot the NMI of each view over time
    axis = range(len(nmi_view1))
    plt.figure("NMI")
    plt.plot(axis, nmi_view1, label = 'view1')
    plt.plot(axis, nmi_view2, label = 'view2')
    if view == 3:
        plt.plot(axis, nmi_view3, label = 'view3')
    
    # Labels and legend
    plt.xlabel("Update")
    plt.ylabel("NMI")
    plt.title("Plot of NMI per Update")
    plt.legend()
    plt.grid(True)
    plt.show   

    Full_data_a, y_true=[], []
    torch.cuda.empty_cache()
    for a, _, _, y, _ in data_loader:
        Full_data_a.append(a)
        y_true.append(y)

    Full_data_a = torch.cat(Full_data_a).to(device)
    y_true = torch.cat(y_true)
    
    with torch.no_grad():
        _, z, _ = model1(Full_data_a)
    #--- Show the latent space
    Show_Partial_Embedding_Space(z, y_true)
    #Show_Embedding_Space(z, u, y_true)
    
    Full_data_a = y = None # Clear variables 
    
    y_pred_total, y_true = [],[]  # To collect y_pred y_true across all batches
    for _, x, _, y, idx in data_loader:  # Using shuffled data
        x = x.to(device)
                    
        x_hat, z, tmp_q = model1(x)
        # update target distribution p
        tmp_q = tmp_q.data
        p[0][idx] = target_distribution(tmp_q)
        p[1][idx] = idx.to(device).long() 
            
        # evaluate clustering performance
        y_pred = tmp_q.cpu().numpy().argmax(1)
        y_pred_total.extend(y_pred)  # Collect y_pred for accuracy, NMI, ARI calculations
                    
        # Collect y_true for accuracy, NMI, ARI calculations
        y_true.extend(y.cpu().numpy())

    Show_Complete_Reconstructions(x, x_hat)    
                    
    acc = cluster_acc(y_true, y_pred_total, NUM_CLASSES)
    nmi = nmi_score(y_true, y_pred_total)
    ari = ari_score(y_true, y_pred_total)
    pur = calculate_purity(y_true, y_pred_total)   
    print('View 1: Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', purity {:.4f}'.format(pur))
    
    END_ACC = acc
    END_NMI = nmi
    END_PUR = pur
    Show_Results(SEED, Start_ACC, Start_NMI, Start_PUR, END_ACC,  END_NMI, END_PUR)
    
    y_pred_total, y_true = [],[]  # To collect y_pred y_true across all batches
    for _, x, _, y, idx in data_loader:  # Using shuffled data
        x = x.to(device)
        x_hat, z, tmp_q = model1(x)
        if MULT_AE:            
            x_hat, z, tmp_q = model2(x)
        # update target distribution p
        tmp_q = tmp_q.data
        p[0][idx] = target_distribution(tmp_q)
        p[1][idx] = idx.to(device).long() 
            
        # evaluate clustering performance
        y_pred = tmp_q.cpu().numpy().argmax(1)
        y_pred_total.extend(y_pred)  # Collect y_pred for accuracy, NMI, ARI calculations
                    
        # Collect y_true for accuracy, NMI, ARI calculations
        y_true.extend(y.cpu().numpy())
                 
    acc = cluster_acc(y_true, y_pred_total, NUM_CLASSES)
    nmi = nmi_score(y_true, y_pred_total)
    ari = ari_score(y_true, y_pred_total)
    pur = calculate_purity(y_true, y_pred_total)   
    print('View 2: Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', purity {:.4f}'.format(pur))
    
    Full_data_p, y_true=[], []
    torch.cuda.empty_cache()
    for _, pos, _, y, _ in data_loader:
        Full_data_p.append(pos)
        y_true.append(y)

    Full_data_p = torch.cat(Full_data_p).to(device)
    y_true = torch.cat(y_true)
    
    with torch.no_grad():
        _, z, _ = model1(Full_data_p)
        if MULT_AE:
            _, z, _ = model2(Full_data_p)
    #--- Show the latent space
    Show_Partial_Embedding_Space(z, y_true)
    #Show_Embedding_Space(z, u, y_true)
    Full_data_p = y = None # Clear variables 
    
    
    if view == 3:
        y_pred_total, y_true = [],[]  # To collect y_pred y_true across all batches
        for _, _, x, y, idx in data_loader:  # Using shuffled data
            x = x.to(device)
                        
            x_hat, z, tmp_q = model1(x)
            if MULT_AE:
                x_hat, z, tmp_q = model3(x)
            # update target distribution p
            tmp_q = tmp_q.data
            p[0][idx] = target_distribution(tmp_q)
            p[1][idx] = idx.to(device).long() 
                
            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            y_pred_total.extend(y_pred)  # Collect y_pred for accuracy, NMI, ARI calculations
                        
            # Collect y_true for accuracy, NMI, ARI calculations
            y_true.extend(y.cpu().numpy())
                             
        acc = cluster_acc(y_true, y_pred_total, NUM_CLASSES)
        nmi = nmi_score(y_true, y_pred_total)
        ari = ari_score(y_true, y_pred_total)
        pur = calculate_purity(y_true, y_pred_total)   
        print('View 3: Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', purity {:.4f}'.format(pur))
        
        Full_data_n, y_true=[], []
        torch.cuda.empty_cache()
        for _, _, n, y, _ in data_loader:
            Full_data_n.append(n)
            y_true.append(y)
    
        Full_data_n = torch.cat(Full_data_n).to(device)
        y_true = torch.cat(y_true)
        
        with torch.no_grad():
            _, z, _ = model1(Full_data_n)
            if MULT_AE:
                _, z, _ = model3(Full_data_n)
        #--- Show the latent space
        Show_Partial_Embedding_Space(z, y_true)
        #Show_Embedding_Space(z, u, y_true)
        Full_data_n = y = None # Clear variables 