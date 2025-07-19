# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 23:30:39 2025

@author: anas.skalli
"""

# Debug SPSA on linear classifier model


# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 12:02:45 2025

@author: anas.skalli
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import asarray
from numpy import savetxt
from NN_utils import *
import torch
import torch.nn as nn
from torchvision import datasets, transforms
#from torchsummary import summary
import time
from types import SimpleNamespace
import pickle
import gc
from optimization_algorithms import *

#use SPSA to optimize The Neural network

#load fashion MNIST DATASET

transform_data = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize((0.2868,), (0.3524,))
])

MNIST_train = datasets.FashionMNIST(root='./data', train=True, transform=transform_data, download=True)
MNIST_test = datasets.FashionMNIST(root='./data', train=False, transform=transform_data, download=True)

train_loader_MNIST = torch.utils.data.DataLoader(dataset=MNIST_train, batch_size=1000, shuffle=True)
test_loader_MNIST = torch.utils.data.DataLoader(dataset=MNIST_test, batch_size=10000, shuffle=False)

X_train_MNIST, Y_train_MNIST = next(iter(train_loader_MNIST))
X_test_MNIST, Y_test_MNIST = next(iter(test_loader_MNIST))


N_neurons_vec = [5, 10, 20, 30, 50, 75, 100]
N_neurons_vec = [50]

n_epochs = 100
results = []

stats = 1

print("Script started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

start_time = time.time()
for s in range(stats):

     #Initialize RNN
 
        RNN_params = {
             "N_in": 784,               # e.g., flattened 28x28 FashionMNIST image
             "N_out": 10,               # number of classes in FashionMNIST
         }
         
        model = Linear_model(params=RNN_params)
         
           
        loss = nn.CrossEntropyLoss()
        
        
        N_dim = model.count_parameters()

        loss = nn.CrossEntropyLoss()
        # learning parameters
        
        init_pos = model.get_params()
        
        # if init_pos.requires_grad:
        #     # Detach the tensor from the computation graph
        #     init_pos = init_pos.detach()
        # if init_pos.is_cuda:
        #     # Move the tensor to the CPU
        #     init_pos = init_pos.cpu()
        # init_pos = init_pos.numpy()
        
        SPSA_optimizer = SPSA_opt(init_pos,alpha=1e-3,epsilon=1e-5)
        Adam = AdamOptimizer(init_pos, lr=1e-3, beta1=0.9, beta2=0.9, epsilon=1e-8)
        
        D = train_online_SPSA_NN(model, n_epochs, train_loader_MNIST, test_loader_MNIST, loss, SPSA_optimizer,adam_optimizer=Adam)
       
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # D = train_BP_torch(model, n_epochs, train_loader_MNIST, test_loader_MNIST, loss, optimizer)
        
        results.append(D)
        
        del model
        del SPSA_optimizer
        del Adam
        torch.cuda.empty_cache()
        gc.collect()
        
        
end_time = time.time()
print(f'Total time = {end_time- start_time} s')

# with open('results_oscillator_dynass_SPSA_0p1dtint_0.01W_inputscaling_2.pkl', 'wb') as f:
#     pickle.dump(results, f)     
        

        
        
        
        