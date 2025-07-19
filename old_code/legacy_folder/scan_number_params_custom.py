# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:45:25 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from CMA_obj import CMA_opt
from PEPG_obj import PEPG_opt
from SPSA_obj import SPSA_opt

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



np.random.seed(42)
torch.manual_seed(42)

#load fashion MNIST DATASET
MNIST_train = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
MNIST_test = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_loader_MNIST = torch.utils.data.DataLoader(dataset=MNIST_train, batch_size=100, shuffle=True)
test_loader_MNIST = torch.utils.data.DataLoader(dataset=MNIST_test, batch_size=10000, shuffle=False)

X_train_MNIST, Y_train_MNIST = next(iter(train_loader_MNIST))
X_test_MNIST, Y_test_MNIST = next(iter(test_loader_MNIST))

N_neurons_vec = [5, 10, 20, 30, 50, 75, 100]
#N_neurons_vec = [50]

n_epochs = 20
results = []

stats = 3

for s in range(stats):
    for n_neurons in N_neurons_vec:
    
        print(f'Using {n_neurons} per layer')
        #Initialize RNN
        
        RNN_params = {
            "N_in": 784,               # e.g., flattened 28x28 FashionMNIST image
            "N_out": 10,               # number of classes in FashionMNIST
            "N_neurons": n_neurons,          # number of hidden units per RNN layer
            "N_layers": 3,             # depth of the RNN
            "time_steady_state": 15    # number of repeated timesteps to reach steady state
        }
        
        model = Custom_RNN(params=RNN_params)
        
        model.init_esn_weights(reservoir = False)
        
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        
        #array to store the accuracy of the model
        
        test_accuracy_list = []
        train_loss = []
        test_loss = []
        
        print(model.count_parameters())
        D = train_BP_torch(model, n_epochs, train_loader_MNIST, test_loader_MNIST, loss, optimizer)
    
        results.append(D)


with open('results_custom_sin.pkl', 'wb') as f:
    pickle.dump(results, f)

test_loss_vec = []
test_acc_vec = []

for k in range(len(results)):
    dummy = results[k]
    test_loss_vec.append(np.mean(dummy['test_loss'][-10:-1]))
    
plt.plot(N_neurons_vec,test_loss_vec,'--o')
    
    
for k in range(len(results)):
    dummy = results[k]
    test_acc_vec.append(np.mean(dummy['test_acc'][-5:-1]))

plt.plot(N_neurons_vec,test_acc_vec,'--o')

