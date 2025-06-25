# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:19:20 2025

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

np.random.seed(42)
torch.manual_seed(42)

#load fashion MNIST DATASET
MNIST_train = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
MNIST_test = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_loader_MNIST = torch.utils.data.DataLoader(dataset=MNIST_train, batch_size=100, shuffle=True)
test_loader_MNIST = torch.utils.data.DataLoader(dataset=MNIST_test, batch_size=100, shuffle=False)

X_train_MNIST, Y_train_MNIST = next(iter(train_loader_MNIST))
X_test_MNIST, Y_test_MNIST = next(iter(test_loader_MNIST))



#Initialize RNN

RNN_params = {
    "N_in": 784,               # e.g., flattened 28x28 FashionMNIST image
    "N_out": 10,               # number of classes in FashionMNIST
    "N_neurons": 30,          # number of hidden units per RNN layer
    "N_layers": 3,             # depth of the RNN
    "time_steady_state": 15    # number of repeated timesteps to reach steady state
}

model = RNN_network(params=RNN_params)

model.init_esn_weights(reservoir = True)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#array to store the accuracy of the model

test_accuracy_list = []
train_loss = []
test_loss = []

n_epochs = 5 


D = train_BP_torch(model, n_epochs, train_loader_MNIST, test_loader_MNIST, loss, optimizer)

# start_time = time.time()
# for epoch in range(n_epochs):
    
#     for i,(images, labels) in enumerate(train_loader_MNIST):
        
#         model.train()
#         #move data to gpu for faster processing
#         images = images.to(device)
#         labels = labels.to(device)
        
#         #forward pass
#         Y_pred = model.forward(images)
#         loss_value = loss(Y_pred,labels)
#         train_loss.append(loss(Y_pred,labels).item())
        
#         #backward pass
#         optimizer.zero_grad()
#         loss_value.backward()
#         optimizer.step()
            
#         # evaluate after some epochs
#         if (i+1) % 50 == 0:
#             model.eval()
#             correct = 0 
#             total = 0
            
#             for images, labels in test_loader_MNIST:
#                 images = images.to(device)
#                 labels = labels.to(device)
#                 Y_pred = model.forward(images)
#                 _, predicted = torch.max(Y_pred.data, 1)
                
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 test_loss.append(loss(Y_pred,labels).item())
            
#             accuracy = ( 100*correct/total)
#             test_accuracy_list.append(accuracy)
            
#             print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader_MNIST)}], Loss: {loss_value.item()}, Test Accuracy: {accuracy}%')


# end_time = time.time()
# print(f'Total time: {(end_time - start_time)/3600}h')












