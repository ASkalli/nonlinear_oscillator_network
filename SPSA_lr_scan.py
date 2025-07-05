# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 21:18:11 2025

@author: anas.skalli
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 12:02:45 2025

@author: anas.skalli
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

train_loader_MNIST = torch.utils.data.DataLoader(dataset=MNIST_train, batch_size=100, shuffle=True)
test_loader_MNIST = torch.utils.data.DataLoader(dataset=MNIST_test, batch_size=10000, shuffle=False)

X_train_MNIST, Y_train_MNIST = next(iter(train_loader_MNIST))
X_test_MNIST, Y_test_MNIST = next(iter(test_loader_MNIST))


lr_vec = [1e-4 ,5e-4, 1e-3,5e-3,1e-2]
n_neurons = 50

n_epochs = 25
results = []

stats = 1

start_time = time.time()
for s in range(stats):
    for lr in lr_vec:
     #Initialize RNN
 
        RNN_params = {
             "N_in": 784,               # e.g., flattened 28x28 FashionMNIST image
             "N_out": 10,               # number of classes in FashionMNIST
             "N_neurons": n_neurons,          # number of hidden units per RNN layer
             "N_layers": 3,             # depth of the RNN
             "time_steady_state": 500    # number of repeated timesteps to reach steady state
         }
         
        model = Oscillator_RNN_dyn(params=RNN_params)
         
        model.init_esn_weights(reservoir = False)
        model.dt = 0.1
        model.eps_int = 0.1
        model.save_activations = False
         
        loss = nn.CrossEntropyLoss()
        print(f'Using {n_neurons} per layer, run {s+1}, number of parameters {model.count_parameters()}, lr = {lr}')
        
        N_dim = model.count_parameters()
        #specify we don't need the computation graph to keep track of the gradients, we will use SPSA to update the weights
        with torch.no_grad():
            for param in model.parameters():
                param.requires_grad = False
        loss = nn.CrossEntropyLoss()
        # learning parameters
        
        init_pos = model.get_params()
        
        if init_pos.requires_grad:
            # Detach the tensor from the computation graph
            init_pos = init_pos.detach()
        if init_pos.is_cuda:
            # Move the tensor to the CPU
            init_pos = init_pos.cpu()
        init_pos = init_pos.numpy()
        
        SPSA_optimizer = SPSA_opt(init_pos,alpha=1e-3,epsilon=1e-5)
        Adam = AdamOptimizer(init_pos, lr=lr, beta1=0.9, beta2=0.9, epsilon=1e-8)
        
        
        D = train_online_SPSA_NN(model, n_epochs, train_loader_MNIST, test_loader_MNIST, loss, SPSA_optimizer,Adam)
        results.append(D)
        
        
end_time = time.time()
print(f'Total time = {end_time- start_time} s')

with open('results_oscillator_dynass_SPSA_lr_scan.pkl', 'wb') as f:
    pickle.dump(results, f)     
        
test_loss_vec = []
test_loss_std = []
test_acc_vec = []
test_acc_std = []
train_loss_vec = []
train_loss_std = []
n_params = []
idx_plot = -5  # average over the last 5 epochs

dummy = np.reshape(results, [stats, len(lr_vec)])

for i in range(len(N_neurons_vec)):
    train_loss_runs = []
    test_loss_runs = []
    test_acc_runs = []
    
    for j in range(stats):
        train_loss = dummy[j, i]['train_loss']
        test_loss = dummy[j, i]['test_loss']
        test_acc = dummy[j, i]['test_acc']
        
        train_loss_runs.append(np.mean(train_loss[idx_plot:-1]))
        test_loss_runs.append(np.mean(test_loss[idx_plot:-1]))
        test_acc_runs.append(np.mean(test_acc[idx_plot:-1]))

    train_loss_vec.append(np.mean(train_loss_runs))
    test_loss_vec.append(np.mean(test_loss_runs))
    test_acc_vec.append(np.mean(test_acc_runs))

    train_loss_std.append(np.std(train_loss_runs))
    test_loss_std.append(np.std(test_loss_runs))
    test_acc_std.append(np.std(test_acc_runs))

    n_params.append(dummy[0, i]['n_params'])

n_params = np.array(n_params)

# Accuracy plot
plt.figure()
plt.loglog(lr_vec, test_acc_vec, '-o', label='Test Accuracy')
plt.fill_between(lr_vec,
                 np.array(test_acc_vec) - np.array(test_acc_std),
                 np.array(test_acc_vec) + np.array(test_acc_std),
                 alpha=0.3)
plt.xlabel('Number of parameters')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.legend()
plt.show()

# Loss plot
plt.figure()
plt.loglog(lr_vec, test_loss_vec, '-o', label='Test Loss')
plt.fill_between(lr_vec,
                 np.array(test_loss_vec) - np.array(test_loss_std),
                 np.array(test_loss_vec) + np.array(test_loss_std),
                 alpha=0.3)

plt.loglog(lr_vec, train_loss_vec, '-o', label='Train Loss')
plt.fill_between(lr_vec,
                 np.array(train_loss_vec) - np.array(train_loss_std),
                 np.array(train_loss_vec) + np.array(train_loss_std),
                 alpha=0.3)

plt.xlabel('Number of parameters')
plt.ylabel('CCE Loss')
plt.grid(True)
plt.legend()
plt.show()
    


# with open('results_oscillator_dynass_dummy_75_100.pkl','rb') as f: 
#     results = pickle.load(f)

        
        
        
        
