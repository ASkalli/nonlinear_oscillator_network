# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 23:41:29 2025

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
from torch.utils.data import DataLoader
import time

torch.set_default_dtype(torch.float32)  # Global default for speed zoooom !
np.random.seed(42)
torch.manual_seed(42)

# --- Parameters ---
RNN_params = {
    "N_in": 784,
    "N_out": 10,
    "N_neurons": 30,
    "N_layers": 3,
    "time_steady_state": 1000
}


model = Oscillator_RNN_dyn(params=RNN_params)
model.init_esn_weights(reservoir = False)
# Set model to eval mode and CPU (or use GPU if preferred)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.alpha = 0.9


# --- Load one FashionMNIST sample ---
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
# Create DataLoader for a batch of 100 samples
loader = DataLoader(dataset, batch_size=100, shuffle=False)

# Get one batch
img, label = next(iter(loader))
# Flatten and prepare repeated input
x = img.to(device)  # (1, 784)
T = RNN_params["time_steady_state"]

# --- Forward Pass ---
start_time = time.time()
with torch.no_grad():
    y_pred = model(x,5e-2,dt=0.1)
end_time = time.time()

print(f' Forward pass time {end_time - start_time}')
layer = 'layer1'
# --- Plot Hidden State Over Time (first neuron for example) ---
rnn_out = torch.stack(model.activations[layer]).squeeze().numpy()  # shape: (T, hidden_size)
plt.figure(figsize=(10, 5))
for i in range(RNN_params["N_neurons"]):  # plot neurons
    plt.plot(rnn_out[:,50,:], label=f"Neuron {i}")
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.title("RNN hidden state evolution (neurons)")
#plt.legend()
plt.grid(True)
plt.show()

dummy = rnn_out[:,50,:]
plt.figure(figsize=(10, 5))
for i in range(RNN_params["N_neurons"]):  # plot first 5 neurons
    plt.semilogy(np.diff(dummy[:, i],axis =0), label=f"Neuron {i}")
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.title("RNN hidden state evolution (first 5 neurons)")
#plt.legend()
plt.grid(True)
plt.show()