# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:15:20 2025

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


# --- Parameters ---
RNN_params = {
    "N_in": 784,
    "N_out": 10,
    "N_neurons": 30,
    "N_layers": 3,
    "time_steady_state": 15
}


model = Custom_RNN(params=RNN_params)

# Set model to eval mode and CPU (or use GPU if preferred)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# --- Load one FashionMNIST sample ---
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
img, label = dataset[0]  # take one sample

# Flatten and prepare repeated input
x = img.view(-1).unsqueeze(0).to(device)  # (1, 784)
T = RNN_params["time_steady_state"]

# --- Forward Pass ---
with torch.no_grad():
    _ = model(x,T)

layer = 'layer1'
# --- Plot Hidden State Over Time (first neuron for example) ---
rnn_out = torch.stack(model.activations[layer]).squeeze().numpy()  # shape: (T, hidden_size)
plt.figure(figsize=(10, 5))
for i in range(30):  # plot first 5 neurons
    plt.plot(rnn_out[:, i], label=f"Neuron {i}")
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.title("RNN hidden state evolution (first 5 neurons)")
#plt.legend()
plt.grid(True)
plt.show()

dummy = rnn_out
plt.figure(figsize=(10, 5))
for i in range(30):  # plot first 5 neurons
    plt.semilogy(np.diff(dummy[:, i],axis =0), label=f"Neuron {i}")
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.title("RNN hidden state evolution (first 5 neurons)")
#plt.legend()
plt.grid(True)
plt.show()