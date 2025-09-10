# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:12:35 2025

@author: anas.skalli
"""

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from datetime import datetime
import time
import sys
import pickle
import gc

# Add your module path
sys.path.append('G:/Utilisateurs/anas.skalli/Desktop/Projects/nonlinear_oscillator_network/Utils')

from NN_utils import *
from optimization_algorithms import *

def run_single_experiment(n_neurons, s, n_epochs, return_dict):
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms

    # Reinitialize DataLoader inside the process
    transform_data = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_data, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_data, download=True)
    
    train_loader_MNIST = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader_MNIST = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

    # Initialize model
    RNN_params = {
        "N_in": 784,
        "N_out": 10,
        "N_neurons": n_neurons,
        "N_layers": 3
    }

    model = Oscillator_RNN_parallel(params=RNN_params).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.init_esn_weights(reservoir=True)
    model.dt = 0.1
    model.eps_int = 1e-4
    model.alpha = 3
    model.max_steps = 40
    model.save_activations = False
    
    loss = nn.CrossEntropyLoss()
    N_dim = model.count_parameters()
    init_pos = model.get_params()
    
    pop_size = int(0.01*N_dim)
    #pop_size = 100
    PEPG_optimizer = PEPG_opt(N_dim, pop_size = pop_size, learning_rate=0.01, starting_mu=init_pos ,starting_sigma=1e-1)
    
    PEPG_optimizer.sigma_decay = 0.9999
    PEPG_optimizer.sigma_alpha=0.2
    PEPG_optimizer.sigma_limit=0.01
    PEPG_optimizer.elite_ratio=0.1
    PEPG_optimizer.weight_decay=0.005

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Neurons: {n_neurons}, Run: {s+1}, Params: {model.count_parameters()}", flush=True)

    result = train_online_pop_parallel(model, n_epochs, train_loader_MNIST, test_loader_MNIST, loss, PEPG_optimizer)

    return_dict[(n_neurons, s)] = result
    
    # Clean up
    del model
    del PEPG_optimizer
    torch.cuda.empty_cache()
    gc.collect()

# ---- Main script ---- #
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    N_neurons_vec = [100]
    #N_neurons_vec = [100]
    n_epochs = 150
    stats = 1

    print("Script started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    start_time = time.time()
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for s in range(stats):
        for n_neurons in N_neurons_vec:
            p = mp.Process(target=run_single_experiment, args=(n_neurons, s, n_epochs, return_dict))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    results = [return_dict[key] for key in sorted(return_dict.keys())]
    print("All results collected successfully.")
    print(f"Total time = {time.time() - start_time:.2f} s")
    
    with open('results/1000_overfit_batch128_popsize1percent_PNN_PEPG_MNIST.pkl', 'wb') as f:
        pickle.dump(results, f)  

    analyze_and_plot(stats, N_neurons_vec, results, top_k=10)
    #takes 500s per epoch
    
    
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
    
    for k, n_neurons in enumerate(N_neurons_vec):
        # Train loss
        axs[0].loglog(results[k]['train_loss'], label=f"{n_neurons}")
        axs[0].set_title("Train Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
    
        # Test loss
        axs[1].loglog(results[k]['test_loss'], label=f"{n_neurons}")
        axs[1].set_title("Test Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
    
    axs[0].legend(title="N_neurons")
    axs[1].legend(title="N_neurons")
    
    plt.tight_layout()
    plt.show()
