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
    
    apply_PCA = False
    number_PCs = 81
    
    transform_data = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_data, download=True)
    test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform_data, download=True)
    
    if apply_PCA:
        # Avoid giant DataLoader just to get arrays: use the raw tensors on the datasets
        X_train = (train_dataset.data.view(-1, 28*28).float() / 255.0).numpy()
        Y_train = train_dataset.targets.numpy()
        X_test  = (test_dataset.data.view(-1, 28*28).float() / 255.0).numpy()
        Y_test  = test_dataset.targets.numpy()
    
        # Fit PCA on train, transform train & test
        pca = PCA_analysis(X_train)
        X_train_pca = pca.transform(X_train, number_PCs)
        X_test_pca  = pca.transform(X_test,  number_PCs)
    
        # Wrap back into your Custom_dataset (expects numpy arrays)
        train_dataset = Custom_dataset(X_train_pca, Y_train)
        test_dataset  = Custom_dataset(X_test_pca,  Y_test)
    
    # Now make normal loaders for training/eval (avoid workers inside Windows subprocess unless needed)
    train_loader_MNIST = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True,  pin_memory=True, num_workers=0)
    test_loader_MNIST  = torch.utils.data.DataLoader(test_dataset,  batch_size=10000, shuffle=False, pin_memory=True, num_workers=0)

    # Initialize model
    input_dim = number_PCs if apply_PCA else 784
    RNN_params = {
        "N_in": input_dim,
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
    
    if init_pos.requires_grad:
    # Detach the tensor from the computation graph
        init_pos = init_pos.detach()
    if init_pos.is_cuda:
        # Move the tensor to the CPU
        init_pos = init_pos.cpu()
    init_pos = init_pos.numpy()
    
    pop_size = int(0.01*N_dim)
    #pop_size = 100

    CMA_optimizer = CMA_opt(N_dim, pop_size, select_pop=int(pop_size/2), sigma_init=0.01, mean_init=init_pos)
    #CMA_optimizer.eigen_update_frequency = N_dim // 10

                

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Neurons: {n_neurons}, Run: {s+1}, Params: {model.count_parameters()}", flush=True)

    result = train_online_pop_parallel(model, n_epochs, train_loader_MNIST, test_loader_MNIST, loss, CMA_optimizer)

    return_dict[(n_neurons, s)] = result
    
    # Clean up
    #del model
    #del PEPG_optimizer
    #torch.cuda.empty_cache()
    #gc.collect()

# ---- Main script ---- #
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    N_neurons_vec =  [5, 10, 20 ,30, 50, 75, 100]
    N_neurons_vec = [30]
    n_epochs = 500
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
    
    # ### stuff for debugging without the multiprocess parallelization
    # n_neurons = N_neurons_vec[0]
    # s = 0
    # n_epochs = 3  # shorter while debugging
    
    # return_dict = {}  # plain dict
    # run_single_experiment(n_neurons, s, n_epochs, return_dict)
    
    # results = [ return_dict[(n_neurons, s)] ]
    # ###
    
    results = [return_dict[key] for key in sorted(return_dict.keys())]
    print("All results collected successfully.")
    print(f"Total time = {time.time() - start_time:.2f} s")
    
    with open('results/30_PNN_CMA_MNIST_run1.pkl', 'wb') as f:
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
