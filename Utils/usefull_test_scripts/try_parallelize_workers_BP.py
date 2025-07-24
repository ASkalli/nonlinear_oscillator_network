import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from datetime import datetime
import time
import sys

# Add your module path
sys.path.append('C:/Users/Admin/Desktop/PhD/simulation/simulation_python/nonlinear_oscillator_network/Utils')

from NN_utils import *
from optimization_algorithms import *

def run_single_experiment_BP(n_neurons, s, n_epochs, return_dict):
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms

    # Reinitialize DataLoader inside the process
    transform_data = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_data, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_data, download=True)
    
    train_loader_MNIST = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)
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



    print(f"[{datetime.now().strftime('%H:%M:%S')}] Neurons: {n_neurons}, Run: {s+1}, Params: {model.count_parameters()}", flush=True)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
    result = train_BP_torch(model, n_epochs, train_loader_MNIST, test_loader_MNIST, loss, optimizer)

    return_dict[(n_neurons, s)] = result

# ---- Main script ---- #
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    N_neurons_vec = [5, 10, 20, 30, 50, 75, 100]
    n_epochs = 20
    stats = 1

    print("Script started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    start_time = time.time()
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for s in range(stats):
        for n_neurons in N_neurons_vec:
            p = mp.Process(target=run_single_experiment_BP, args=(n_neurons, s, n_epochs, return_dict))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    results = [return_dict[key] for key in sorted(return_dict.keys())]
    print("All results collected successfully.")
    print(f"Total parallel loop time = {time.time() - start_time:.2f} s")
    
    
    


test_loss_vec = []
test_loss_std = []
test_acc_vec = []
test_acc_std = []
train_loss_vec = []
train_loss_std = []
n_params = []

dummy = np.reshape(results, [stats, len(N_neurons_vec)])
top_k = 10  # Number of best points to average over

for i in range(len(N_neurons_vec)):
    train_loss_runs = []
    test_loss_runs = []
    test_acc_runs = []
    
    for j in range(stats):
        train_loss = np.array(dummy[j, i]['train_loss'])
        test_loss = np.array(dummy[j, i]['test_loss'])
        test_acc = np.array(dummy[j, i]['test_acc'])

        # Take mean over best 10 points for each metric
        best_train_loss = np.mean(np.sort(train_loss)[:top_k])
        best_test_loss = np.mean(np.sort(test_loss)[:top_k])
        best_test_acc = np.mean(np.sort(test_acc)[-top_k:])  # top accuracy

        train_loss_runs.append(best_train_loss)
        test_loss_runs.append(best_test_loss)
        test_acc_runs.append(best_test_acc)

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
plt.loglog(n_params, test_acc_vec, '-o', label='Test Accuracy')
plt.fill_between(n_params,
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
plt.loglog(n_params, test_loss_vec, '-o', label='Test Loss')
plt.fill_between(n_params,
                 np.array(test_loss_vec) - np.array(test_loss_std),
                 np.array(test_loss_vec) + np.array(test_loss_std),
                 alpha=0.3)

plt.loglog(n_params, train_loss_vec, '-o', label='Train Loss')
plt.fill_between(n_params,
                 np.array(train_loss_vec) - np.array(train_loss_std),
                 np.array(train_loss_vec) + np.array(train_loss_std),
                 alpha=0.3)

plt.xlabel('Number of parameters')
plt.ylabel('CCE Loss')
plt.grid(True)
plt.legend()
plt.show()