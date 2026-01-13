import numpy as np
from kneed import KneeLocator
%load_ext memory_profiler
import os, gc
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import pickle
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import sys
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')
from NN_utils import *
import scipy
#from torchvision import datasets, transforms
#from torchsummary import summary
import time
from optimization_algorithms import * 




def average_data_dicts(dicts_list,scanned_param):
"""
Takes a list of dictionaries containing the training data and averages them
"""
    # Average them
    avg_results = []
    for i in range(len(scanned_param)):          # loop over the param that was studied  e.g: neuron numebr
        avg_dict = {}
        keys = dicts_list[0][i].keys()           # e.g. 'train_loss', 'test_loss'
        for key in keys:
            stacked = np.stack([d[i][key] for d in dicts_list])  # shape: (n_runs, vec_len)
            avg_dict[key] = stacked.mean(axis=0)
        avg_results.append(avg_dict)

    return avg_results


def epochs_to_conv(avg_results,if_plot=False):

    """

    Returns:
        knee_idx_vec: array with the indices of the knee points for each convergence curve from
        the avg_dict
    """

    test_loss_mat = np.zeros((len(N_neurons_vec),len(avg_results[0]['test_loss'])))
    train_loss_mat = np.zeros((len(N_neurons_vec),len(avg_results[0]['train_loss'])))

    for k,D in enumerate(avg_results):

        test_loss_mat[k,:] = D['test_loss']
        train_loss_mat[k,:] = D['train_loss']


    test_loss_smooth = scipy.signal.savgol_filter(test_loss_mat,window_length=100,polyorder=3,axis=-1)
    knee_idx_vec = []
    for k in range(len(N_neurons_vec)):
        kneedle = KneeLocator(
            np.arange(idx_start,np.shape(test_loss_smooth)[-1]), test_loss_smooth[k,idx_start:len(avg_results[k]['test_loss'])+1],
            S=1.0,                     # sensitivity (larger -> fewer knees)
            curve="convex",            
            direction="decreasing",    # loss goes down with epochs
            online=False
        )

        knee_idx_vec.append(int(kneedle.knee + idx_start)) 

    if if_plot:
        plt.loglog(test_loss_smooth.T)
        plt.scatter(knee_idx,test_loss_smooth[np.arange(len(N_neurons_vec)),knee_idx])


    return knee_idx_vec




def fit_convergence_slope(N_dim_vec,mem_cost_vec,knee_idx_vec):

    cost_to_conv = mem_cost_vec * knee_idx_vec
    a, b = np.polyfit(np.log10(N_dim_vec), np.log10(cost_to_conv), 1)  # linear fit

    return


#memory cost computation 
def peak_mem_increment(func, *args, interval=0.05, **kwargs):
    # Track process memory over time while func runs; return (peak - baseline) MiB
    baseline = memory_usage(max_iterations=1)[0]
    mem_series = memory_usage((func, args, kwargs), interval=interval)
    return max(mem_series) - baseline

def mem_cost_pop(optimizer,avg_results,scanned_param):
    """
    returns memory cost for population based algos
    """
    mem_cost_pop = []
    N_dim_vec=[]
    for k,n in enumerate(scanned_param):
        X = []
        N_dim = int(avg_results[k]['n_params'])
        N_dim_vec.append(N_dim)
        pop_size = int((0.01*N_dim))
        pop_size = pop_size + 1 if pop_size % 2 else pop_size

        coordinates = np.random.randn(pop_size,N_dim)
        init_pos = np.random.randn(N_dim,1)
        rewards = np.random.randn(pop_size,1)

        #PEPG_optimizer = PEPG_opt(N_dim, pop_size = pop_size, learning_rate=0.01, starting_mu=init_pos ,starting_sigma=1e-1)
        mem_load2 = peak_mem_increment(optimizer.ask, interval=0.05)

        mem_load1 = peak_mem_increment(optimizer.tell, rewards, interval=0.05)

        mem_cost_pop.append(mem_load1 + mem_load2)

        del optimizer
    #conversion because it's in MiB NOT MB ... big yikes
    mem_cost_pop = np.array(mem_cost_pop) * 1048576/1e6

    return N_dim_vec,mem_cost_pop


def mem_cost_SPSA(optimizer,avg_results,scanned_param):
    """
    returns memory cost for population based algos
    """
    mem_cost_pop = []
    N_dim_vec=[]
    for k,n in enumerate(scanned_param):
        X = []
        N_dim = int(avg_results[k]['n_params'])
        N_dim_vec.append(N_dim)
        pop_size = int((0.01*N_dim))
        pop_size = pop_size + 1 if pop_size % 2 else pop_size

        coordinates = np.random.randn(pop_size,N_dim)
        init_pos = np.random.randn(N_dim,1)
        rewards = np.random.randn(pop_size,1)

        #PEPG_optimizer = PEPG_opt(N_dim, pop_size = pop_size, learning_rate=0.01, starting_mu=init_pos ,starting_sigma=1e-1)
        mem_load2 = peak_mem_increment(optimizer.ask, interval=0.05)

        mem_load1 = peak_mem_increment(optimizer.tell, rewards, interval=0.05)

        mem_cost_pop.append(mem_load1 + mem_load2)

        del optimizer
    #conversion because it's in MiB NOT MB ... big yikes
    mem_cost_pop = np.array(mem_cost_pop) * 1048576/1e6

    return N_dim_vec,mem_cost_pop


