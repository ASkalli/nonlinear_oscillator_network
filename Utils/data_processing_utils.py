#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 08:24:30 2025

@author: anas
"""


import os, gc
import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
from NN_utils import *
from kneed import KneeLocator
import scipy
import time


# some useful processing functions so I stop copy pasting bits of code like a crazy person ...


def average_data_dicts(dicts_list):
    """
    Parameters
    ----------
    dicts_list : list
        list of the dictionaries containing data of the learning runs I want to average
    
    Returns
    avg_dic
    
    """
    # Average them
    avg_results = []
    for i in range(len(dicts_list[0])):          # loop over neuron settings
        avg_dict = {}
        keys = dicts_list[0][i].keys()           # e.g. 'train_loss', 'test_loss'
        for key in keys:
            stacked = np.stack([d[i][key] for d in dicts_list])  # shape: (n_runs, vec_len)
            avg_dict[key] = stacked.mean(axis=0)
        avg_results.append(avg_dict)
    
    return avg_results


def locate_curve_knee(avg_results_dict,N_neurons_vec,idx_start,idx_run=0,window_len=100,plot_bool=True):
    
    """
    
    """
    
    
    knee_idx_vec = []
    
    
    test_loss_mat = np.zeros((len(N_neurons_vec),len(avg_results_dict[idx_run]['test_loss'])))
    train_loss_mat = np.zeros((len(N_neurons_vec),len(avg_results_dict[idx_run]['train_loss'])))
    test_loss_smooth_mat = np.zeros((len(N_neurons_vec),len(avg_results_dict[idx_run]['test_loss'])))
    test_acc_mat = np.zeros((len(N_neurons_vec),len(avg_results_dict[idx_run]['test_acc'])))
    
    knee_idx_vec = []
    
    for k,D in enumerate(avg_results_dict):
    
        test_loss_mat[k,:len( D['test_loss'])] = D['test_loss']
        train_loss_mat[k,:len(D['train_loss'])] = D['train_loss']
        
        test_loss_smooth = scipy.signal.savgol_filter(test_loss_mat[k,:len( D['test_loss'])],window_length=100,polyorder=3,axis=-1)
        test_loss_smooth_mat[k,:len(test_loss_smooth)] = test_loss_smooth
        test_acc_mat[k,:len(D['test_acc'])] = D['test_acc']
    
        kneedle = KneeLocator(
            np.arange(idx_start,np.shape(test_loss_smooth)[-1]), test_loss_smooth[idx_start:],
            S=1.0,                     # sensitivity (larger -> fewer knees)
            curve="convex",            # try "concave" if your curve bends the other way
            direction="decreasing",    # loss goes down with epochs
            online=False
        )
    
        knee_idx_vec.append(int(kneedle.knee + idx_start)) 
    
    knee_idx_vec = np.array(knee_idx_vec)
    
    if plot_bool:
        loss_cure_conv = test_loss_smooth_mat[np.arange(len(N_neurons_vec)),knee_idx_vec]
        acc_curve_conv = test_acc_mat[np.arange(len(N_neurons_vec)),knee_idx_vec]
        
        plt.plot(test_loss_smooth_mat.T)
        plt.scatter(knee_idx_vec,loss_cure_conv)
            
    
    return knee_idx_vec,loss_cure_conv,acc_curve_conv


def process_cost_to_conv(avg_results_dict,knee_idx_vec,algo_mem_perepoch,plot_bool=True):
    
    
    N_dim_vec_alg = []
    for k in range(len(avg_results_dict)):
        N_dim = int(avg_results_dict[k]['n_params'])
        N_dim_vec_alg.append(N_dim)
    
    alg_cost_to_conv = algo_mem_perepoch * knee_idx_vec
    
    
    a, b = np.polyfit(np.log10(N_dim_vec_alg), np.log10(alg_cost_to_conv), 1)  # linear fit
    fit_data    = 10**(a   * np.log10(N_dim_vec_alg) + b)

    print(f'slope {a}')
    print(f'intercept {b}')
    print(f'NUms params : {N_dim_vec_alg}')
    
    if plot_bool:
        plt.figure(figsize=(4, 3))
        plt.loglog(N_dim_vec_alg,alg_cost_to_conv,'o',color = 'dimgray')
        plt.loglog(N_dim_vec_alg, fit_data, '--',  color='dimgray', label=f'Fit slope={a:.2f}')
        plt.ylabel('Cost to convergence [GB]')
        plt.xlabel('# parameters')
        plt.xlabel(r'# parameters', fontsize=12)

        plt.legend()
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.tight_layout()
        plt.show()
    return a,b,N_dim_vec_alg,alg_cost_to_conv











































