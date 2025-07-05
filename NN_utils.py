# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:59:41 2025

@author: Anas Skalli
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import pdb



class Base_Model(nn.Module):
    
    def __init__(self):
        super(Base_Model, self).__init__()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_params(self):
        return torch.cat([
            p.view(-1).detach().cpu() for p in self.parameters()
        ])

    def set_params(self, params_to_send):
        current_idx = 0
        for param in self.parameters():
            n = param.numel()
            new_param = torch.from_numpy(params_to_send[current_idx:current_idx + n]).view(param.shape)
            param.data.copy_(new_param)
            current_idx += n

    def forward_pass_params(self, params_to_send, X):
        self.set_params(params_to_send)
        return self.forward(X)


class RNN_network(Base_Model):
    def __init__(self, params):
        super(RNN_network, self).__init__()

        self.N_in = params["N_in"]
        self.N_out = params["N_out"]
        self.N_neurons = params["N_neurons"]
        self.N_layers = params["N_layers"]
        self.T_SS = params["time_steady_state"]
        
        self.activations = {}

        self.rnn = nn.RNN(
            input_size=self.N_in,
            hidden_size=self.N_neurons,
            num_layers=self.N_layers,
            nonlinearity='tanh',
            batch_first=True
        )

        self.fc_out = nn.Linear(self.N_neurons, self.N_out, bias=True)

        # Optional ESN-style weight customization
        spectral_radius = 0.8
        sparsity = 0.2

    def init_esn_weights(self, spectral_radius=0.8, sparsity=0.2,reservoir = False):
        with torch.no_grad():
            for layer in range(self.N_layers):
                W = (2 * torch.rand(self.N_neurons, self.N_neurons) - 1)
                W *= (torch.rand_like(W) < sparsity)
                eigvals = torch.linalg.eigvals(W).abs().max()
                W *= spectral_radius / eigvals
    
                # Set weight_hh_lX
                getattr(self.rnn, f"weight_hh_l{layer}").copy_(W)
                if reservoir == True:
                    # Optional: freeze recurrent weights if you want pure ESN behavior
                    getattr(self.rnn, f"weight_hh_l{layer}").requires_grad = False

    
    def forward(self, X, T=None):
        if T is None:
            T=self.T_SS
        X = X.view(X.size(0), -1)
        X_seq = X.unsqueeze(1).repeat(1, T, 1)  # repeat same input T times to ensure steadystate for memory independant tasks
        rnn_out, _ = self.rnn(X_seq)           # evolve dynamics
        steady_output = rnn_out[:, -1, :]      # get last state (steady)
        out = self.fc_out(steady_output)       # classify from it
        return out

    
    def save_activation(self,name):
        def hook(module, input, output):
            # If output is a tuple, take the first item
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach().cpu()
        return hook



def train_BP_torch(model, n_epochs, train_loader, test_loader, loss, optimizer):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print(f"Using {device} device")
    print(model)

    #array to store the accuracy of the model
    
    test_acc = []
    train_loss = []
    test_loss = []
    
    #dict to return
    
    data_dict = {}

    start_time = time.time()
    for epoch in range(n_epochs):
        
        for i,(images, labels) in enumerate(train_loader):
            
            model.train()
            #move data to gpu for faster processing
            images = images.to(device)
            labels = labels.to(device)
            
            #forward pass
            Y_pred = model.forward(images)
            loss_value = loss(Y_pred,labels)
            train_loss_value =loss_value.item()
            if i%5==0:
                train_loss.append(train_loss_value)
            
            #backward pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
                
            # evaluate after some epochs
            if i == 0 or (i+1) % 50 == 0:
                test_loss_minibatches = []
                model.eval()
                correct = 0 
                total = 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        Y_pred = model.forward(images)
                        _, predicted = torch.max(Y_pred.data, 1)
                        
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        test_loss_minibatches.append(loss(Y_pred,labels).item())
                    
                    accuracy = ( 100*correct/total)
                    test_acc.append(accuracy)
                    test_loss.append(np.mean(test_loss_minibatches))
                    
                
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_value.item()}, Test Accuracy: {accuracy}%')


    end_time = time.time()
    train_time = end_time - start_time
    print(f'Total time: {(end_time - start_time)/3600}h')
    
    data_dict = {
        'train_loss' : train_loss,
        'test_loss':test_loss ,
        'test_acc' : test_acc ,
        'time' : train_time,
        'n_params': model.count_parameters()
        
        }
    return data_dict




class Custom_RNN(Base_Model):
    def __init__(self, params):
        super(Custom_RNN, self).__init__()

        self.N_in = params["N_in"]
        self.N_out = params["N_out"]
        self.N_neurons = params["N_neurons"]
        self.N_layers = params["N_layers"]
        self.T_SS = params["time_steady_state"]
        
        self.activations = {}

        self.W_in = nn.Linear(in_features=self.N_in, out_features = self.N_neurons)
        self.W_recurrent = nn.ModuleList([
            nn.Linear(self.N_neurons, self.N_neurons)
            for _ in range(self.N_layers)
        ])
        self.W_out = nn.Linear(in_features = self.N_neurons, out_features = self.N_out)

        

        # Optional ESN-style weight customization
        spectral_radius = 0.8
        sparsity = 0.2

    def init_esn_weights(self, spectral_radius=0.8, sparsity=0.2, reservoir=False):
        with torch.no_grad():
            for layer in range(self.N_layers):
                W = (2 * torch.rand(self.N_neurons, self.N_neurons) - 1)
                W *= (torch.rand_like(W) < sparsity)
                eigvals = torch.linalg.eigvals(W).abs().max()
                W *= spectral_radius / eigvals

                self.W_recurrent[layer].weight.copy_(W)
                if reservoir:
                    self.W_recurrent[layer].weight.requires_grad = False

    
    def forward(self, X, T=None):
        if T is None:
            T = self.T_SS
        
        #self.activations = {f"layer{l}": [] for l in range(self.N_layers)}

        batch_size = X.size(0)
        h = [torch.zeros(batch_size, self.N_neurons, device=X.device) for _ in range(self.N_layers)]
        x_in = self.W_in(X.view(X.size(0), -1))

        for t in range(T):
            h[0] = torch.sin(x_in + self.W_recurrent[0](h[0]))
            #self.activations["layer0"].append(h[0].detach().cpu().clone())
            for l in range(1, self.N_layers):
                h[l] = torch.sin(h[l-1] + self.W_recurrent[l](h[l]))
                #self.activations[f"layer{l}"].append(h[l].detach().cpu().clone())
        out = self.W_out(h[-1])
        return out
    
    
class Oscillator_RNN(Base_Model):
    def __init__(self, params):
        super(Oscillator_RNN, self).__init__()

        self.N_in = params["N_in"]
        self.N_out = params["N_out"]
        self.N_neurons = params["N_neurons"]
        self.N_layers = params["N_layers"]
        self.T_SS = params["time_steady_state"]
        
        self.activations = {}

        self.W_input = nn.Linear(in_features=self.N_in, out_features = self.N_neurons)
        
        self.W_in = nn.ModuleList([
            nn.Linear(self.N_neurons, self.N_neurons)
            for _ in range(self.N_layers)
        ])
        self.W_recurrent = nn.ModuleList([
            nn.Linear(self.N_neurons, self.N_neurons)
            for _ in range(self.N_layers)
        ])
        self.W_out = nn.Linear(in_features = self.N_neurons, out_features = self.N_out)
        
        self.alpha = 0.9
        

        # Optional ESN-style weight customization
        spectral_radius = 0.8
        sparsity = 0.2

    def init_esn_weights(self, spectral_radius=0.8, sparsity=0.2, reservoir=False):
        with torch.no_grad():
            for layer in range(self.N_layers):
                W = (2 * torch.rand(self.N_neurons, self.N_neurons) - 1)
                W *= (torch.rand_like(W) < sparsity)
                eigvals = torch.linalg.eigvals(W).abs().max()
                
                if eigvals < 1e-6:
                    eigvals = 1e-6  # prevent divide by zero
                W *= spectral_radius / eigvals

                self.W_recurrent[layer].weight.copy_(W)
                
                W = (2 * torch.rand(self.N_neurons, self.N_neurons) - 1)
                #W *= (torch.rand_like(W) < sparsity)
                eigvals = torch.linalg.eigvals(W).abs().max()
                eigvals = torch.linalg.eigvals(W).abs().max()
                if eigvals < 1e-6:
                    eigvals = 1e-6  # prevent divide by zero

                W *= spectral_radius / eigvals
                
                self.W_in[layer].weight.copy_(W)
                
                # self.W_input.bias.data.zero_()
                # self.W_out.bias.data.zero_()
               
                # self.W_in[layer].bias.data.zero_()
                # self.W_recurrent[layer].bias.data.zero_()
                
                if reservoir:
                    self.W_recurrent[layer].weight.requires_grad = False

    
    
    def forward(self, X, T=None,dt = 0.1):
        if T is None:
            T = self.T_SS
        
        self.activations = {f"layer{l}": [] for l in range(self.N_layers)}

        batch_size = X.size(0)
        h = [torch.zeros(batch_size, self.N_neurons, device=X.device) for _ in range(self.N_layers)]
        dh = [torch.zeros(batch_size, self.N_neurons, device=X.device) for _ in range(self.N_layers)]
        
        x_in = self.W_input(X.view(X.size(0), -1))

        for t in range(T):
            dh[0] = -self.alpha*h[0] + torch.sin(x_in + self.W_recurrent[0](h[0]))
            h[0] = h[0] + dh[0]*dt
            
            self.activations["layer0"].append(h[0].detach().cpu().clone())
            
            for l in range(1, self.N_layers):
                dh[l] = -self.alpha*h[l] +  torch.sin(self.W_in[l](h[l-1]) + self.W_recurrent[l](h[l]))
                h[l] = h[l] + dh[l]*dt
                self.activations[f"layer{l}"].append(h[l].detach().cpu().clone())
        out = self.W_out(h[-1])
        return out
 
    
class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features,spectral_radius = 0.8, sparsity=0.2, bias=True,sym = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if sym:
            mask = self.gen_sparse_sym_mask(in_features, out_features, sparsity)
        else:
            mask = (torch.rand(out_features, in_features) < sparsity).float()
            
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        
        self.register_buffer('mask', mask)
        
        
        # Initialize weights: uniform between -1 and 1
        W = (2 * torch.rand(out_features, in_features) - 1) * mask
        
        # Spectral normalization (if square matrix)
        if in_features == out_features:
            eigvals = torch.linalg.eigvals(W).abs().max()
            eigvals = eigvals if eigvals > 1e-6 else 1e-6  # Avoid division by zero
            W *= spectral_radius / eigvals
        
        self.weight.data = W
        
        
        
        
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

        # Ensure masked weights have zero gradients
        self.weight.register_hook(lambda grad: grad * self.mask)

    def forward(self, X):
        return F.linear(X, self.weight * self.mask, self.bias)
    
    def gen_sparse_sym_mask(self,in_features, out_features,sparsity=0.2):
        assert in_features == out_features, "Symmetric mask requires square weight matrix."
        
        size = in_features
        mask = torch.zeros(size, size)
        
        # Number of unique pairs for sparsity (upper triangular without diagonal)
        num_elements = size * (size - 1) // 2 + size
        num_active = int(sparsity * num_elements)

        # Get upper triangular indices
        triu_indices = torch.triu_indices(size, size)
        idx = torch.randperm(triu_indices.shape[1])[:num_active]

        selected_rows = triu_indices[0, idx]
        selected_cols = triu_indices[1, idx]

        mask[selected_rows, selected_cols] = 1
        mask[selected_cols, selected_rows] = 1  # Symmetry
        
        #diagonal set to ones for self coupling
        
        mask = mask + torch.eye(size)
        
        
        return mask
    
    
class Oscillator_RNN_dyn(Base_Model):
    def __init__(self, params):
        super(Oscillator_RNN_dyn, self).__init__()

        self.N_in = params["N_in"]
        self.N_out = params["N_out"]
        self.N_neurons = params["N_neurons"]
        self.N_layers = params["N_layers"]
        self.T_SS = params["time_steady_state"]
        
        self.alpha = 0.9
        
        self.eps_int = 5e-2
        self.dt = 0.05
        self.max_steps = 1000
        self.k = 0
        self.k_vec = []
        
        self.sparsity = 0.2
        self.spectral_radius = 0.8
        
        self.activations = {}
        self.save_activations = False
        
        self.W_input = nn.Linear(in_features=self.N_in, out_features = self.N_neurons)
        
        self.W_in = nn.ModuleList([
            nn.Linear(self.N_neurons, self.N_neurons)
            for _ in range(self.N_layers)
        ])
    
        self.W_recurrent =  nn.ModuleList([
            SparseLinear(self.N_neurons, self.N_neurons,sparsity=0.2,bias=True,sym=True)
            for _ in range(self.N_layers)
        ])
        
        self.W_out = nn.Linear(in_features = self.N_neurons, out_features = self.N_out)
        


    def init_esn_weights(self, spectral_radius=0.8, sparsity=0.2, reservoir=False):
        
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        
        with torch.no_grad():
            
            W = (2 * torch.rand(self.N_neurons, self.N_in) - 1)/100
            #W *= self.W_recurrent[layer].mask

            
            self.W_input.weight.data.copy_(W)
            
            for layer in range(self.N_layers):
                # W = (2 * torch.rand(self.N_neurons, self.N_neurons) - 1)
                # W *= self.W_recurrent[layer].mask
                # eigvals = torch.linalg.eigvals(W).abs().max()
                
                # if eigvals < 1e-6:
                #     eigvals = 1e-6
                
                # W *= self.spectral_radius / eigvals
                # self.W_recurrent[layer].weight.copy_(W)
                
                W = (2 * torch.rand(self.N_neurons, self.N_neurons) - 1)
                #W *= self.W_recurrent[layer].mask
                eigvals = torch.linalg.eigvals(W).abs().max()
                
                if eigvals < 1e-6:
                    eigvals = 1e-6
                
                W *= self.spectral_radius / eigvals
                
                self.W_in[layer].weight.copy_(W)
                
                # self.W_input.bias.data.zero_()
                # self.W_out.bias.data.zero_()
               
                # self.W_in[layer].bias.data.zero_()
                # self.W_recurrent[layer].bias.data.zero_()
                
                if reservoir:
                    self.W_recurrent[layer].weight.requires_grad = False

    
    
    def forward(self, X, eps_int=None, dt=None,save_activations = None):
        if eps_int is None:
            eps_int = self.eps_int
        
        if dt is None: 
            dt = self.dt
        if save_activations is None:
            save_activations = self.save_activations
        
        if save_activations:
            self.activations = {f"layer{l}": [] for l in range(self.N_layers)}
    
        batch_size = X.size(0)
    
        # Initialize hidden states: shape (N_layers, batch_size, N_neurons)
        h = [torch.zeros(batch_size, self.N_neurons, device=X.device) for _ in range(self.N_layers)]
        dh = [torch.zeros(batch_size, self.N_neurons, device=X.device) for _ in range(self.N_layers)]
    
        # Input projection
        x_in = self.W_input(X.view(batch_size, -1))
    
    
        
        self.k=0
        while True:
            self.k+=1
            # Layer 0 update
            dh[0] = -self.alpha * h[0] + torch.sin(x_in + self.W_recurrent[0](h[0]))
            h[0] = h[0] + dh[0] * dt
            if save_activations:
                self.activations["layer0"].append(h[0].detach().cpu().clone())
    
            # Updates for other layers
            for l in range(1, self.N_layers):
                dh[l] = -self.alpha * h[l] + torch.sin(
                    self.W_in[l](h[l-1]) + self.W_recurrent[l](h[l])
                )
                h[l] = h[l] + dh[l] * dt
                if save_activations:
                    self.activations[f"layer{l}"].append(h[l].detach().cpu().clone())
                
            
            # if self.k >= 1000:
            #     print(f"[Warning] High number of steps: {self.k}")
            #     pdb.set_trace()
                # layer = 'layer1'

                # rnn_out = torch.stack(self.activations[layer]).squeeze().numpy()  # shape: (T, hidden_size)
                # plt.plot(rnn_out[:,50,:])

                
            # Steady-state check
            with torch.no_grad():
                max_delta = max(torch.max(torch.abs(dh_layer)) for dh_layer in dh)
            if max_delta < eps_int or self.k >= self.max_steps:
                if self.k >= self.max_steps:
                    print(f"[Warning] Max steps {self.max_steps} reached. max_delta={max_delta:.5f}")
                break
            
                
        self.k_vec.append(self.k)
        # Output projection from last layer
        out = self.W_out(h[-1])
        return out





def train_online_pop_NN(model, n_epochs, train_loader, test_loader, loss, optimizer):
    "function to train a model using the population based training algorithm,  returns the accuracy and best reward lists"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print(f"Using {device} device")
    print(model)
    
    best_reward = []
    #array to store the accuracy of the model
    
    test_acc = []
    train_loss = []
    test_loss = []
    
    #dict to return
    
    data_dict = {}
    start_time = time.time()
    for epoch in range(n_epochs):
        model.eval()
        for i, (features,labels) in enumerate(train_loader):
            
            coordinates = optimizer.ask()
            rewards_list = []
            for k in range(coordinates.shape[0]):
                if device == 'cuda':
                    features = features.to(device)
                    labels = labels.to(device)
                    Y_pred = model.forward_pass_params(coordinates[k,:],features)
                if device == 'cpu':
                    Y_pred = model.forward_pass_params(coordinates[k,:],features)    
                loss_value = loss(Y_pred,labels)
                rewards_list.append(loss_value.detach().cpu().item())
            
            rewards = np.array(rewards_list)[:,np.newaxis]
            optimizer.tell(rewards)
            best_params = coordinates[np.argmin(rewards),:]
            train_loss.append(np.min(rewards))
            #print('\r{i+1}',end='')
            #print accuracy every 100 steps for the test set
            
            if i == 0 or (i+1) % 50 == 0:
                test_loss_minibatches = []
                model.eval()
                correct = 0 
                total = 0
                with torch.no_grad():
                    for features, labels in test_loader:
                        features = features.to(device)
                        labels = labels.to(device)
                        Y_pred = model.forward_pass_params(best_params,features)
                        loss_value = loss(Y_pred,labels)
                        _, predicted = torch.max(Y_pred.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        test_loss_minibatches.append(loss(Y_pred,labels).item())
                        
                    accuracy = ( 100*correct/total)
                    test_acc.append(accuracy)
                    test_loss.append(np.mean(test_loss_minibatches))
                    print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_value.item()}, Test Accuracy: {accuracy}%')
    return data_dict
    

def train_online_SPSA_NN(model, n_epochs, train_loader, test_loader, loss, spsa_optimizer,adam_optimizer):
    "function to train a model using the population based training algorithm,  returns the accuracy and best reward lists"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print(f"Using {device} device")
    print(model)
    
    best_reward = []
    #array to store the accuracy of the model
    
    test_acc = []
    train_loss = []
    test_loss = []
    
    #dict to return
    
    data_dict = {}
    start_time = time.time()
    for epoch in range(n_epochs):
        model.eval()
        for i, (features,labels) in enumerate(train_loader):
            
            params_plus,params_minus = spsa_optimizer.perturb_parameters()
            
            
            if device == 'cuda':
                features = features.to(device)
                labels = labels.to(device)
                Y_pred_plus = model.forward_pass_params(params_plus,features)
                Y_pred_minus = model.forward_pass_params(params_minus,features)
            if device == 'cpu':
                Y_pred_plus = model.forward_pass_params(params_plus,features)
                Y_pred_minus = model.forward_pass_params(params_minus,features)
                
            loss_value_plus = loss(Y_pred_plus,labels)
            loss_value_minus = loss(Y_pred_minus,labels)
            
            reward_plus = loss_value_plus.detach().cpu().item()
            reward_minus = loss_value_minus.detach().cpu().item()
            
            grad_spsa = spsa_optimizer.approximate_gradient(reward_plus ,reward_minus)
            step = adam_optimizer.step(grad_spsa)
            
            current_params= spsa_optimizer.update_parameters_step(step)

            train_loss.append(np.min([reward_plus,reward_minus]))
            #print('\r{i+1}',end='')
            #print accuracy every 100 steps for the test set
            if i == 0 or (i+1) % 50 == 0:
                test_loss_minibatches = []
                model.eval()
                correct = 0 
                total = 0
                with torch.no_grad():
                    for features, labels in test_loader:
                        features = features.to(device)
                        labels = labels.to(device)
                        Y_pred = model.forward_pass_params(current_params,features)
                        loss_value = loss(Y_pred,labels)
                        _, predicted = torch.max(Y_pred.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        test_loss_minibatches.append(loss(Y_pred,labels).item())
                    accuracy = ( 100*correct/total)
                    test_acc.append(accuracy)
                    test_loss.append(np.mean(test_loss_minibatches))
                    print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_value.item()}, Test Accuracy: {accuracy}%')
    end_time = time.time()
    train_time = end_time - start_time
    print(f'Total time: {(end_time - start_time)/3600}h')
    
    data_dict = {
        'train_loss' : train_loss,
        'test_loss':test_loss ,
        'test_acc' : test_acc ,
        'time' : train_time,
        'n_params': model.count_parameters()
        
        }
    
    return data_dict







