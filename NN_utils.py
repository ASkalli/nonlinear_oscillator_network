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
            if i%10==0:
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
        'time' : train_time
        
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










