# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 11:58:19 2025

@author: anas.skalli
"""

#simple Adam optimizer class
import numpy as np

class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = np.array(params)
        self.lr = lr  # Learning rate
        self.beta1 = beta1  # Decay rate for the first moment estimates
        self.beta2 = beta2  # Decay rate for the second moment estimates
        self.epsilon = epsilon  # Small value to prevent division by zero
        self.m = np.zeros_like(params)  # First moment vector
        self.v = np.zeros_like(params)  # Second moment vector
        self.iteration = 0  # Initialization of the timestep

    def step(self, grad):
        """Calculate and return the step to update parameters based on the Adam optimization algorithm."""
        self.iteration += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad  # Update biased first moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)  # Update biased second raw moment estimate

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.iteration)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.iteration)

        # Update parameters
        step = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return step
    
    
"""
Simple SPSA class for optimization, 

probably too many methods for such a simple algorithm


"""


class SPSA_opt:
    def __init__(self, params, alpha=0.01,epsilon=1e-5):
        self.params = np.array(params)  # Initial parameters to optimize
        self.alpha = alpha  # Step size
        self.epsilon = epsilon  # Perturbation size
        self.iteration = 0  # Track the current iteration
        self.delta = None

    def perturb_parameters(self):
        """Generate perturbation vector."""
        self.delta = (np.random.rand(*self.params.shape) > 0.5) * 2 - 1  # Random {-1, 1} for each parameter
        params_plus = self.params + self.epsilon * self.delta
        params_minus = self.params - self.epsilon * self.delta
        return params_plus,params_minus

    def approximate_gradient_func(self, loss_func):
        """Approximate the gradient of the loss function; if you have a loss function you can pass as parameter"""
        params_plus = self.params + self.epsilon  * self.delta
        params_minus = self.params - self.epsilon  * self.delta
        loss_plus = loss_func(params_plus)
        loss_minus = loss_func(params_minus)
        gradient = ((loss_plus - loss_minus) / (2 * self.epsilon  * np.var(self.delta))) *self.delta
        return gradient
    
    def approximate_gradient(self,loss_plus ,loss_minus):
        """Approximate the gradient of the loss function, with precalculated loss values"""
        if np.var(self.delta) == 0:
            gradient = (loss_plus - loss_minus) / (2 * self.epsilon) *self.delta
        else :
            gradient = ((loss_plus - loss_minus) / (2 * self.epsilon  * np.var(self.delta))) *self.delta
        return gradient
    
    def update_parameters(self, gradient):
        """Update the parameters based on the approximated gradient."""
        #ak = self.alpha / (self.iteration + 1)**0.602
        self.params -= self.alpha * gradient        
        return self.params
    
    def update_parameters_step(self, step):
        """Update the parameters based on the approximated gradient."""
        #ak = self.alpha / (self.iteration + 1)**0.602
        self.params -= step         
        return self.params
    
    
    
    
    
"""
Based on the purecmaes matlab code from Nikolaus Hansen:

https://cma-es.github.io/cmaes_sourcecode_page.html#matlab

"""

class CMA_opt:
    def __init__(self, N_dim, pop_size, select_pop, sigma_init, mean_init):
        # Ensure mean_init is a numpy array and has the correct shape
        if not isinstance(mean_init, np.ndarray):
            raise ValueError("mean_init must be a numpy array")
        if mean_init.shape not in [(N_dim,), (N_dim, 1)]:
            raise ValueError(f"mean_init must be of shape ({N_dim},) or ({N_dim}, 1)")

        # Reshape mean_init to a column vector if necessary
        self.xmean = mean_init.reshape(N_dim, 1) if mean_init.ndim == 1 else mean_init

        self.N_dim = N_dim
        self.sigma = sigma_init
        self.pop_size = pop_size
        self.select_pop = select_pop

        # Calculate weights for the top 'select_pop' individuals with a power law
        self.weights = np.array([np.log(select_pop + 0.5) - np.log(i) for i in range(1, select_pop + 1)])
        #calculate weights with a linear scaling
        #self.weights = np.linspace(1, 0.01, select_pop)
        #constant weights
        #self.weights = np.ones(select_pop)
        self.weights = self.weights / np.sum(self.weights)  # Normalize weights


        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)

        # Strategy parameter setting: Adaptation
        self.cc = (4 + self.mueff / self.N_dim) / (self.N_dim + 4 + 2 * self.mueff / self.N_dim)
        self.cs = (self.mueff + 2) / (self.N_dim + self.mueff + 5)
        self.c1 = 2 / ((self.N_dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.N_dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.N_dim + 1)) - 1) + self.cs

        # Initialize evolution paths and covariance matrix
        self.pc = np.zeros((N_dim, 1))
        self.ps = np.zeros((N_dim, 1))
        self.B = np.eye(N_dim)
        self.D = np.ones([N_dim, 1])
        self.C = self.B @ np.diag((self.D**2).flatten()) @ self.B.T
        self.invsqrtC = self.B @ np.diag((self.D**-1).flatten()) @ self.B.T
        self.eigeneval = 0
        self.chiN = N_dim**0.5 * (1 - 1/(4*N_dim) + 1/(21*N_dim**2))

        self.epsilon = 1e-8  # A small value to ensure numerical stability
        self.sigma_max = 1e10  # Upper bound for sigma
        self.sigma_min = 1e-10  # Lower bound for sigma
        self.eigen_update_frequency = int(N_dim / 10)
        #self.eigen_update_frequency = 1
        
        self.population = None
        self.counteval = 0
        self.best_solution = None
        self.best_fitness = None
        
    def ask(self):
        # Generate a new population
        self.population = self.xmean + self.sigma * (self.B @ (self.D * np.random.randn(self.N_dim, self.pop_size)))
        
        # Validation check for population generation
        if self.population.shape != (self.N_dim, self.pop_size):
            raise ValueError("Generated population size does not match the specified dimensions")

        return np.transpose(self.population)

    def tell(self, reward_table):
        # Validation check for reward_table
        if not isinstance(reward_table, (list, np.ndarray)) or len(reward_table) != self.pop_size:
            raise ValueError("reward_table must be a list or numpy array with length equal to the population size")

        arfitness = np.array(reward_table)
        arindex = np.argsort(arfitness,axis=0)
        xold = self.xmean.copy()

        # Update mean
        self.xmean = np.dot(self.population[:, arindex[:self.select_pop]].squeeze(), self.weights[:, np.newaxis])

        # Update evolution paths
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.invsqrtC @ (self.xmean - xold) / self.sigma

        hsig = np.sum(self.ps**2) / (1 - (1 - self.cs)**(2 * self.counteval / self.pop_size) + self.epsilon) / self.N_dim < 2 + 4 / (self.N_dim + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.xmean - xold) / self.sigma

        # Adapt covariance matrix C with numerical stability enhancement
        artmp = (1 / self.sigma) * (self.population[:, arindex[:self.select_pop]].squeeze() - np.tile(xold, (1, self.select_pop)))
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * artmp @ np.diag(self.weights) @ artmp.T
        np.fill_diagonal(self.C, self.C.diagonal() + self.epsilon)

        # Adapt step size sigma with boundary checks
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = min(max(self.sigma, self.sigma_min), self.sigma_max)

        # Update eigen decomposition of the covariance matrix C less frequently
        if self.counteval % self.eigen_update_frequency == 0:
            self.C = (self.C + self.C.T) / 2  # Ensure C is symmetric
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D[:,np.newaxis])
            #self.D = np.sqrt(np.diag(self.D))
            self.invsqrtC = self.B @ np.diag((self.D**-1).flatten()) @ self.B.T

        # Log the best individual solution and its fitness
        current_best_index = np.argmin(reward_table)
        current_best_solution = self.population[:, current_best_index]
        current_best_fitness = reward_table[current_best_index]

        if self.best_fitness is None or current_best_fitness < self.best_fitness:
            self.best_solution = current_best_solution.copy()
            self.best_fitness = current_best_fitness

        self.counteval += 1    
    
    
    
    
    
"""
Based mostly on the oto code from David Ha:

https://blog.otoro.net/2017/10/29/visual-evolution-strategies/
"""

class PEPG_opt:
    
    def __init__(self, num_params ,pop_size, learning_rate, starting_mu, starting_sigma):
        
        self.pop_size = pop_size + 1 if pop_size % 2 else pop_size
        self.num_params = num_params
        self.mu = np.array(starting_mu).flatten()
        self.batch_size = self.pop_size // 2
        self.batch_reward = np.zeros(2 * self.batch_size)
        self.sigma = np.ones(num_params) * starting_sigma
        self.sigma_init = starting_sigma
        
        # Parameters for the PEPG optimizer
        self.sigma_alpha = 0.30
        self.sigma_decay = 0.999
        self.sigma_limit = 0.01
        self.sigma_max_change = 0.2
        self.learning_rate = learning_rate
        self.learning_rate_decay = 0.99
        self.learning_rate_limit = 0.01
        self.elite_ratio = 0
        self.weight_decay = 0.01
        self.forget_best = True
        self.rank_fitness = True
        self.average_baseline = True
        self.use_elite = False

        self.first_iteration = True
        self.best_reward = None
        self.best_mu = None

    def compute_ranks(self, x):
        #ranks = np.empty_like(x)
        #ranks[x.argsort(axis=0)] = np.arange(len(x))
        #return ranks
        argsorted_x = np.argsort(x,axis=0)
        ranks = np.empty_like(argsorted_x, dtype=float)
        ranks[argsorted_x] = np.arange(len(x))
        return ranks

    def compute_centered_ranks(self, x):
        y = self.compute_ranks(x)
        y = y / (len(x) - 1)
        return y - 0.5

    def ask(self):
        self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma
        self.epsilon_full = np.vstack((self.epsilon, -self.epsilon))
        if self.average_baseline:
            return self.mu + self.epsilon_full
        else:
            return np.vstack((np.zeros(self.num_params), self.mu + self.epsilon_full))

    def tell(self, reward_table_result):
        reward_table = np.array(reward_table_result).flatten()
        if self.rank_fitness:
            reward_table = self.compute_centered_ranks(reward_table)

        if self.average_baseline:
            b = np.mean(reward_table)
        else:
            b = reward_table[0]

        reward = reward_table[0:] if self.average_baseline else reward_table[1:]

        #best_reward_index = np.argmin(reward) if self.rank_fitness else np.argmax(reward)
        #best_reward = reward[best_reward_index]
        best_reward_index = np.argmin(reward)
        best_reward = reward[best_reward_index]

        if best_reward > b or self.average_baseline:
            best_mu = self.mu + self.epsilon_full[best_reward_index]
        else:
            best_mu = self.mu

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_iteration:
            self.sigma = np.ones(self.num_params) * self.sigma_init
            self.first_iteration = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        elif self.forget_best or self.curr_best_reward > self.best_reward:
            self.best_mu = best_mu
            self.best_reward = self.curr_best_reward

        # Update mu
        if self.use_elite:
            elite_indices = np.argsort(reward_table_result)[:int(self.elite_ratio * len(reward_table_result))]
            self.mu += np.mean(self.epsilon_full[elite_indices], axis=0)
        else:
            if self.rank_fitness:
                reward_table = self.compute_centered_ranks(np.array(reward_table_result).flatten())
            rT = reward_table[:self.batch_size] - reward_table[self.batch_size:]
            change_mu = np.dot(rT, self.epsilon_full[:self.batch_size])
            self.mu -= self.learning_rate * change_mu
    
        # Update sigma
        if self.sigma_alpha > 0:
            stdev_reward = 1.0
            if not self.rank_fitness:
                stdev_reward = np.std(reward_table_result)
            S = (self.epsilon ** 2 - self.sigma ** 2) / self.sigma
            reward_avg = (reward_table[:self.batch_size] + reward_table[self.batch_size:]) / 2.0
            rS = reward_avg - b
            delta_sigma = np.dot(rS.T, S) / (2 * self.batch_size * stdev_reward)
            change_sigma = self.sigma_alpha * delta_sigma
            change_sigma = np.clip(change_sigma, -self.sigma_max_change * self.sigma, self.sigma_max_change * self.sigma)
            self.sigma -= change_sigma.squeeze()
    
        # Apply sigma decay
        if self.sigma_decay < 1:
            self.sigma *= self.sigma_decay
            self.sigma = np.maximum(self.sigma, self.sigma_limit)
    
        # Apply learning rate decay
        if self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit:
            self.learning_rate *= self.learning_rate_decay
    
        return 0 


# Example usage:
# pe = PEPG_Optimization(pop_size, num_params, learning_rate, starting_mu, starting_sigma)
# solutions = pe.ask()
# pe.tell(reward_table_result)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    