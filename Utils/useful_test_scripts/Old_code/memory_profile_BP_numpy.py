import numpy as np
import sys
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')

import tracemalloc
import psutil, os   
import scipy
import time
from optimization_algorithms import *
import gc

import sys, psutil, os, gc, tracemalloc, time


import numpy as np

import numpy as np

class SimpleFFNN_Numpy:
    """
    NumPy-only MLP with full forward/backward caches.
    Hidden: tanh, Output: linear logits.
    """

    def __init__(self, params, seed=None, dtype=np.float32):
        self.N_in         = int(params["N_in"])
        self.N_out        = int(params["N_out"])
        self.N_neurons    = int(params["N_neurons"])
        self.N_hidden_layers = int(params["N_layers"])  # number of hidden layers

        self.rng   = np.random.default_rng(seed)
        self.dtype = dtype

        # Build layer shapes
        shapes = [(self.N_in, self.N_neurons)]
        for _ in range(self.N_hidden_layers - 1):
            shapes.append((self.N_neurons, self.N_neurons))
        shapes.append((self.N_neurons, self.N_out))

        # Params: Glorot uniform
        self.W = []
        self.b = []
        for fan_in, fan_out in shapes:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W.append(self.rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(self.dtype))
            self.b.append(np.zeros((fan_out,), dtype=self.dtype))

        # Grad buffers
        self.dW = [np.zeros_like(W) for W in self.W]
        self.db = [np.zeros_like(b) for b in self.b]

        # Caches for memory tracking
        self.fw = None  # forward cache
        self.bw = None  # backward intermediates

    # -------- activations --------
    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _tanh_prime(y):
        # y = tanh(z) -> d/dz tanh(z) = 1 - y^2
        return 1.0 - y * y

    # -------- utils --------
    def zero_grad(self):
        for g in self.dW: g.fill(0)
        for g in self.db: g.fill(0)

    def clear_caches(self):
        """Free forward/backward caches to release memory."""
        self.fw = None
        self.bw = None

    def _nbytes_list(self, xs):
        total = 0
        for x in xs:
            if isinstance(x, np.ndarray):
                total += x.nbytes
            elif isinstance(x, (list, tuple)):
                total += self._nbytes_list(x)
        return total

    def memory_snapshot(self, verbose=True):
        """
        Return a dict summarizing memory use (bytes and MB) for:
        - params
        - grads
        - forward_cache
        - backward_cache
        - total_model_plus_caches
        If verbose=True, also prints a nicely formatted summary in MB.
        """
        def _nbytes_list(xs):
            total = 0
            for x in xs:
                if isinstance(x, np.ndarray):
                    total += x.nbytes
                elif isinstance(x, (list, tuple)):
                    total += _nbytes_list(x)
            return total

        param_bytes = _nbytes_list(self.W) + _nbytes_list(self.b)
        grad_bytes  = _nbytes_list(self.dW) + _nbytes_list(self.db)

        fw_bytes = 0
        if self.fw is not None:
            fw_bytes += _nbytes_list(self.fw.get("activations", []))
            fw_bytes += _nbytes_list(self.fw.get("preacts", []))

        bw_bytes = 0
        if self.bw is not None:
            for k in ("dA", "dZ"):
                bw_bytes += _nbytes_list(self.bw.get(k, []))
            for k in ("dW_steps", "db_steps"):
                bw_bytes += _nbytes_list(self.bw.get(k, []))
            if isinstance(self.bw.get("dX"), np.ndarray):
                bw_bytes += self.bw["dX"].nbytes

        total_bytes = param_bytes + grad_bytes + fw_bytes + bw_bytes
        total_MB = total_bytes / (1024 ** 2)

        info = {
            "params_MB": round(param_bytes / (1024 ** 2), 3),
            "grads_MB": round(grad_bytes / (1024 ** 2), 3),
            "forward_cache_MB": round(fw_bytes / (1024 ** 2), 3),
            "backward_cache_MB": round(bw_bytes / (1024 ** 2), 3),
            "total_MB": round(total_MB, 3),
        }

        if verbose:
            print(f"Memory usage summary:")
            print(f"  Params:           {info['params_MB']} MB")
            print(f"  Grads:            {info['grads_MB']} MB")
            print(f"  Forward cache:    {info['forward_cache_MB']} MB")
            print(f"  Backward cache:   {info['backward_cache_MB']} MB")
            print(f"  -------------------------------")
            print(f"  TOTAL:            {info['total_MB']} MB")

        return info


    # -------- forward --------
    def forward(self, X, return_cache=True):
        """
        X: (B, ...) flattened to (B, N_in)
        Returns logits (B, N_out) and stores forward cache if requested.
        """
        X = np.asarray(X, dtype=self.dtype)
        B = X.shape[0]
        x = X.reshape(B, -1)
        assert x.shape[1] == self.N_in, f"Expected input dim {self.N_in}, got {x.shape[1]}"

        activations = [x]   # a^0 = input
        preacts = []        # z^1..z^{Lh}, z^{out}
        Lh = self.N_hidden_layers

        # hidden stack
        for i in range(Lh):
            z = activations[-1] @ self.W[i] + self.b[i]
            a = self._tanh(z)
            preacts.append(z)
            activations.append(a)

        # output (linear logits)
        z_out = activations[-1] @ self.W[-1] + self.b[-1]
        preacts.append(z_out)
        activations.append(z_out)  # keep logits in activations for convenience

        if return_cache:
            self.fw = {
                "activations": activations,  # [a^0, a^1, ..., a^{Lh}, logits]
                "preacts": preacts,          # [z^1, ..., z^{Lh}, z^{out}]
                "input_shape": X.shape,
            }
        return z_out

    # -------- backward --------
    def backward(self, dL_dout):
        """
        dL_dout: (B, N_out) = gradient of loss w.r.t. logits.
        Saves all intermediates to self.bw for memory tracking, computes self.dW/db, and returns dL/dX.
        IMPORTANT: This assumes dL_dout is already averaged over the batch if your loss is a mean.
        """
        assert self.fw is not None, "Call forward(..., return_cache=True) before backward()."
        A = self.fw["activations"]         # a^0..a^{Lh}, logits
        Lh = self.N_hidden_layers
        L_total = len(self.W)              # hidden affines + output
        B = dL_dout.shape[0]

        # Prepare backward cache containers
        dA_list = [None] * (Lh + 1)  # dA for a^1..a^{Lh}, and for input we'll return dX
        dZ_list = [None] * (Lh + 1)  # dZ for each affine, including output "layer" as last
        dW_steps = [None] * L_total  # per-layer param grads (snapshots)
        db_steps = [None] * L_total

        # --- Output layer ---
        grad = dL_dout.astype(self.dtype, copy=False)       # dL/d(logits)
        dZ_list[-1] = grad                                  # z_out gradient
        # Parameter grads (NOTE: no extra /B here; assume grad already averaged if needed)
        self.dW[-1][...] = A[Lh].T @ grad                   # A[Lh] is last hidden activation
        self.db[-1][...] = grad.sum(axis=0)                 # sum if loss already averaged; matches grad semantics
        dW_steps[-1] = self.dW[-1].copy()
        db_steps[-1] = self.db[-1].copy()

        # Propagate to previous activation
        grad = grad @ self.W[-1].T                          # dL/da^{Lh}
        dA_list[-1] = grad.copy()

        # --- Hidden layers (reverse) ---
        for i in reversed(range(Lh)):
            # a^{i+1} = tanh(z^{i+1})
            tanh_p = self._tanh_prime(A[i+1])               # using activation output
            dz = grad * tanh_p                               # dL/dz^{i+1}
            dZ_list[i] = dz.copy()

            # Param grads at layer i
            self.dW[i][...] = A[i].T @ dz
            self.db[i][...] = dz.sum(axis=0)
            dW_steps[i] = self.dW[i].copy()
            db_steps[i] = self.db[i].copy()

            # Propagate to previous activation
            if i > 0:
                grad = dz @ self.W[i].T                      # dL/da^{i}
                dA_list[i] = grad.copy()
            else:
                dX = dz @ self.W[i].T                        # gradient wrt input (a^0)
                # No previous activation to store beyond input

        # Save backward intermediates
        self.bw = {
            "dA": dA_list,           # list of arrays (may have None at some entries)
            "dZ": dZ_list,
            "dW_steps": dW_steps,
            "db_steps": db_steps,
            "dX": dX.astype(self.dtype, copy=False),
        }

        return self.bw["dX"]

    # -------- optimizer step --------
    def step(self, lr=1e-2, weight_decay=0.0, average_grads_over_batch=True, batch_size=None):
        """
        SGD update with optional decoupled weight decay.
        If your dL_dout was NOT averaged, you can set average_grads_over_batch=True with batch_size=B to average once here.
        """
        if average_grads_over_batch and (batch_size is not None and batch_size > 0):
            scale = 1.0 / batch_size
        else:
            scale = 1.0

        for i in range(len(self.W)):
            if weight_decay != 0.0:
                self.W[i] -= lr * (scale * self.dW[i] + weight_decay * self.W[i])
                self.b[i] -= lr * (scale * self.db[i])  # usually no decay on bias
            else:
                self.W[i] -= lr * (scale * self.dW[i])
                self.b[i] -= lr * (scale * self.db[i])

    # -------- helpers --------
    def predict(self, X):
        logits = self.forward(X, return_cache=False)
        return logits.argmax(axis=1)

    def get_params(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def set_params(self, W_list, B_list):
        assert len(W_list) == len(self.W) and len(B_list) == len(self.b)
        for i, (Wn, Bn) in enumerate(zip(W_list, B_list)):
            assert Wn.shape == self.W[i].shape and Bn.shape == self.b[i].shape
            self.W[i] = Wn.astype(self.dtype, copy=True)
            self.b[i] = Bn.astype(self.dtype, copy=True)


# ---- Stable softmax-CE returning mean loss and dL/dlogits (already averaged) ----
def softmax_xent_with_logits(logits, y_true):
    logits = np.asarray(logits)
    B, C = logits.shape
    if y_true.ndim == 1:
        y = np.zeros((B, C), dtype=logits.dtype)
        y[np.arange(B), y_true] = 1.0
    else:
        y = y_true.astype(logits.dtype)

    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    p = exp_z / exp_z.sum(axis=1, keepdims=True)

    eps = 1e-12
    loss = -np.sum(y * np.log(p + eps)) / B
    dlogits = (p - y) / B  # averaged
    return loss, dlogits


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

# Toy data
B = 1

params = {"N_in": 28*28,
    "N_out": 10,
    "N_neurons": 50,
    "N_layers": 3}


model = SimpleFFNN_Numpy(params, seed=0)

X = np.random.randn(B, 28, 28).astype(np.float32)
y = np.random.randint(0, 10, size=(B,))

# Forward
logits = model.forward(X)                # stores cache
loss, dL_dlogits = softmax_xent_with_logits(logits, y)




tracemalloc.start()
rss_before = rss_mb()

# Backward & update
model.zero_grad()
model.backward(dL_dlogits)
model.step(lr=1e-2, average_grads_over_batch=False)
print(model.memory_snapshot())  # bytes & MiB per section


# force a GC cycle if we want a “post-op” snapshot
gc.collect()

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

rss_after = rss_mb()
rss_delta = rss_after - rss_before

print(f"\ntracemalloc current Python-heap: {current/1e6:.3f} MB")
print(f"tracemalloc peak   Python-heap: {peak/1e6:.3f} MB")
print(f"Process RSS delta (real):       {rss_delta:.3f} MB")
print(f"Process RSS total (now):        {rss_after:.3f} MB")






