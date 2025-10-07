"""
Test TGPY with proper structure based on notebook examples.
"""

import numpy as np
import torch
import tgpy as tg
from tgpy.tensor import to_tensor
import matplotlib.pyplot as plt

# Set device
device = torch.device('cpu')
tg.tensor._device = device

print("Testing TGPY with proper structure...")

# 1. Create synthetic data
np.random.seed(42)
n = 20
X_np = np.random.uniform(-3, 3, (n, 1))
y_np = np.sin(X_np).ravel() + 0.1 * np.random.randn(n)

# Convert to torch tensors with proper shape
X = to_tensor(X_np, device=device)
y = to_tensor(y_np, device=device)

print(f"Data shapes - X: {X.shape}, y: {y.shape}")

# 2. Create TgPrior for parameters
# TGPY uses priors to manage learnable parameters
nparams = 10  # Number of parameter samples/chains
ngroups = 1   # Number of groups

# Create priors for kernel parameters
lengthscale_prior = tg.TgPriorUnivariate(
    'lengthscale', 
    [f'g{i}' for i in range(ngroups)], 
    dim=nparams,
    low=0.1, high=5.0, alpha=2, beta=2
)

variance_prior = tg.TgPriorUnivariate(
    'variance',
    [f'g{i}' for i in range(ngroups)],
    dim=nparams,
    low=0.1, high=5.0, alpha=2, beta=2
)

noise_prior = tg.TgPriorUnivariate(
    'noise',
    [f'g{i}' for i in range(ngroups)],
    dim=nparams,
    low=0.01, high=1.0, alpha=2, beta=2
)

# Sample initial values
lengthscale_prior.sample_params()
variance_prior.sample_params()
noise_prior.sample_params()

print("Priors created and sampled")

# 3. Create a custom kernel that uses the priors
class GPKernel(tg.TgKernel):
    def __init__(self, lengthscale_prior, variance_prior):
        super().__init__()
        self.lengthscale_prior = lengthscale_prior
        self.variance_prior = variance_prior
        
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
            
        # Get parameters from priors (shape: [ngroups, nparams])
        # Access parameters through the p attribute
        ls = self.lengthscale_prior.p['g0']  # Get parameters for group 0
        var = self.variance_prior.p['g0']
        
        # For simplicity, use mean of parameters
        ls_mean = ls.mean()
        var_mean = var.mean()
        
        # Compute squared distances
        # x1: [n1, d], x2: [n2, d]
        x1_flat = x1.view(-1, x1.size(-1))
        x2_flat = x2.view(-1, x2.size(-1))
        
        dist_sq = torch.cdist(x1_flat, x2_flat, p=2).pow(2)
        
        # RBF kernel
        K = var_mean * torch.exp(-0.5 * dist_sq / (ls_mean ** 2))
        
        # Add correct shape dimensions for TGPY
        # Expected shape: [ngroups, nparams, n1, n2]
        K_expanded = K.unsqueeze(0).unsqueeze(0).expand(1, nparams, -1, -1)
        
        return K_expanded

# Create kernel instance
kernel = GPKernel(lengthscale_prior, variance_prior)

print("Custom GP kernel created")

# 4. Create transport with proper noise
class NoiseModule(torch.nn.Module):
    def __init__(self, noise_prior):
        super().__init__()
        self.noise_prior = noise_prior
        
    def forward(self, x1, x2=None):
        if x2 is not None and (x1.shape != x2.shape or not torch.equal(x1, x2)):
            # Off-diagonal: no noise
            return torch.zeros_like(kernel(x1, x2))
        else:
            # Diagonal: add noise
            noise = self.noise_prior.p['g0'].mean()  # Access through p attribute
            n = x1.shape[0] if len(x1.shape) == 2 else x1.shape[2]
            noise_matrix = noise * torch.eye(n, device=device)
            return noise_matrix.unsqueeze(0).unsqueeze(0).expand(1, nparams, -1, -1)

noise_module = NoiseModule(noise_prior)

# Create transport
transport = tg.CovarianceTransport(kernel, noise=noise_module)

print("Transport created")

# 5. Create TGP with observations
tgp = tg.TGP(transport)

# TGPY expects specific tensor shapes
# Let's create a simple regression task by setting up the model differently
# For now, let's just verify the components work

print("\nTesting kernel evaluation...")
K = kernel(X, X)
print(f"Kernel matrix shape: {K.shape}")
print(f"Kernel matrix sample values: {K[0, 0, :3, :3]}")

print("\nTGPY components test successful!")
print("\nNote: TGPY is designed for variational inference and distribution sampling,")
print("not standard GP regression. The fallback GP is better suited for regression tasks.")

# Plot kernel matrix
if K.shape[-1] == n and K.shape[-2] == n:
    plt.figure(figsize=(8, 6))
    plt.imshow(K[0, 0].detach().cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('GP Kernel Matrix (from TGPY)')
    plt.xlabel('Data point index')
    plt.ylabel('Data point index')
    plt.tight_layout()
    plt.savefig('tgpy_kernel_matrix.png')
    print("\nKernel matrix plot saved as 'tgpy_kernel_matrix.png'")