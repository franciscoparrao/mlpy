"""
Direct test of TGPY to understand its API.
"""

import numpy as np
import torch
import tgpy
from tgpy.random import TGP, CovarianceTransport
from tgpy.kernel import SE
from tgpy.modules import Constant
from tgpy.learning import TgLearning

# Set device
device = torch.device('cpu')
tgpy.tensor._device = device

# Create simple data
np.random.seed(42)
n = 20
X = torch.tensor(np.random.uniform(-3, 3, (n, 1)), dtype=torch.float32, device=device)
y = torch.sin(X).squeeze() + 0.1 * torch.randn(n, device=device)

print("Data shapes:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")

# Create kernel parameters
relevance = Constant(torch.tensor(1.0, device=device))
var = Constant(torch.tensor(1.0, device=device))

# Create kernel
kernel = SE(
    var=var,
    relevance=relevance
)

# Create noise
noise = Constant(torch.tensor(0.1, device=device))

# Create transport
transport = CovarianceTransport(kernel, noise=noise)

# Create TGP
try:
    # Try different initialization approaches
    print("\nTrying to create TGP...")
    
    # Approach 1: Direct initialization
    tgp = TGP(transport=transport)
    print("TGP created successfully!")
    
    # Set observations
    tgp.obs_x = X
    tgp.obs_y = y
    
    # Create learning object
    learning = TgLearning(tgp, lr=0.01)
    print("Learning object created!")
    
    # Try to train
    print("\nAttempting to train...")
    learning.execute_svgd(niters=10, mcmc=False)
    print("Training successful!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()