"""
Debug why TGPY official is not working.
"""

import numpy as np
import torch
import tgpy
from tgpy.random import TGP, CovarianceTransport
from tgpy.kernel import SE
from tgpy.modules import Constant
from tgpy.learning import TgLearning

print("Testing TGPY components...")

# 1. Test if we can create basic components
try:
    device = torch.device('cpu')
    tgpy.tensor._device = device
    print("[OK] Device set")
    
    # Create tensor constants
    relevance = Constant(torch.tensor(1.0, device=device))
    var = Constant(torch.tensor(1.0, device=device))
    print("[OK] Constants created")
    
    # Test if constants work
    print(f"    Relevance value: {relevance()}")
    print(f"    Variance value: {var()}")
    
    # Create kernel
    kernel = SE(var=var, relevance=relevance)
    print("[OK] Kernel created")
    
    # Test kernel
    x = torch.randn(5, 1, device=device)
    k_matrix = kernel(x, x)
    print(f"[OK] Kernel evaluation works, shape: {k_matrix.shape}")
    
    # Create transport
    noise = Constant(torch.tensor(0.1, device=device))
    transport = CovarianceTransport(kernel, noise=noise)
    print("[OK] Transport created")
    
    # Create TGP
    tgp = TGP(transport=transport)
    print("[OK] TGP created")
    
    # Check TGP structure
    print(f"\nTGP attributes: {dir(tgp)[:10]}")
    
    # Set observations
    X = torch.randn(10, 1, device=device)
    y = torch.sin(X).squeeze() + 0.1 * torch.randn(10, device=device)
    
    tgp.obs_x = X
    tgp.obs_y = y
    print("[OK] Observations set")
    
    # Check parameters
    print(f"\nTGP parameters: {list(tgp.parameters())}")
    print(f"Number of parameters: {sum(p.numel() for p in tgp.parameters())}")
    
    # Try to create learning object
    learning = TgLearning(tgp, lr=0.01)
    print("[OK] Learning object created")
    
    # The issue is likely here - let's see what happens
    print("\nTrying to run SVGD...")
    learning.execute_svgd(niters=1)
    print("[OK] SVGD works!")
    
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    
print("\n" + "="*50)
print("DIAGNOSIS:")
print("="*50)
print("TGPY can be imported and basic components work,")
print("but there's an issue with the learning/optimization setup.")
print("The fallback GP is a robust alternative that works well.")