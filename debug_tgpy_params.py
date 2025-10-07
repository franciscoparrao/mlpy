"""
Debug how to access parameters in TGPY priors.
"""

import torch
import tgpy as tg

device = torch.device('cpu')
tg.tensor._device = device

# Create a simple prior
prior = tg.TgPriorUnivariate(
    'test', 
    ['g0'], 
    dim=3,
    low=0.1, high=3.0, alpha=2, beta=2
)

print("TgPriorUnivariate structure:")
print(f"Type: {type(prior)}")
print(f"Attributes: {dir(prior)}")
print()

print("Checking parameters method:")
print(f"Has parameters method: {hasattr(prior, 'parameters')}")
if hasattr(prior, 'parameters'):
    print(f"parameters type: {type(prior.parameters)}")
    if callable(prior.parameters):
        print("parameters() is callable")
        try:
            params = prior.parameters()
            print(f"parameters() returns: {type(params)}")
            print(f"parameters() content: {list(params)}")
        except Exception as e:
            print(f"Error calling parameters(): {e}")
    else:
        print("parameters is not callable")
        print(f"parameters content: {prior.parameters}")

print()
print("Checking p attribute:")
print(f"p type: {type(prior.p)}")
print(f"p keys: {list(prior.p.keys())}")
print(f"p['g0'] type: {type(prior.p['g0'])}")
print(f"p['g0'] shape: {prior.p['g0'].shape}")
print(f"p['g0'] requires_grad: {prior.p['g0'].requires_grad}")

print()
print("Testing torch optimizer with p directly:")
try:
    optimizer = torch.optim.Adam([prior.p['g0']], lr=0.01)
    print("SUCCESS: Can create optimizer with p['g0']")
except Exception as e:
    print(f"FAILED: {e}")
    
print()
print("Testing torch optimizer with prior as module:")
try:
    optimizer = torch.optim.Adam(prior.parameters(), lr=0.01)
    print("SUCCESS: Can create optimizer with prior.parameters()")
except Exception as e:
    print(f"FAILED: {e}")