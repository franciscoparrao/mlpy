"""
Test to clearly show which GP implementation is being used.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from mlpy.tasks import TaskRegr
from mlpy.learners.tgpy_wrapper import LearnerTGPRegressor

# Create simple data
np.random.seed(42)
X = np.random.uniform(-3, 3, 30).reshape(-1, 1)
y = np.sin(X).ravel() + 0.1 * np.random.randn(30)

data = pd.DataFrame(X, columns=['x'])
data['y'] = y

task = TaskRegr(id="test", data=data, target="y")

# Create learner
print("Creating TGPY learner...")
learner = LearnerTGPRegressor(
    kernel='SE',
    lengthscale=1.0,
    variance=1.0,
    noise=0.1,
    n_iterations=10
)

# Check TGPY availability
print(f"\nTGPY available: {learner._tgpy_available}")

# Train
print("\nTraining...")
learner.train(task)

# Check which implementation was used
print("\nImplementation used:")
if hasattr(learner, '_use_fallback') and learner._use_fallback:
    print(">>> FALLBACK GP (SimpleGP)")
    if learner.fallback_gp is not None:
        print(f"    - Optimized lengthscale: {learner.fallback_gp.lengthscale:.4f}")
        print(f"    - Optimized variance: {learner.fallback_gp.variance:.4f}")
        print(f"    - Optimized noise: {learner.fallback_gp.noise:.4f}")
else:
    print(">>> TGPY OFFICIAL")
    print("    - Using Transport Gaussian Process")

# Make predictions to verify it works
pred = learner.predict(task)
print(f"\nPredictions work: {pred.response is not None}")
print(f"Mean prediction: {np.mean(pred.response):.4f}")
print(f"Std prediction: {np.std(pred.response):.4f}")