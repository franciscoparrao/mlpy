"""
Simple test for TGPY integration - checking if it works.
"""

import sys
import os

# Add MLPY to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# Test basic MLPY imports
try:
    from mlpy.tasks import TaskRegr
    from mlpy.measures import MeasureRegrRMSE
    print("[OK] MLPY imports successful")
except Exception as e:
    print(f"[FAIL] MLPY import error: {e}")
    sys.exit(1)

# Test TGPY wrapper import
try:
    from mlpy.learners.tgpy_wrapper import LearnerTGPRegressor
    print("[OK] TGPY wrapper import successful")
except Exception as e:
    print(f"[FAIL] TGPY wrapper import error: {e}")

# Create simple test data
print("\nCreating test data...")
np.random.seed(42)
n_samples = 50
X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
y = np.sin(X).ravel() + 0.1 * np.random.randn(n_samples)

# Create dataframe and task
data = pd.DataFrame(X, columns=['x'])
data['y'] = y

task = TaskRegr(
    id="test_tgpy",
    data=data,
    target="y"
)
print(f"[OK] Created task with {n_samples} samples")

# Try to create and use TGPY learner
print("\nTesting TGPY learner...")
try:
    learner = LearnerTGPRegressor(
        kernel='SE',
        lengthscale=1.0,
        variance=1.0,
        noise=0.1,
        n_iterations=10,  # Small number for testing
        learning_rate=0.01
    )
    print("[OK] TGPY learner created")
    
    # Train
    train_ids = list(range(40))
    learner.train(task, row_ids=train_ids)
    print("[OK] Training completed")
    
    # Predict
    test_ids = list(range(40, 50))
    pred = learner.predict(task, row_ids=test_ids)
    print("[OK] Prediction completed")
    
    # Calculate RMSE
    rmse = MeasureRegrRMSE()
    score = rmse.score(pred)
    print(f"[OK] RMSE: {score:.4f}")
    
    # Check if TGPY is actually available
    if hasattr(learner, '_tgpy_available') and learner._tgpy_available:
        print("\n[OK] TGPY is available and working!")
    else:
        print("\n[WARNING] TGPY not available - using fallback implementation")
        
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Test complete!")