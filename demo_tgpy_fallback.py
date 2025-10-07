"""
Demo of TGPY with fallback visualization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlpy.tasks import TaskRegr
from mlpy.measures import MeasureRegrRMSE, MeasureRegrMAE
from mlpy.learners.tgpy_wrapper import LearnerTGPRegressor

# Generate 1D regression data
np.random.seed(42)
n_train = 30
n_test = 20

# Training data - sparse
X_train = np.random.uniform(-3, 3, n_train).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + 0.5 * X_train.ravel() + 0.1 * np.random.randn(n_train)

# Test data - dense for smooth plotting
X_test = np.linspace(-3.5, 3.5, 100).reshape(-1, 1)
y_test_true = np.sin(X_test).ravel() + 0.5 * X_test.ravel()

# Combine for MLPY
X_all = np.vstack([X_train, X_test])
y_all = np.hstack([y_train, y_test_true])

# Create dataframe
data = pd.DataFrame(X_all, columns=['x'])
data['y'] = y_all

# Create task
task = TaskRegr(id="demo", data=data, target="y")

# Train and test indices
train_ids = list(range(n_train))
test_ids = list(range(n_train, n_train + len(X_test)))

# Create TGPY learner
print("Creating TGPY learner...")
learner = LearnerTGPRegressor(
    kernel='SE',
    lengthscale=1.0,
    variance=1.0,
    noise=0.1,
    n_iterations=50
)

# Train
print("Training...")
learner.train(task, row_ids=train_ids)

# Predict with uncertainty
learner.predict_type = "se"
pred = learner.predict(task, row_ids=test_ids)

# Extract predictions
y_pred = pred.response
y_std = pred.se if hasattr(pred, 'se') and pred.se is not None else np.zeros_like(y_pred)

# Create visualization
plt.figure(figsize=(12, 8))

# True function
plt.plot(X_test, y_test_true, 'k--', label='True function', alpha=0.8, linewidth=2)

# Training data
plt.scatter(X_train, y_train, c='blue', s=50, alpha=0.7, 
           edgecolors='black', linewidth=1, label='Training data')

# Predictions
plt.plot(X_test, y_pred, 'r-', label='TGPY prediction (fallback)', linewidth=2)

# Uncertainty bands
plt.fill_between(X_test.ravel(), 
                y_pred - 2*y_std,
                y_pred + 2*y_std,
                alpha=0.3, color='red', 
                label='95% confidence interval')

# Styling
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('TGPY Integration with MLPY - Fallback Implementation', fontsize=14)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add text about fallback
if hasattr(learner, '_use_fallback') and learner._use_fallback:
    plt.text(0.02, 0.98, 
            'Note: Using fallback implementation\n(inverse distance weighting)',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Calculate metrics on a subset
eval_ids = list(range(n_train, n_train + 20))
pred_eval = learner.predict(task, row_ids=eval_ids)

rmse = MeasureRegrRMSE()
mae = MeasureRegrMAE()

print(f"\nEvaluation Metrics:")
print(f"RMSE: {rmse.score(pred_eval):.4f}")
print(f"MAE: {mae.score(pred_eval):.4f}")

# Show plot
plt.savefig('tgpy_fallback_demo.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'tgpy_fallback_demo.png'")
plt.show()

print("\n" + "="*50)
print("TGPY Integration Demo Complete!")
print("="*50)

if hasattr(learner, '_tgpy_available') and not learner._tgpy_available:
    print("\nTo use the full TGPY implementation:")
    print("1. Install IPython: pip install ipython")
    print("2. Install TGPY: cd tgpy-master && pip install -e .")
else:
    print("\nTGPY is fully operational!")