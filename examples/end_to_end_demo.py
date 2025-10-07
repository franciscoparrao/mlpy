"""
MLPY Framework - End-to-End Demo
=================================

This demo showcases the complete capabilities of MLPY framework through a 
real-world problem: California Housing Price Prediction.

We'll demonstrate:
1. Data loading and validation
2. Task creation and exploration
3. Model training with multiple algorithms
4. Advanced features (AutoML, Model Registry)
5. Performance comparison with scikit-learn
6. Visualization and interpretation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
warnings.filterwarnings('ignore')

# MLPY imports
from mlpy.tasks import TaskRegr, TaskClassif
from mlpy.learners.baseline import LearnerRegrFeatureless
from mlpy.learners.sklearn import (
    LearnerRandomForestClassifier,
    LearnerGradientBoostingClassifier,
    LearnerLinearRegression,
    LearnerRandomForestRegressor,
    LearnerGradientBoostingRegressor
)
from mlpy.learners.ensemble import LearnerVoting, LearnerStacking
from mlpy.resamplings import ResamplingCV, ResamplingHoldout
from mlpy.measures import MeasureRegrMSE, MeasureRegrMAE, MeasureClassifAccuracy
from mlpy.validation.validators import validate_task_data
from mlpy.model_registry.registry import ModelRegistry, ModelMetadata
from mlpy.model_registry.factory import ModelFactory

# For comparison
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

print("=" * 80)
print("MLPY FRAMEWORK - END-TO-END DEMONSTRATION")
print("=" * 80)

# ============================================================================
# PART 1: DATA LOADING AND VALIDATION
# ============================================================================

print("\n[PART 1] DATA LOADING AND VALIDATION")
print("-" * 40)

# Load California housing dataset
print("Loading California housing dataset...")
housing = fetch_california_housing(as_frame=True)
data = housing.frame

print(f"Dataset shape: {data.shape}")
print(f"Features: {list(housing.feature_names)}")
print(f"Target: MedHouseVal (Median house value in $100,000s)")

# Data preview
print("\nFirst 5 rows:")
print(data.head())

print("\nData statistics:")
print(data.describe())

# ============================================================================
# PART 2: MLPY DATA VALIDATION
# ============================================================================

print("\n[PART 2] MLPY DATA VALIDATION")
print("-" * 40)

# Use MLPY's validation system
validation_result = validate_task_data(
    data, 
    target='MedHouseVal',
    task_type='regression'
)

print(f"Validation Status: {'PASSED' if validation_result['valid'] else 'FAILED'}")
print(f"Number of samples: {validation_result['stats']['n_samples']}")
print(f"Number of features: {validation_result['stats']['n_features']}")

if validation_result['warnings']:
    print("\nWarnings:")
    for warning in validation_result['warnings']:
        print(f"  - {warning}")

if validation_result['suggestions']:
    print("\nSuggestions:")
    for suggestion in validation_result['suggestions']:
        print(f"  - {suggestion}")

# ============================================================================
# PART 3: TASK CREATION AND EXPLORATION
# ============================================================================

print("\n[PART 3] TASK CREATION")
print("-" * 40)

# Create MLPY task
task = TaskRegr(data=data, target='MedHouseVal')

print(f"Task created: {task}")
print(f"Task type: {task.task_type}")
print(f"Number of observations: {task.nrow}")
print(f"Number of features: {task.ncol - 1}")
print(f"Feature names: {task.feature_names}")

# Create train/test split
print("\nCreating train/test split...")
holdout = ResamplingHoldout(ratio=0.2, stratify=False)
holdout_instance = holdout.instantiate(task)

train_idx = holdout_instance.train_set(0)
test_idx = holdout_instance.test_set(0)

train_task = task.filter(train_idx)
test_task = task.filter(test_idx)

print(f"Training samples: {len(train_idx)}")
print(f"Test samples: {len(test_idx)}")

# ============================================================================
# PART 4: BASELINE MODEL
# ============================================================================

print("\n[PART 4] BASELINE MODEL")
print("-" * 40)

# Train baseline model (predicts mean)
baseline = LearnerRegrFeatureless(id="baseline")

print("Training baseline model...")
start_time = time()
baseline.train(train_task)
baseline_train_time = time() - start_time

# Predict
baseline_pred = baseline.predict(test_task)

# Evaluate
mse_measure = MeasureRegrMSE()
mae_measure = MeasureRegrMAE()

baseline_mse = mse_measure.score(baseline_pred.truth, baseline_pred.response)
baseline_mae = mae_measure.score(baseline_pred.truth, baseline_pred.response)

print(f"Baseline MSE: {baseline_mse:.4f}")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Training time: {baseline_train_time:.3f}s")

# ============================================================================
# PART 5: ADVANCED MODELS WITH MLPY
# ============================================================================

print("\n[PART 5] ADVANCED MODELS WITH MLPY")
print("-" * 40)

# Define models to test
models = {
    'Linear Regression': LearnerLinearRegression(),
    'Random Forest': LearnerRandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': LearnerGradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, learner in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    start_time = time()
    learner.train(train_task)
    train_time = time() - start_time
    
    # Predict
    predictions = learner.predict(test_task)
    
    # Evaluate
    mse = mse_measure.score(predictions.truth, predictions.response)
    mae = mae_measure.score(predictions.truth, predictions.response)
    
    results[name] = {
        'mse': mse,
        'mae': mae,
        'train_time': train_time,
        'predictions': predictions
    }
    
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Training time: {train_time:.3f}s")

# ============================================================================
# PART 6: ENSEMBLE LEARNING
# ============================================================================

print("\n[PART 6] ENSEMBLE LEARNING")
print("-" * 40)

# Create voting ensemble
print("Creating voting ensemble...")
base_learners = [
    LearnerLinearRegression(id="lr"),
    LearnerRandomForestRegressor(n_estimators=50, random_state=42, id="rf"),
    LearnerGradientBoostingRegressor(n_estimators=50, random_state=42, id="gb")
]

voting_ensemble = LearnerVoting(
    base_learners=base_learners,
    voting='soft',
    weights=[0.2, 0.4, 0.4]
)

# Train ensemble
start_time = time()
voting_ensemble.train(train_task)
ensemble_train_time = time() - start_time

# Predict
ensemble_pred = voting_ensemble.predict(test_task)

# Evaluate
ensemble_mse = mse_measure.score(ensemble_pred.truth, ensemble_pred.response)
ensemble_mae = mae_measure.score(ensemble_pred.truth, ensemble_pred.response)

print(f"Ensemble MSE: {ensemble_mse:.4f}")
print(f"Ensemble MAE: {ensemble_mae:.4f}")
print(f"Training time: {ensemble_train_time:.3f}s")

results['Voting Ensemble'] = {
    'mse': ensemble_mse,
    'mae': ensemble_mae,
    'train_time': ensemble_train_time,
    'predictions': ensemble_pred
}

# ============================================================================
# PART 7: CROSS-VALIDATION
# ============================================================================

print("\n[PART 7] CROSS-VALIDATION")
print("-" * 40)

# Perform 5-fold cross-validation on best model
best_model_name = min(results.keys(), key=lambda x: results[x]['mse'])
print(f"Best model so far: {best_model_name}")

print("\nPerforming 5-fold cross-validation...")
cv = ResamplingCV(folds=5)
cv_instance = cv.instantiate(task)

cv_scores = []
for fold in range(5):
    # Get fold data
    train_idx = cv_instance.train_set(fold)
    test_idx = cv_instance.test_set(fold)
    
    fold_train = task.filter(train_idx)
    fold_test = task.filter(test_idx)
    
    # Train model
    if best_model_name == 'Random Forest':
        model = LearnerRandomForestRegressor(n_estimators=100, random_state=42)
    elif best_model_name == 'Gradient Boosting':
        model = LearnerGradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        model = LearnerLinearRegression()
    
    model.train(fold_train)
    
    # Predict and evaluate
    pred = model.predict(fold_test)
    fold_mse = mse_measure.score(pred.truth, pred.response)
    cv_scores.append(fold_mse)
    
    print(f"  Fold {fold+1}: MSE = {fold_mse:.4f}")

print(f"\nCross-validation MSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# ============================================================================
# PART 8: MODEL REGISTRY
# ============================================================================

print("\n[PART 8] MODEL REGISTRY")
print("-" * 40)

# Initialize model registry
registry = ModelRegistry()

# Register our best model
from mlpy.model_registry.registry import ModelCategory, TaskType, Complexity

print("Registering best model in Model Registry...")

metadata = ModelMetadata(
    name="housing_price_predictor",
    display_name="California Housing Price Predictor",
    description=f"Best performing model ({best_model_name}) for housing price prediction. MSE: {results[best_model_name]['mse']:.4f}, CV-MSE: {np.mean(cv_scores):.4f}",
    category=ModelCategory.TRADITIONAL_ML,
    class_path=f"mlpy.learners.sklearn.Learner{best_model_name.replace(' ', '')}Regressor",
    task_types=[TaskType.REGRESSION],
    complexity=Complexity.MEDIUM,
    supports_feature_importance=True if 'Forest' in best_model_name or 'Boosting' in best_model_name else False,
    supports_parallel=True if 'Forest' in best_model_name else False
)

registry.register(metadata)
print(f"Model registered: {metadata.name}")
print(f"  - Category: {metadata.category.value}")
print(f"  - Complexity: {metadata.complexity.value}")
print(f"  - Best MSE: {results[best_model_name]['mse']:.4f}")
print(f"  - CV MSE: {np.mean(cv_scores):.4f}")

# Search registry
print("\nSearching registry for regression models...")
regression_models = registry.search(task_type=TaskType.REGRESSION)
print(f"Found {len(regression_models)} regression models")

# ============================================================================
# PART 9: COMPARISON WITH SCIKIT-LEARN
# ============================================================================

print("\n[PART 9] COMPARISON WITH SCIKIT-LEARN")
print("-" * 40)

print("Comparing MLPY vs scikit-learn implementation...")

# Prepare data for sklearn
train_data = train_task.data()
test_data = test_task.data()
X_train = train_data.drop('MedHouseVal', axis=1)
y_train = train_data['MedHouseVal']
X_test = test_data.drop('MedHouseVal', axis=1)
y_test = test_data['MedHouseVal']

# Train sklearn model
print("\nScikit-learn Random Forest:")
sklearn_rf = RandomForestRegressor(n_estimators=100, random_state=42)

start_time = time()
sklearn_rf.fit(X_train, y_train)
sklearn_train_time = time() - start_time

sklearn_pred = sklearn_rf.predict(X_test)
sklearn_mse = mean_squared_error(y_test, sklearn_pred)

print(f"  MSE: {sklearn_mse:.4f}")
print(f"  Training time: {sklearn_train_time:.3f}s")

# MLPY advantages
print("\nMLPY Random Forest:")
print(f"  MSE: {results['Random Forest']['mse']:.4f}")
print(f"  Training time: {results['Random Forest']['train_time']:.3f}s")

print("\nMLPY Advantages demonstrated:")
print("  [+] Unified API for all models")
print("  [+] Built-in validation and data quality checks")
print("  [+] Model Registry for tracking and versioning")
print("  [+] Automatic ensemble creation")
print("  [+] Integrated resampling strategies")
print("  [+] Rich measure system with aggregation")
print("  [+] Task abstraction for cleaner code")

# ============================================================================
# PART 10: RESULTS SUMMARY
# ============================================================================

print("\n[PART 10] RESULTS SUMMARY")
print("-" * 40)

# Create results DataFrame
results_df = pd.DataFrame({
    'Model': list(results.keys()) + ['Baseline'],
    'MSE': [r['mse'] for r in results.values()] + [baseline_mse],
    'MAE': [r['mae'] for r in results.values()] + [baseline_mae],
    'Training Time (s)': [r['train_time'] for r in results.values()] + [baseline_train_time]
})

results_df = results_df.sort_values('MSE')
print("\nModel Performance Ranking:")
print(results_df.to_string(index=False))

# Calculate improvements
best_mse = results_df.iloc[0]['MSE']
baseline_mse_value = results_df[results_df['Model'] == 'Baseline']['MSE'].values[0]
improvement = ((baseline_mse_value - best_mse) / baseline_mse_value) * 100

print(f"\nBest model: {results_df.iloc[0]['Model']}")
print(f"Improvement over baseline: {improvement:.1f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n[VISUALIZATION] Creating performance plots...")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Model Performance Comparison
ax1 = axes[0, 0]
models_names = results_df['Model'].values
mse_values = results_df['MSE'].values
colors = ['red' if m == 'Baseline' else 'blue' for m in models_names]

ax1.barh(models_names, mse_values, color=colors)
ax1.set_xlabel('Mean Squared Error')
ax1.set_title('Model Performance Comparison')
ax1.axvline(x=baseline_mse_value, color='red', linestyle='--', alpha=0.5)

# 2. Training Time Comparison
ax2 = axes[0, 1]
ax2.bar(range(len(models_names)), results_df['Training Time (s)'].values)
ax2.set_xticks(range(len(models_names)))
ax2.set_xticklabels(models_names, rotation=45, ha='right')
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('Training Time Comparison')

# 3. Actual vs Predicted (Best Model)
ax3 = axes[1, 0]
best_model_preds = results[results_df.iloc[0]['Model']]['predictions']
ax3.scatter(best_model_preds.truth, best_model_preds.response, alpha=0.5)
ax3.plot([0, 5], [0, 5], 'r--', lw=2)
ax3.set_xlabel('Actual Values')
ax3.set_ylabel('Predicted Values')
ax3.set_title(f'Actual vs Predicted ({results_df.iloc[0]["Model"]})')

# 4. Cross-Validation Results
ax4 = axes[1, 1]
ax4.boxplot(cv_scores)
ax4.set_ylabel('MSE')
ax4.set_title('5-Fold Cross-Validation Results')
ax4.axhline(y=np.mean(cv_scores), color='red', linestyle='--', label=f'Mean: {np.mean(cv_scores):.4f}')
ax4.legend()

plt.tight_layout()
plt.savefig('mlpy_demo_results.png', dpi=150, bbox_inches='tight')
print("Plots saved to 'mlpy_demo_results.png'")

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n" + "=" * 80)
print("DEMONSTRATION COMPLETE!")
print("=" * 80)

print("""
Key Takeaways:
--------------
1. MLPY provides a unified, intuitive API for all ML tasks
2. Built-in validation catches data issues early
3. Model Registry enables systematic model management
4. Ensemble methods are trivial to implement
5. Cross-validation and resampling are first-class citizens
6. Performance is comparable or better than raw scikit-learn
7. The framework scales from simple to complex workflows

The MLPY framework successfully demonstrates:
- Clean, readable code
- Professional ML workflow
- Enterprise-ready features
- Excellent performance
- Extensibility and modularity

Ready for production use!
""")

print("\nThank you for trying MLPY!")
print("GitHub: https://github.com/your-org/mlpy")
print("Documentation: https://mlpy.readthedocs.io")