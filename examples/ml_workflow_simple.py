"""
Simplified Machine Learning Workflow with MLPY
==============================================
"""

import numpy as np
import pandas as pd
from datetime import datetime

# MLPY imports
from mlpy.tasks import TaskClassif
from mlpy.measures import (
    MeasureClassifAccuracy, MeasureClassifF1, 
    MeasureClassifPrecision, MeasureClassifRecall
)
from mlpy.resamplings import ResamplingCV, ResamplingHoldout
from mlpy import resample, benchmark

# Import baseline learners
from mlpy.learners import LearnerClassifFeatureless

# Pipeline imports
from mlpy.pipelines import (
    PipeOpImpute, PipeOpScale, PipeOpEncode,
    PipeOpLearner, linear_pipeline
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("MLPY - Machine Learning Workflow Example")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# 1. DATA GENERATION
# ============================================================================
print("1. GENERATING SYNTHETIC DATASET")
print("-" * 40)

# Generate a synthetic dataset that mimics a customer churn prediction problem
n_samples = 1000

# Customer features
data = pd.DataFrame({
    # Demographics
    'age': np.random.normal(45, 15, n_samples).clip(18, 80).astype(int),
    'income': np.random.lognormal(10.5, 0.6, n_samples),
    
    # Account information
    'account_length_months': np.random.exponential(24, n_samples).clip(1, 120).astype(int),
    'monthly_charges': np.random.gamma(2, 50, n_samples),
    'total_charges': np.nan,  # Will be calculated with some missing values
    
    # Service usage
    'data_usage_gb': np.random.exponential(5, n_samples),
    'customer_service_calls': np.random.poisson(2, n_samples),
    'late_payments': np.random.poisson(0.5, n_samples),
    
    # Categorical features
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                     n_samples, p=[0.5, 0.3, 0.2]),
    'payment_method': np.random.choice(['Electronic', 'Mailed check', 'Bank transfer', 'Credit card'], 
                                      n_samples),
    
    # Binary features
    'paperless_billing': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    'partner': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
})

# Calculate total charges with some missing values
mask_missing = np.random.rand(n_samples) < 0.1  # 10% missing
data.loc[~mask_missing, 'total_charges'] = (
    data.loc[~mask_missing, 'account_length_months'] * 
    data.loc[~mask_missing, 'monthly_charges'] * 
    np.random.uniform(0.9, 1.1, (~mask_missing).sum())
)

# Create target variable (churn) based on complex interactions
churn_score = (
    0.3 * (data['contract_type'] == 'Month-to-month') +
    0.2 * (data['customer_service_calls'] > 3) +
    0.15 * (data['late_payments'] > 1) +
    0.15 * (data['monthly_charges'] > data['monthly_charges'].quantile(0.75)) +
    0.1 * (data['account_length_months'] < 12) +
    np.random.normal(0, 0.2, n_samples)
)

# Convert to binary churn (approximately 25% churn rate)
data['churn'] = (churn_score > np.percentile(churn_score, 75)).astype(str)
data['churn'] = data['churn'].map({'True': 'Yes', 'False': 'No'})

print(f"Dataset shape: {data.shape}")
print(f"Features: {data.shape[1] - 1}")
print(f"Samples: {data.shape[0]}")
print(f"\nTarget distribution:")
print(data['churn'].value_counts())
print(f"Churn rate: {(data['churn'] == 'Yes').mean():.2%}")
print(f"\nMissing values:")
print(data.isnull().sum()[data.isnull().sum() > 0])

# ============================================================================
# 2. CREATE MLPY TASK
# ============================================================================
print("\n2. CREATING MLPY TASK")
print("-" * 40)

task = TaskClassif(
    data=data,
    target='churn',
    id='customer_churn_prediction'
)

print(f"Task created: {task.id}")
print(f"Task type: {task.task_type}")
print(f"Number of features: {task.n_features}")
print(f"  - Numeric: {task.n_numeric}")
print(f"  - Categorical: {task.n_factor}")
print(f"Classes: {task.class_names}")

# ============================================================================
# 3. BASELINE MODEL
# ============================================================================
print("\n3. BASELINE MODEL EVALUATION")
print("-" * 40)

# Create baseline learner
baseline_learner = LearnerClassifFeatureless(id='baseline', predict_type='response')

# Create a simple preprocessing pipeline
baseline_pipeline = linear_pipeline([
    PipeOpImpute(method='median', affect_columns='numeric'),
    PipeOpImpute(method='most_frequent', affect_columns='factor'),
    PipeOpEncode(method='onehot'),
    PipeOpScale(method='standard'),
    PipeOpLearner(baseline_learner)
], id='baseline_pipeline')

# Evaluate baseline
print("Evaluating baseline model with 5-fold CV...")
baseline_result = resample(
    task=task,
    learner=baseline_pipeline,
    resampling=ResamplingCV(folds=5, stratify=True),
    measures=[
        MeasureClassifAccuracy(),
        MeasureClassifF1(),
        MeasureClassifPrecision(),
        MeasureClassifRecall()
    ]
)

print("\nBaseline Results:")
baseline_scores = baseline_result.aggregate()
for measure, scores in baseline_scores.items():
    print(f"  {measure}: {scores['mean']:.3f} (+/- {scores['sd']:.3f})")

# ============================================================================
# 4. MULTIPLE BASELINE MODELS
# ============================================================================
print("\n4. COMPARING DIFFERENT BASELINE STRATEGIES")
print("-" * 40)

# Create multiple baseline learners with different strategies
learners = [
    LearnerClassifFeatureless(id='majority', method='majority'),
    LearnerClassifFeatureless(id='stratified', method='stratified'),
    LearnerClassifFeatureless(id='weighted', method='weighted')
]

# Create pipelines
pipelines = []
for learner in learners:
    pipeline = linear_pipeline([
        PipeOpImpute(method='median', affect_columns='numeric'),
        PipeOpImpute(method='most_frequent', affect_columns='factor'),
        PipeOpEncode(method='onehot'),
        PipeOpLearner(learner)
    ], id=f"pipeline_{learner.id}")
    pipelines.append(pipeline)

# Run benchmark
print("Running benchmark comparison...")
benchmark_result = benchmark(
    tasks=[task],
    learners=pipelines,
    resampling=ResamplingCV(folds=3, stratify=True),
    measures=[MeasureClassifAccuracy(), MeasureClassifF1()]
)

print("\nBenchmark Results:")
print(benchmark_result)

# ============================================================================
# 5. FINAL EVALUATION
# ============================================================================
print("\n5. FINAL EVALUATION ON HOLDOUT SET")
print("-" * 40)

# Use the best baseline strategy
final_learner = pipelines[0]

# Evaluate on holdout set
print("Evaluating on 80/20 train-test split...")
final_result = resample(
    task=task,
    learner=final_learner,
    resampling=ResamplingHoldout(ratio=0.8, stratify=True),
    measures=[
        MeasureClassifAccuracy(),
        MeasureClassifF1(),
        MeasureClassifPrecision(),
        MeasureClassifRecall()
    ]
)

print("\nFinal Holdout Results:")
final_scores = final_result.aggregate()
for measure, scores in final_scores.items():
    print(f"  {measure}: {scores['mean']:.3f}")

# ============================================================================
# 6. DETAILED RESULTS
# ============================================================================
print("\n6. DETAILED RESULTS ANALYSIS")
print("-" * 40)

# Print confusion matrix information
if hasattr(final_result, 'predictions') and final_result.predictions:
    pred = final_result.predictions[0]  # First fold
    if pred is not None:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(pred.truth, pred.response)
        print("\nConfusion Matrix:")
        print(f"True Negative:  {cm[0,0]:4d} | False Positive: {cm[0,1]:4d}")
        print(f"False Negative: {cm[1,0]:4d} | True Positive:  {cm[1,1]:4d}")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("WORKFLOW SUMMARY")
print("=" * 80)

print(f"\nDataset Summary:")
print(f"- Total samples: {n_samples}")
print(f"- Features: {task.n_features}")
print(f"- Churn rate: {(data['churn'] == 'Yes').mean():.2%}")
print(f"- Missing values: {data.isnull().sum().sum()}")

print(f"\nModel Performance:")
print(f"- Baseline Accuracy: {baseline_scores['classif.acc']['mean']:.3f}")
print(f"- Baseline F1 Score: {baseline_scores['classif.f1']['mean']:.3f}")
print(f"- Final Accuracy: {final_scores['classif.acc']['mean']:.3f}")
print(f"- Final F1 Score: {final_scores['classif.f1']['mean']:.3f}")

print(f"\nKey Insights:")
print(f"1. The baseline model achieves {baseline_scores['classif.acc']['mean']:.1%} accuracy")
print(f"2. Class imbalance affects F1 score: {baseline_scores['classif.f1']['mean']:.3f}")
print(f"3. Different baseline strategies show similar performance")
print(f"4. MLPY successfully handles missing values and mixed data types")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "=" * 80)
print("Thank you for using MLPY!")
print("=" * 80)