"""
Complete Machine Learning Workflow with MLPY
=============================================

This example demonstrates a typical end-to-end machine learning workflow using MLPY,
including data preparation, model selection, evaluation, and interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# MLPY imports
from mlpy.tasks import TaskClassif
from mlpy.measures import (
    MeasureClassifAccuracy, MeasureClassifF1, 
    MeasureClassifAUC, MeasureClassifPrecision, MeasureClassifRecall
)
from mlpy.resamplings import ResamplingCV, ResamplingHoldout
from mlpy import resample, benchmark

# Import learners (assuming sklearn is available)
try:
    from mlpy.learners import (
        LearnerLogisticRegression,
        LearnerRandomForest,
        LearnerGradientBoosting,
        LearnerSVM,
        LearnerKNN
    )
    HAS_SKLEARN = True
except ImportError:
    print("Scikit-learn not available. Using baseline learners.")
    from mlpy.learners import LearnerClassifFeatureless
    HAS_SKLEARN = False

# Pipeline imports
from mlpy.pipelines import (
    PipeOpImpute, PipeOpScale, PipeOpEncode,
    PipeOpLearner, linear_pipeline
)

# AutoML imports
from mlpy.automl import (
    ParamSet, ParamInt, ParamFloat, ParamCategorical,
    TunerRandom, FeatureEngineer
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("MLPY - Complete Machine Learning Workflow Example")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# 1. DATA GENERATION
# ============================================================================
print("1. GENERATING SYNTHETIC DATASET")
print("-" * 40)

# Generate a synthetic dataset that mimics a customer churn prediction problem
n_samples = 2000

# Customer features
data = pd.DataFrame({
    # Demographics
    'age': np.random.normal(45, 15, n_samples).clip(18, 80).astype(int),
    'income': np.random.lognormal(10.5, 0.6, n_samples),  # Log-normal income distribution
    
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
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                        n_samples, p=[0.3, 0.5, 0.2]),
    
    # Binary features
    'paperless_billing': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    'partner': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    'dependents': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
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
    0.1 * (data['internet_service'] == 'Fiber optic') +
    0.05 * (1 - data['paperless_billing']) +
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

if HAS_SKLEARN:
    # Simple logistic regression as baseline
    baseline_learner = LearnerLogisticRegression(
        id='baseline_logreg',
        C=1.0,
        class_weight='balanced'  # Handle class imbalance
    )
else:
    baseline_learner = LearnerClassifFeatureless(id='baseline')

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
        MeasureClassifAUC(),
        MeasureClassifPrecision(),
        MeasureClassifRecall()
    ]
)

print("\nBaseline Results:")
baseline_scores = baseline_result.aggregate()
for measure, scores in baseline_scores.items():
    print(f"  {measure}: {scores['mean']:.3f} (+/- {scores['sd']:.3f})")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\n4. AUTOMATIC FEATURE ENGINEERING")
print("-" * 40)

# Apply automatic feature engineering
fe = FeatureEngineer(
    max_features=50,
    include_interactions=True,
    include_polynomial=False,  # Not useful for this type of data
    include_ratios=True,
    include_statistical=True
)

print("Applying feature engineering...")
task_engineered = fe.transform(task)
print(f"Original features: {task.n_features}")
print(f"Engineered features: {task_engineered.n_features}")
print(f"New features created: {task_engineered.n_features - task.n_features}")

# Show some created features
new_features = [f for f in task_engineered.feature_names if f not in task.feature_names]
print("\nSample of created features:")
for feat in new_features[:5]:
    print(f"  - {feat}")

# ============================================================================
# 5. MODEL COMPARISON
# ============================================================================
print("\n5. MODEL COMPARISON")
print("-" * 40)

if HAS_SKLEARN:
    # Define multiple learners to compare
    learners = [
        LearnerLogisticRegression(
            id='logreg_balanced',
            C=1.0,
            class_weight='balanced',
            max_iter=1000
        ),
        LearnerRandomForest(
            id='rf_balanced',
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ),
        LearnerGradientBoosting(
            id='gb',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        LearnerSVM(
            id='svm_rbf',
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True  # Need this for AUC
        ),
        LearnerKNN(
            id='knn',
            n_neighbors=10,
            weights='distance'
        )
    ]
else:
    learners = [LearnerClassifFeatureless(id=f'baseline_{i}') for i in range(3)]

# Create pipelines for each learner
pipelines = []
for learner in learners:
    pipeline = linear_pipeline([
        PipeOpImpute(method='median', affect_columns='numeric'),
        PipeOpImpute(method='most_frequent', affect_columns='factor'),
        PipeOpEncode(method='onehot'),
        PipeOpScale(method='standard'),
        PipeOpLearner(learner)
    ], id=f"pipeline_{learner.id}")
    pipelines.append(pipeline)

# Run benchmark
print("Running benchmark comparison...")
benchmark_result = benchmark(
    tasks=[task_engineered],  # Use engineered features
    learners=pipelines,
    resampling=ResamplingCV(folds=5, stratify=True),
    measures=[
        MeasureClassifAccuracy(),
        MeasureClassifF1(),
        MeasureClassifAUC()
    ]
)

print("\nBenchmark Results:")
print(benchmark_result)

# Get detailed results for each measure
print("\nDetailed Results by Measure:")
for measure_id in ['classif.acc', 'classif.f1', 'classif.auc']:
    print(f"\n{measure_id.upper()}:")
    rankings = benchmark_result.rank_learners(measure_id)
    for _, row in rankings.iterrows():
        print(f"  {row['learner']:20s}: {row['mean_score']:.3f} (rank {int(row['rank'])})")

# ============================================================================
# 6. HYPERPARAMETER TUNING
# ============================================================================
print("\n6. HYPERPARAMETER TUNING FOR BEST MODEL")
print("-" * 40)

# Select the best model from benchmark (based on F1 score)
best_learner_id = benchmark_result.rank_learners('classif.f1').iloc[0]['learner']
print(f"Best model from benchmark: {best_learner_id}")

if HAS_SKLEARN and 'rf' in best_learner_id:
    # Tune Random Forest hyperparameters
    param_set = ParamSet([
        ParamInt("n_estimators", lower=50, upper=200),
        ParamInt("max_depth", lower=5, upper=20),
        ParamInt("min_samples_split", lower=2, upper=20),
        ParamInt("min_samples_leaf", lower=1, upper=10),
        ParamCategorical("max_features", values=["sqrt", "log2", 0.3, 0.5])
    ])
    
    print("\nTuning Random Forest hyperparameters...")
    tuner = TunerRandom(n_evals=20, random_state=42)
    
    # Get the best learner pipeline
    best_pipeline = next(p for p in pipelines if p.id == best_learner_id)
    
    tune_result = tuner.tune(
        learner=best_pipeline,
        task=task_engineered,
        resampling=ResamplingCV(folds=3, stratify=True),
        measure=MeasureClassifF1(),
        param_set=param_set
    )
    
    print(f"\nBest parameters found:")
    for param, value in tune_result.best_params.items():
        print(f"  {param}: {value}")
    print(f"Best CV F1 score: {tune_result.best_score:.3f}")
else:
    print("Skipping hyperparameter tuning for baseline model")

# ============================================================================
# 7. FINAL EVALUATION
# ============================================================================
print("\n7. FINAL EVALUATION ON HOLDOUT SET")
print("-" * 40)

# Use the best model for final evaluation
final_learner = pipelines[0]  # Would be the tuned model in practice

# Evaluate on holdout set
print("Evaluating on 80/20 train-test split...")
final_result = resample(
    task=task_engineered,
    learner=final_learner,
    resampling=ResamplingHoldout(ratio=0.8, stratify=True),
    measures=[
        MeasureClassifAccuracy(),
        MeasureClassifF1(),
        MeasureClassifAUC(),
        MeasureClassifPrecision(),
        MeasureClassifRecall()
    ]
)

print("\nFinal Holdout Results:")
final_scores = final_result.aggregate()
for measure, scores in final_scores.items():
    print(f"  {measure}: {scores['mean']:.3f}")

# ============================================================================
# 8. RESULTS VISUALIZATION
# ============================================================================
print("\n8. VISUALIZING RESULTS")
print("-" * 40)

# Create a results summary plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Model Comparison
ax1 = axes[0, 0]
if len(pipelines) > 1:
    model_names = [p.id.replace('pipeline_', '') for p in pipelines]
    f1_scores = []
    for pipeline in pipelines:
        scores = benchmark_result.get_scores(pipeline.id, task_engineered.id, 'classif.f1')
        f1_scores.append(np.mean(scores))
    
    ax1.bar(model_names, f1_scores, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Model Comparison (F1 Score)')
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for i, (name, score) in enumerate(zip(model_names, f1_scores)):
        ax1.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')

# 2. Feature Engineering Impact
ax2 = axes[0, 1]
baseline_f1 = baseline_scores['classif.f1']['mean']
if 'tune_result' in locals():
    tuned_f1 = tune_result.best_score
else:
    tuned_f1 = f1_scores[0] if f1_scores else baseline_f1

improvements = ['Baseline', 'With FE', 'Tuned']
scores = [baseline_f1, f1_scores[0] if f1_scores else baseline_f1, tuned_f1]
colors = ['coral', 'lightgreen', 'gold']

bars = ax2.bar(improvements, scores, color=colors, edgecolor='black')
ax2.set_ylabel('F1 Score')
ax2.set_title('Progressive Improvements')
ax2.set_ylim(0, 1)

for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{score:.3f}', ha='center', va='bottom')

# 3. Cross-validation scores distribution
ax3 = axes[1, 0]
if hasattr(final_result, 'scores'):
    cv_scores = final_result.scores['classif.f1']
    ax3.boxplot([cv_scores], labels=['F1 Score'])
    ax3.scatter([1]*len(cv_scores), cv_scores, alpha=0.5)
    ax3.set_title('Cross-Validation Score Distribution')
    ax3.set_ylabel('Score')

# 4. Summary statistics
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
Model Performance Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset: Customer Churn Prediction
Samples: {n_samples} | Features: {task_engineered.n_features}
Churn Rate: {(data['churn'] == 'Yes').mean():.1%}

Best Model: {best_learner_id.replace('pipeline_', '')}

Final Performance (Holdout):
• Accuracy:  {final_scores['classif.acc']['mean']:.3f}
• F1 Score:  {final_scores['classif.f1']['mean']:.3f}
• AUC:       {final_scores.get('classif.auc', {}).get('mean', 'N/A')}
• Precision: {final_scores['classif.precision']['mean']:.3f}
• Recall:    {final_scores['classif.recall']['mean']:.3f}

Improvement over baseline:
{((final_scores['classif.f1']['mean'] - baseline_f1) / baseline_f1 * 100):+.1f}%
"""
ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('mlpy_results.png', dpi=150, bbox_inches='tight')
print("Results visualization saved to 'mlpy_results.png'")

# ============================================================================
# 9. CONCLUSIONS
# ============================================================================
print("\n" + "=" * 80)
print("WORKFLOW COMPLETED")
print("=" * 80)
print(f"\nKey Findings:")
print(f"1. Baseline F1 Score: {baseline_f1:.3f}")
print(f"2. Best Model: {best_learner_id.replace('pipeline_', '')}")
print(f"3. Final F1 Score: {final_scores['classif.f1']['mean']:.3f}")
print(f"4. Improvement: {((final_scores['classif.f1']['mean'] - baseline_f1) / baseline_f1 * 100):+.1f}%")
print(f"5. Feature Engineering added {task_engineered.n_features - task.n_features} new features")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# 10. OPTIONAL: MODEL INTERPRETATION (if SHAP/LIME available)
# ============================================================================
try:
    from mlpy.interpretability import SHAPInterpreter, plot_feature_importance
    
    print("\n10. MODEL INTERPRETATION")
    print("-" * 40)
    
    # Train the best model on full data for interpretation
    final_learner.train(task_engineered)
    
    # Use SHAP for interpretation
    interpreter = SHAPInterpreter(explainer_type="auto")
    interpretation = interpreter.interpret(
        learner=final_learner,
        task=task_engineered,
        indices=[0, 1, 2],  # Explain first 3 instances
        compute_global=True
    )
    
    if interpretation.has_global_importance():
        print("\nTop 10 Most Important Features:")
        top_features = interpretation.global_importance.top_features(10)
        for i, feat in enumerate(top_features, 1):
            print(f"  {i}. {feat}")
            
except ImportError:
    print("\n(Model interpretation skipped - SHAP not available)")

print("\n" + "=" * 80)
print("Thank you for using MLPY!")
print("=" * 80)