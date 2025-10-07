"""Example: AutoML capabilities in MLPY.

This example demonstrates the AutoML features of MLPY including:
- Hyperparameter tuning
- Automatic feature engineering
- Pipeline optimization
- Model selection
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1, MeasureRegrRMSE, MeasureRegrR2
from mlpy.resamplings import ResamplingCV, ResamplingHoldout

# AutoML imports
from mlpy.automl import (
    ParamSet, ParamInt, ParamFloat, ParamCategorical,
    TunerGrid, TunerRandom, TunerBayesian,
    FeatureEngineer, create_auto_pipeline
)

# Try importing learners
try:
    from mlpy.learners import (
        LearnerRandomForest,
        LearnerLogisticRegression,
        LearnerSVM,
        LearnerGradientBoosting,
        LearnerRidge,
        LearnerElasticNet
    )
    from mlpy.pipelines import PipeOpScale, PipeOpImpute, PipeOpEncode, PipeOpLearner, linear_pipeline
    HAS_SKLEARN = True
except ImportError:
    print("This example requires scikit-learn to be installed.")
    HAS_SKLEARN = False


def create_sample_data_classification():
    """Create a sample classification dataset with various feature types."""
    np.random.seed(42)
    n_samples = 500
    
    # Numeric features
    numeric_1 = np.random.randn(n_samples)
    numeric_2 = np.random.exponential(2, n_samples)
    numeric_3 = np.random.uniform(-1, 1, n_samples)
    
    # Features with missing values
    numeric_with_na = np.random.randn(n_samples)
    numeric_with_na[np.random.choice(n_samples, 50, replace=False)] = np.nan
    
    # Categorical features
    categorical_1 = np.random.choice(['A', 'B', 'C'], n_samples)
    categorical_2 = np.random.choice(['X', 'Y', 'Z', 'W'], n_samples)
    
    # Target based on complex interaction
    target_prob = (
        0.3 * numeric_1 + 
        0.2 * (numeric_2 > 2) + 
        0.4 * (categorical_1 == 'A') +
        0.1 * (categorical_2 == 'X') +
        np.random.randn(n_samples) * 0.5
    )
    target = (target_prob > np.median(target_prob)).astype(int)
    target = ['Class_' + str(t) for t in target]
    
    # Create DataFrame
    data = pd.DataFrame({
        'num_feature_1': numeric_1,
        'num_feature_2': numeric_2,
        'num_feature_3': numeric_3,
        'num_with_missing': numeric_with_na,
        'cat_feature_1': categorical_1,
        'cat_feature_2': categorical_2,
        'target': target
    })
    
    return TaskClassif(data=data, target='target', id='automl_classification')


def example_hyperparameter_tuning():
    """Example of hyperparameter tuning with different tuners."""
    print("=== Hyperparameter Tuning Example ===\n")
    
    # Create task
    task = create_sample_data_classification()
    print(f"Task: {task.id}")
    print(f"Features: {task.n_features} ({task.n_numeric} numeric, {task.n_factor} categorical)")
    print(f"Observations: {task.n_obs}")
    print(f"Classes: {task.class_names}\n")
    
    # Define hyperparameter space for Random Forest
    print("Defining hyperparameter space for Random Forest...")
    param_set = ParamSet([
        ParamInt("n_estimators", lower=50, upper=200),
        ParamInt("max_depth", lower=3, upper=20),
        ParamInt("min_samples_split", lower=2, upper=20),
        ParamInt("min_samples_leaf", lower=1, upper=10),
        ParamCategorical("max_features", values=["sqrt", "log2", 0.5, 0.8])
    ])
    
    # Create base learner
    learner = LearnerRandomForest(random_state=42)
    
    # Grid Search
    print("\n1. Grid Search Tuning")
    tuner_grid = TunerGrid(resolution=3)
    result_grid = tuner_grid.tune(
        learner=learner,
        task=task,
        resampling=ResamplingCV(folds=3),
        measure=MeasureClassifAccuracy(),
        param_set=param_set
    )
    
    print(f"Best parameters: {result_grid.best_params}")
    print(f"Best score: {result_grid.best_score:.3f}")
    print(f"Configurations tested: {len(result_grid.results)}")
    
    # Random Search
    print("\n2. Random Search Tuning")
    tuner_random = TunerRandom(n_evals=20, random_state=42)
    result_random = tuner_random.tune(
        learner=learner,
        task=task,
        resampling=ResamplingCV(folds=3),
        measure=MeasureClassifAccuracy(),
        param_set=param_set
    )
    
    print(f"Best parameters: {result_random.best_params}")
    print(f"Best score: {result_random.best_score:.3f}")
    
    # Bayesian Optimization (if available)
    try:
        print("\n3. Bayesian Optimization")
        tuner_bayes = TunerBayesian(n_evals=15, random_state=42)
        result_bayes = tuner_bayes.tune(
            learner=learner,
            task=task,
            resampling=ResamplingCV(folds=3),
            measure=MeasureClassifAccuracy(),
            param_set=param_set
        )
        
        print(f"Best parameters: {result_bayes.best_params}")
        print(f"Best score: {result_bayes.best_score:.3f}")
    except ImportError:
        print("Skipping Bayesian optimization (requires optuna)")
    
    return result_grid


def example_auto_feature_engineering():
    """Example of automatic feature engineering."""
    print("\n\n=== Automatic Feature Engineering Example ===\n")
    
    # Create task with raw features
    task = create_sample_data_classification()
    
    # Create feature engineer
    print("Creating automatic feature engineer...")
    feature_engineer = FeatureEngineer(
        max_features=20,
        include_interactions=True,
        include_polynomial=True,
        include_ratios=True,
        include_statistical=True
    )
    
    # Apply feature engineering
    print(f"Original features: {task.n_features}")
    task_engineered = feature_engineer.transform(task)
    print(f"Engineered features: {task_engineered.n_features}")
    print(f"New features created: {task_engineered.n_features - task.n_features}")
    
    # Compare performance
    print("\nComparing performance with and without feature engineering...")
    
    learner = LearnerLogisticRegression(C=1.0, max_iter=1000)
    measure = MeasureClassifAccuracy()
    resampling = ResamplingCV(folds=5)
    
    # Original features
    from mlpy import resample
    result_original = resample(task, learner, resampling, measure)
    print(f"Original features - Accuracy: {result_original.aggregate()[measure.id]['mean']:.3f}")
    
    # Engineered features
    result_engineered = resample(task_engineered, learner, resampling, measure)
    print(f"Engineered features - Accuracy: {result_engineered.aggregate()[measure.id]['mean']:.3f}")
    
    # Show some of the created features
    print("\nSample of created features:")
    new_features = [f for f in task_engineered.feature_names if f not in task.feature_names]
    for feat in new_features[:5]:
        print(f"  - {feat}")


def example_auto_pipeline():
    """Example of automatic pipeline creation."""
    print("\n\n=== Automatic Pipeline Creation Example ===\n")
    
    # Create task
    task = create_sample_data_classification()
    
    # Create automatic pipeline
    print("Creating automatic pipeline with preprocessing and tuning...")
    
    # Define learners to try
    learners = [
        LearnerLogisticRegression(id='logreg'),
        LearnerRandomForest(id='rf'),
        LearnerGradientBoosting(id='gb', n_estimators=50)
    ]
    
    # Create auto pipeline for each learner
    results = {}
    
    for base_learner in learners:
        print(f"\nCreating pipeline for {base_learner.id}...")
        
        # Create automatic pipeline
        pipeline = create_auto_pipeline(
            learner=base_learner,
            task=task,
            include_impute=True,
            include_encode=True,
            include_scale=True,
            include_feature_select=False  # Could add feature selection
        )
        
        # Evaluate pipeline
        from mlpy import resample
        result = resample(
            task=task,
            learner=pipeline,
            resampling=ResamplingCV(folds=5),
            measures=[MeasureClassifAccuracy(), MeasureClassifF1()]
        )
        
        results[base_learner.id] = result
        print(f"Accuracy: {result.aggregate()['classif.acc']['mean']:.3f}")
        print(f"F1 Score: {result.aggregate()['classif.f1']['mean']:.3f}")
    
    return results


def example_full_automl():
    """Example of complete AutoML workflow."""
    print("\n\n=== Complete AutoML Workflow Example ===\n")
    
    # Create a more complex dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features with different distributions
    data = pd.DataFrame({
        'gaussian_1': np.random.randn(n_samples),
        'gaussian_2': np.random.randn(n_samples) * 2 + 1,
        'exponential': np.random.exponential(2, n_samples),
        'uniform': np.random.uniform(-1, 1, n_samples),
        'binary': np.random.choice([0, 1], n_samples),
        'categorical': np.random.choice(['cat_A', 'cat_B', 'cat_C', 'cat_D'], n_samples),
        'text_length': np.random.poisson(10, n_samples),  # Simulating text feature
        'count': np.random.poisson(5, n_samples),
        'percentage': np.random.beta(2, 5, n_samples)
    })
    
    # Add missing values
    for col in ['gaussian_1', 'exponential', 'percentage']:
        missing_idx = np.random.choice(n_samples, int(0.1 * n_samples), replace=False)
        data.loc[missing_idx, col] = np.nan
    
    # Create complex target
    target_score = (
        0.5 * data['gaussian_1'].fillna(0) +
        0.3 * (data['exponential'].fillna(0) > 2) +
        0.4 * (data['categorical'] == 'cat_A') +
        0.2 * data['binary'] +
        0.1 * np.log1p(data['count']) +
        np.random.randn(n_samples) * 0.5
    )
    data['target'] = (target_score > target_score.quantile(0.6)).astype(str)
    
    # Create task
    task = TaskClassif(data=data, target='target', id='complex_classification')
    print(f"Task: {task.id}")
    print(f"Shape: {task.shape}")
    print(f"Missing values: {task.data.isnull().sum().sum()}")
    
    # Step 1: Feature Engineering
    print("\n1. Automatic Feature Engineering")
    fe = FeatureEngineer(
        max_features=30,
        include_interactions=True,
        include_polynomial=True,
        include_statistical=True
    )
    task_fe = fe.transform(task)
    print(f"Features after engineering: {task_fe.n_features}")
    
    # Step 2: Create preprocessing pipeline
    print("\n2. Creating preprocessing pipeline")
    pipe_impute = PipeOpImpute(method="median", affect_columns="numeric")
    pipe_encode = PipeOpEncode(method="onehot")
    pipe_scale = PipeOpScale(method="standard")
    
    preprocessing = linear_pipeline([pipe_impute, pipe_encode, pipe_scale])
    
    # Step 3: Model selection with tuning
    print("\n3. Model selection with hyperparameter tuning")
    
    models_config = {
        'LogisticRegression': {
            'learner': LearnerLogisticRegression(),
            'param_set': ParamSet([
                ParamFloat("C", lower=0.01, upper=10.0, log=True),
                ParamCategorical("penalty", values=["l1", "l2"]),
                ParamCategorical("solver", values=["liblinear", "saga"])
            ])
        },
        'RandomForest': {
            'learner': LearnerRandomForest(),
            'param_set': ParamSet([
                ParamInt("n_estimators", lower=50, upper=300),
                ParamInt("max_depth", lower=3, upper=20),
                ParamCategorical("max_features", values=["sqrt", "log2", 0.3, 0.5])
            ])
        },
        'GradientBoosting': {
            'learner': LearnerGradientBoosting(),
            'param_set': ParamSet([
                ParamInt("n_estimators", lower=50, upper=200),
                ParamFloat("learning_rate", lower=0.01, upper=0.3),
                ParamInt("max_depth", lower=3, upper=10)
            ])
        }
    }
    
    best_score = 0
    best_model = None
    best_params = None
    
    tuner = TunerRandom(n_evals=10, random_state=42)
    
    for model_name, config in models_config.items():
        print(f"\nTuning {model_name}...")
        
        # Create full pipeline
        learner_op = PipeOpLearner(config['learner'])
        full_pipeline = linear_pipeline([preprocessing, learner_op])
        
        # Tune
        tune_result = tuner.tune(
            learner=full_pipeline,
            task=task_fe,
            resampling=ResamplingCV(folds=3),
            measure=MeasureClassifAccuracy(),
            param_set=config['param_set']
        )
        
        print(f"Best score: {tune_result.best_score:.3f}")
        print(f"Best params: {tune_result.best_params}")
        
        if tune_result.best_score > best_score:
            best_score = tune_result.best_score
            best_model = model_name
            best_params = tune_result.best_params
    
    print(f"\n=== AutoML Results ===")
    print(f"Best model: {best_model}")
    print(f"Best score: {best_score:.3f}")
    print(f"Best parameters: {best_params}")
    
    # Step 4: Final evaluation on holdout
    print("\n4. Final evaluation on holdout set")
    
    # Create best pipeline with tuned parameters
    if best_model == 'LogisticRegression':
        final_learner = LearnerLogisticRegression(**best_params)
    elif best_model == 'RandomForest':
        final_learner = LearnerRandomForest(**best_params)
    else:
        final_learner = LearnerGradientBoosting(**best_params)
    
    final_pipeline = linear_pipeline([
        preprocessing,
        PipeOpLearner(final_learner)
    ])
    
    # Evaluate on holdout
    from mlpy import resample
    final_result = resample(
        task=task_fe,
        learner=final_pipeline,
        resampling=ResamplingHoldout(ratio=0.8),
        measures=[MeasureClassifAccuracy(), MeasureClassifF1()]
    )
    
    print(f"Holdout Accuracy: {final_result.aggregate()['classif.acc']['mean']:.3f}")
    print(f"Holdout F1 Score: {final_result.aggregate()['classif.f1']['mean']:.3f}")
    
    return final_pipeline, final_result


def example_automl_regression():
    """Example of AutoML for regression tasks."""
    print("\n\n=== AutoML for Regression Example ===\n")
    
    # Create regression dataset
    np.random.seed(42)
    n_samples = 800
    
    # Features
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    X3 = np.random.exponential(1, n_samples)
    X4 = np.random.uniform(-2, 2, n_samples)
    cat = np.random.choice(['A', 'B', 'C'], n_samples)
    
    # Target with non-linear relationships
    y = (
        2 * X1 +
        X2 ** 2 +
        np.sin(X3) +
        0.5 * X4 * (cat == 'A') +
        np.random.randn(n_samples) * 0.5
    )
    
    data = pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4,
        'category': cat,
        'y': y
    })
    
    task = TaskRegr(data=data, target='y', id='automl_regression')
    print(f"Regression task: {task.n_obs} observations, {task.n_features} features")
    
    # Feature engineering for non-linear relationships
    fe = FeatureEngineer(
        max_features=25,
        include_polynomial=True,  # Important for X2^2
        include_interactions=True,
        include_sin_cos=True  # Important for sin(X3)
    )
    task_fe = fe.transform(task)
    print(f"Features after engineering: {task_fe.n_features}")
    
    # Preprocessing
    preprocessing = linear_pipeline([
        PipeOpEncode(method="onehot"),
        PipeOpScale(method="standard")
    ])
    
    # Try different models
    models = {
        'Ridge': {
            'learner': LearnerRidge(),
            'params': ParamSet([ParamFloat("alpha", lower=0.01, upper=10.0, log=True)])
        },
        'ElasticNet': {
            'learner': LearnerElasticNet(),
            'params': ParamSet([
                ParamFloat("alpha", lower=0.01, upper=1.0),
                ParamFloat("l1_ratio", lower=0.1, upper=0.9)
            ])
        }
    }
    
    # Quick tuning
    tuner = TunerGrid(resolution=5)
    best_rmse = float('inf')
    
    for name, config in models.items():
        print(f"\nTuning {name}...")
        pipeline = linear_pipeline([preprocessing, PipeOpLearner(config['learner'])])
        
        result = tuner.tune(
            learner=pipeline,
            task=task_fe,
            resampling=ResamplingCV(folds=3),
            measure=MeasureRegrRMSE(),
            param_set=config['params']
        )
        
        print(f"Best RMSE: {result.best_score:.3f}")
        
        if result.best_score < best_rmse:
            best_rmse = result.best_score
            best_model = name


def main():
    """Run all AutoML examples."""
    if not HAS_SKLEARN:
        print("This example requires scikit-learn. Install with: pip install scikit-learn")
        return
        
    try:
        # Basic hyperparameter tuning
        tune_result = example_hyperparameter_tuning()
        
        # Feature engineering
        example_auto_feature_engineering()
        
        # Automatic pipelines
        pipeline_results = example_auto_pipeline()
        
        # Complete AutoML workflow
        best_pipeline, final_result = example_full_automl()
        
        # Regression AutoML
        example_automl_regression()
        
        print("\n\n=== AutoML Examples Completed Successfully! ===")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()