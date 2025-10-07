"""
Demo of H2O integration with MLPY.

This example shows how to use H2O models within the MLPY framework.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Import MLPY components
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners.h2o_wrapper import learner_h2o, LearnerClassifH2O, LearnerRegrH2O
from mlpy.resamplings import ResamplingCV
from mlpy.measures import MeasureClassifAccuracy, MeasureRegrRMSE
from mlpy import resample, benchmark

# Import H2O models
try:
    import h2o
    from h2o.estimators import (
        H2ORandomForestEstimator,
        H2OGradientBoostingEstimator,
        H2ODeepLearningEstimator,
        H2OGeneralizedLinearEstimator,
        H2OXGBoostEstimator
    )
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    print("H2O is not installed. Install it with: pip install h2o")


def demo_h2o_classification():
    """Demonstrate H2O classification with MLPY."""
    print("\n" + "="*60)
    print("H2O CLASSIFICATION DEMO")
    print("="*60)
    
    # Initialize H2O
    h2o.init()
    
    # Create synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    # Create MLPY task
    task = TaskClassif(data=df, target='target')
    print(f"\nCreated classification task: {task.nrow} samples, {len(task.feature_names)} features")
    print(f"Classes: {task.class_names}")
    
    # Create H2O learners
    learners = [
        learner_h2o(
            H2ORandomForestEstimator(ntrees=50, seed=42),
            id="h2o_rf"
        ),
        learner_h2o(
            H2OGradientBoostingEstimator(ntrees=50, seed=42),
            id="h2o_gbm"
        ),
        learner_h2o(
            H2ODeepLearningEstimator(
                hidden=[20, 20],
                epochs=10,
                seed=42
            ),
            id="h2o_dl"
        )
    ]
    
    # Also compare with sklearn via MLPY
    from mlpy.learners import learner_sklearn
    from sklearn.ensemble import RandomForestClassifier
    learners.append(
        learner_sklearn(
            RandomForestClassifier(n_estimators=50, random_state=42),
            id="sklearn_rf"
        )
    )
    
    # Benchmark models
    print("\n🏃 Running benchmark...")
    result = benchmark(
        tasks=[task],
        learners=learners,
        resampling=ResamplingCV(folds=5),
        measures=MeasureClassifAccuracy()
    )
    
    # Show results
    print("\n📊 Results:")
    print(result.aggregate())
    
    # Train a single model and inspect
    print("\n🔍 Training H2O Random Forest for detailed inspection...")
    h2o_rf = learners[0]
    h2o_rf.train(task)
    
    # Get feature importances
    if h2o_rf.feature_importances is not None:
        print("\n📊 Feature Importances (top 10):")
        print(h2o_rf.feature_importances.head(10))
    
    # Make predictions
    pred = h2o_rf.predict(task)
    print(f"\n✅ Predictions shape: {pred.response.shape}")
    print(f"Accuracy: {np.mean(pred.response == pred.truth):.3f}")


def demo_h2o_regression():
    """Demonstrate H2O regression with MLPY."""
    print("\n" + "="*60)
    print("H2O REGRESSION DEMO")
    print("="*60)
    
    # Create synthetic regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    # Create MLPY task
    task = TaskRegr(data=df, target='target')
    print(f"\nCreated regression task: {task.nrow} samples, {len(task.feature_names)} features")
    
    # Create H2O learners
    learners = [
        learner_h2o(
            H2ORandomForestEstimator(ntrees=50, seed=42),
            id="h2o_rf"
        ),
        learner_h2o(
            H2OGradientBoostingEstimator(ntrees=50, seed=42),
            id="h2o_gbm"
        ),
        learner_h2o(
            H2OGeneralizedLinearEstimator(family="gaussian"),
            id="h2o_glm"
        )
    ]
    
    # Run evaluation
    print("\n🏃 Evaluating models...")
    for learner in learners:
        result = resample(
            task=task,
            learner=learner,
            resampling=ResamplingCV(folds=3),
            measures=MeasureRegrRMSE()
        )
        scores = result.aggregate()
        print(f"{learner.id}: RMSE = {scores['regr.rmse'].mean():.3f} ± {scores['regr.rmse'].std():.3f}")


def demo_h2o_automl():
    """Demonstrate H2O AutoML integration."""
    print("\n" + "="*60)
    print("H2O AUTOML DEMO")
    print("="*60)
    
    try:
        from h2o.automl import H2OAutoML
    except ImportError:
        print("H2O AutoML not available")
        return
    
    # Create a wrapper for H2O AutoML
    class LearnerH2OAutoML(LearnerClassifH2O):
        """Special wrapper for H2O AutoML."""
        
        def __init__(self, max_runtime_secs=60, **kwargs):
            self.max_runtime_secs = max_runtime_secs
            self.automl = None
            super().__init__(
                estimator=H2OGeneralizedLinearEstimator(),  # Placeholder
                id="h2o_automl",
                **kwargs
            )
            
        def train(self, task, row_ids=None):
            """Train using AutoML."""
            # Prepare data
            h2o_frame, feature_cols, target_col = self._prepare_h2o_frame(task, row_ids)
            
            # Run AutoML
            self.automl = H2OAutoML(
                max_runtime_secs=self.max_runtime_secs,
                seed=42
            )
            self.automl.train(
                x=feature_cols,
                y=target_col,
                training_frame=h2o_frame
            )
            
            # Use the best model
            self._model = self.automl.leader
            self._feature_names = feature_cols
            self._target_name = target_col
            self._train_task = task
            
            return self
    
    # Create classification data
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df['target'] = y
    
    task = TaskClassif(data=df, target='target')
    
    # Run AutoML
    print("\n🤖 Running H2O AutoML (60 seconds)...")
    automl_learner = LearnerH2OAutoML(max_runtime_secs=60)
    automl_learner.train(task)
    
    # Evaluate
    pred = automl_learner.predict(task)
    accuracy = np.mean(pred.response == pred.truth)
    print(f"\n✅ AutoML Best Model: {automl_learner._model.__class__.__name__}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Show leaderboard
    if automl_learner.automl:
        print("\n📊 AutoML Leaderboard:")
        lb = automl_learner.automl.leaderboard
        print(lb.head())


def demo_h2o_with_pipelines():
    """Demonstrate using H2O models in MLPY pipelines."""
    print("\n" + "="*60)
    print("H2O WITH MLPY PIPELINES DEMO")
    print("="*60)
    
    from mlpy.pipelines import PipeOpScale, PipeOpSelect, PipeOpLearner, linear_pipeline, GraphLearner
    
    # Create data
    X, y = make_classification(n_samples=500, n_features=50, n_informative=10, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df['target'] = y
    
    task = TaskClassif(data=df, target='target')
    
    # Create pipeline with H2O model
    h2o_model = learner_h2o(H2ORandomForestEstimator(ntrees=20))
    
    # Build pipeline: Scale -> Select features -> H2O model
    graph = linear_pipeline(
        PipeOpScale(),
        PipeOpSelect(k=10),
        PipeOpLearner(h2o_model)
    )
    
    pipeline = GraphLearner(graph, id="h2o_pipeline")
    
    # Evaluate pipeline
    print("\n🔧 Evaluating H2O pipeline...")
    result = resample(
        task=task,
        learner=pipeline,
        resampling=ResamplingCV(folds=3),
        measures=MeasureClassifAccuracy()
    )
    
    scores = result.aggregate()
    print(f"Pipeline accuracy: {scores['classif.acc'].mean():.3f} ± {scores['classif.acc'].std():.3f}")


if __name__ == "__main__":
    if not H2O_AVAILABLE:
        print("H2O is not installed. Install it with: pip install h2o")
        print("\nExample of what the integration would look like:")
        print("""
        # Create H2O learner
        from mlpy.learners.h2o_wrapper import learner_h2o
        from h2o.estimators import H2ORandomForestEstimator
        
        # Wrap any H2O model
        h2o_rf = learner_h2o(H2ORandomForestEstimator(ntrees=100))
        
        # Use it like any MLPY learner
        h2o_rf.train(task)
        predictions = h2o_rf.predict(task)
        """)
    else:
        # Run demos
        demo_h2o_classification()
        demo_h2o_regression()
        demo_h2o_automl()
        demo_h2o_with_pipelines()
        
        # Shutdown H2O
        h2o.cluster().shutdown()