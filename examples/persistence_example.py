"""
Example of model persistence in MLPY.

This example demonstrates various ways to save and load models,
including basic serialization, model registry, and package export.
"""

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

# Import MLPY components
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners import LearnerRegrFeatureless
from mlpy.learners.sklearn import learner_sklearn
from mlpy.pipelines import PipeOpScale, PipeOpImpute, PipeOpLearner, linear_pipeline
from mlpy.resamplings import ResamplingCV
from mlpy.measures import MeasureRegrRMSE, MeasureClassifAcc
from mlpy.resample import resample

# Import persistence components
from mlpy.persistence import (
    save_model, load_model,
    ModelRegistry, export_model_package,
    PickleSerializer, JoblibSerializer
)

# Check for optional dependencies
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some examples will be skipped.")


def create_sample_data():
    """Create sample datasets for examples."""
    np.random.seed(42)
    
    # Regression data
    n_samples = 200
    X_reg = np.random.randn(n_samples, 5)
    y_reg = 2 * X_reg[:, 0] + X_reg[:, 1] - 0.5 * X_reg[:, 2] + np.random.randn(n_samples) * 0.5
    
    df_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(5)])
    df_reg['target'] = y_reg
    
    # Classification data
    X_class = np.random.randn(n_samples, 4)
    y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)
    
    df_class = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(4)])
    df_class['target'] = y_class
    
    return df_reg, df_class


def example_basic_save_load():
    """Example: Basic model saving and loading."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Save and Load")
    print("="*60)
    
    # Create data
    df_reg, _ = create_sample_data()
    task = TaskRegr(data=df_reg, target='target', id='housing_prices')
    
    # Train a simple model
    learner = LearnerRegrFeatureless(id='baseline_mean')
    learner.train(task)
    
    # Make predictions before saving
    pred_before = learner.predict(task)
    print(f"Predictions before saving - Mean: {pred_before.response.mean():.4f}")
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "baseline_model.pkl"
        
        # Save with metadata
        save_model(
            learner,
            model_path,
            metadata={
                "dataset": "housing_prices",
                "features": list(task.feature_names),
                "training_samples": task.nrow
            }
        )
        
        print(f"\nModel saved to: {model_path}")
        print(f"File size: {model_path.stat().st_size} bytes")
        
        # Load model
        loaded_learner = load_model(model_path)
        
        # Make predictions after loading
        pred_after = loaded_learner.predict(task)
        print(f"\nPredictions after loading - Mean: {pred_after.response.mean():.4f}")
        
        # Verify predictions are identical
        assert np.allclose(pred_before.response, pred_after.response)
        print("✓ Predictions match!")
        
        # Load with metadata
        bundle = load_model(model_path, return_bundle=True)
        print(f"\nMetadata: {bundle.metadata}")


def example_sklearn_models():
    """Example: Saving scikit-learn models."""
    if not SKLEARN_AVAILABLE:
        print("\nSkipping sklearn example - library not available")
        return
        
    print("\n" + "="*60)
    print("EXAMPLE 2: Scikit-learn Model Persistence")
    print("="*60)
    
    # Create data
    _, df_class = create_sample_data()
    task = TaskClassif(data=df_class, target='target', id='binary_classification')
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    learner = learner_sklearn(rf, id='random_forest')
    
    # Evaluate model
    result = resample(
        task=task,
        learner=learner,
        resampling=ResamplingCV(folds=3),
        measure=MeasureClassifAcc()
    )
    
    print(f"Cross-validation accuracy: {result.aggregate()['acc'][0]:.4f}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save with joblib (recommended for sklearn)
        model_path = Path(tmpdir) / "rf_model.joblib"
        
        save_model(
            learner,
            model_path,
            serializer="joblib",
            metadata={
                "cv_accuracy": result.aggregate()['acc'][0],
                "cv_std": result.aggregate()['acc'][1],
                "algorithm": "RandomForest",
                "n_estimators": 10
            }
        )
        
        print(f"\nModel saved with joblib to: {model_path}")
        
        # Load and verify
        loaded_learner = load_model(model_path)
        
        # Check feature importances
        importance = loaded_learner.importance()
        if importance is not None:
            print("\nFeature importances:")
            for feat, imp in zip(task.feature_names, importance):
                print(f"  {feat}: {imp:.4f}")


def example_pipeline_persistence():
    """Example: Saving ML pipelines."""
    if not SKLEARN_AVAILABLE:
        print("\nSkipping pipeline example - sklearn not available")
        return
        
    print("\n" + "="*60)
    print("EXAMPLE 3: Pipeline Persistence")
    print("="*60)
    
    # Create data with missing values
    df_reg, _ = create_sample_data()
    # Add some missing values
    df_reg.loc[10:20, 'feature_1'] = np.nan
    df_reg.loc[30:35, 'feature_3'] = np.nan
    
    task = TaskRegr(data=df_reg, target='target')
    
    # Create pipeline
    rf_learner = learner_sklearn(
        RandomForestRegressor(n_estimators=20, random_state=42),
        id='rf'
    )
    
    pipeline = linear_pipeline(
        PipeOpImpute(id="impute", strategy="mean"),
        PipeOpScale(id="scale", method="standard"),
        PipeOpLearner(rf_learner, id="learner")
    )
    
    print("Pipeline structure:")
    print(f"  1. Impute missing values (mean)")
    print(f"  2. Scale features (standardization)")
    print(f"  3. Random Forest Regressor")
    
    # Train pipeline
    pipeline.train(task)
    
    # Evaluate
    measure = MeasureRegrRMSE()
    pred = pipeline.predict(task)['output']
    rmse = measure.score(pred)
    print(f"\nTraining RMSE: {rmse:.4f}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save entire pipeline
        pipeline_path = Path(tmpdir) / "ml_pipeline.pkl"
        
        save_model(
            pipeline,
            pipeline_path,
            metadata={
                "pipeline_steps": ["impute", "scale", "rf"],
                "training_rmse": rmse,
                "n_features": len(task.feature_names)
            }
        )
        
        print(f"\nPipeline saved to: {pipeline_path}")
        
        # Load pipeline
        loaded_pipeline = load_model(pipeline_path)
        
        # Test on new data
        new_data = df_reg.iloc[:10].copy()
        new_data.loc[0, 'feature_2'] = np.nan  # Add missing value
        new_task = TaskRegr(data=new_data, target='target')
        
        # Predictions should handle missing values automatically
        new_pred = loaded_pipeline.predict(new_task)['output']
        print(f"\nPredictions on new data: {new_pred.response[:5]}")


def example_model_registry():
    """Example: Using model registry for organization."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Model Registry")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(tmpdir)
        
        # Create and train multiple models
        df_reg, df_class = create_sample_data()
        
        # Model 1: Baseline regressor
        task_reg = TaskRegr(data=df_reg, target='target')
        baseline = LearnerRegrFeatureless(id='baseline', response='mean')
        baseline.train(task_reg)
        
        # Register baseline
        registry.register_model(
            baseline,
            name="price_predictor",
            version="v1.0",
            tags=["baseline", "regression"],
            metadata={
                "description": "Simple mean baseline",
                "rmse": 2.156  # Example metric
            }
        )
        
        print("Registered: price_predictor v1.0 (baseline)")
        
        if SKLEARN_AVAILABLE:
            # Model 2: Improved model
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_learner = learner_sklearn(rf, id='rf_improved')
            rf_learner.train(task_reg)
            
            # Register improved version
            registry.register_model(
                rf_learner,
                name="price_predictor",
                version="v2.0",
                tags=["random_forest", "regression", "production"],
                metadata={
                    "description": "Random Forest with 50 trees",
                    "rmse": 0.834,  # Example metric
                    "improvement": "61% better than baseline"
                }
            )
            
            print("Registered: price_predictor v2.0 (random forest)")
            
            # Model 3: Classification model
            task_class = TaskClassif(data=df_class, target='target')
            lr = LogisticRegression(random_state=42)
            lr_learner = learner_sklearn(lr, id='logistic')
            lr_learner.train(task_class)
            
            registry.register_model(
                lr_learner,
                name="classifier",
                version="v1.0",
                tags=["logistic_regression", "classification"],
                metadata={
                    "accuracy": 0.92,
                    "use_case": "Binary classification"
                }
            )
            
            print("Registered: classifier v1.0 (logistic regression)")
        
        # List all models
        print("\nRegistry contents:")
        all_models = registry.list_models()
        for model_name, versions in all_models.items():
            print(f"  {model_name}: {versions}")
            
        # Load specific version
        if "price_predictor" in all_models and "v1.0" in all_models["price_predictor"]:
            model_v1 = registry.load_model("price_predictor", version="v1.0")
            print(f"\nLoaded v1.0: {type(model_v1).__name__}")
            
        # Load latest version
        if "price_predictor" in all_models:
            model_latest, metadata = registry.load_model(
                "price_predictor",
                return_metadata=True
            )
            print(f"Loaded latest: {metadata['version']} - {metadata.get('description', 'N/A')}")
            
        # Get metadata without loading model
        if "classifier" in all_models:
            meta = registry.get_metadata("classifier")
            print(f"\nClassifier metadata: {meta.get('use_case', 'N/A')}")


def example_export_package():
    """Example: Exporting model as a package."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Model Package Export")
    print("="*60)
    
    # Create and train a model
    df_reg, _ = create_sample_data()
    task = TaskRegr(data=df_reg, target='target')
    
    if SKLEARN_AVAILABLE:
        # Use a more interesting model
        from sklearn.ensemble import GradientBoostingRegressor
        gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
        learner = learner_sklearn(gb, id='gradient_boosting')
    else:
        # Fallback to simple model
        learner = LearnerRegrFeatureless(id='baseline')
        
    learner.train(task)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export as package
        package_path = Path(tmpdir) / "house_price_model.zip"
        
        export_model_package(
            learner,
            package_path,
            name="HousePricePredictor",
            metadata={
                "author": "Data Science Team",
                "version": "1.0.0",
                "description": "Production model for house price prediction",
                "features": list(task.feature_names),
                "target": "price",
                "performance": {
                    "rmse": 45000,
                    "mae": 32000,
                    "r2": 0.89
                }
            }
        )
        
        print(f"Model package exported to: {package_path}")
        print(f"Package size: {package_path.stat().st_size / 1024:.1f} KB")
        
        # Show package contents
        import zipfile
        with zipfile.ZipFile(package_path, 'r') as zf:
            print("\nPackage contents:")
            for filename in zf.namelist():
                file_info = zf.getinfo(filename)
                print(f"  {filename} ({file_info.file_size} bytes)")
                
        print("\nPackage ready for distribution!")
        print("Recipients can install requirements and use the model immediately.")


def example_custom_serialization():
    """Example: Custom serialization options."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Serialization")
    print("="*60)
    
    # Create model
    learner = LearnerRegrFeatureless(id='custom_example')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Example 1: Pickle with different protocols
        print("1. Pickle serialization:")
        
        # Default protocol (highest)
        path1 = tmpdir / "model_default.pkl"
        save_model(learner, path1, serializer=PickleSerializer())
        print(f"  Default protocol: {path1.stat().st_size} bytes")
        
        # Protocol 4 (Python 3.4+)
        path2 = tmpdir / "model_p4.pkl"
        save_model(learner, path2, serializer=PickleSerializer(protocol=4))
        print(f"  Protocol 4: {path2.stat().st_size} bytes")
        
        # Example 2: Joblib with compression
        if 'joblib' in globals():
            print("\n2. Joblib with compression:")
            
            # No compression
            path3 = tmpdir / "model_nocomp.joblib"
            save_model(learner, path3, serializer=JoblibSerializer(compression=0))
            print(f"  No compression: {path3.stat().st_size} bytes")
            
            # High compression
            path4 = tmpdir / "model_comp9.joblib"
            save_model(learner, path4, serializer=JoblibSerializer(compression=9))
            print(f"  Max compression: {path4.stat().st_size} bytes")
            
        # Example 3: Metadata only with JSON
        print("\n3. JSON metadata:")
        metadata = {
            "model_type": type(learner).__name__,
            "model_id": learner.id,
            "properties": list(learner.properties) if hasattr(learner, 'properties') else [],
            "hyperparameters": {
                "response": getattr(learner, 'response', 'mean')
            }
        }
        
        path5 = tmpdir / "model_meta.json"
        save_model(metadata, path5, serializer="json", create_bundle=False)
        print(f"  Metadata saved: {path5.stat().st_size} bytes")
        
        # Load and display
        import json
        with open(path5, 'r') as f:
            loaded_meta = json.load(f)
        print(f"  Content: {loaded_meta}")


def main():
    """Run all examples."""
    print("MLPY Model Persistence Examples")
    print("===============================")
    
    # Run examples
    example_basic_save_load()
    example_sklearn_models()
    example_pipeline_persistence()
    example_model_registry()
    example_export_package()
    example_custom_serialization()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    
    print("\nKey takeaways:")
    print("1. save_model() and load_model() provide simple persistence")
    print("2. Different serializers offer different trade-offs")
    print("3. ModelRegistry helps organize multiple models and versions")
    print("4. export_model_package() creates distributable packages")
    print("5. Metadata can be attached to any saved model")
    print("6. Pipelines and complex objects are fully supported")


if __name__ == "__main__":
    main()