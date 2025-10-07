#!/usr/bin/env python
"""
Quick validation script to test MLPY framework functionality.
"""

import sys
import traceback
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that basic imports work."""
    print("\n1. Testing Imports...")
    try:
        # Core imports
        from mlpy.learners.base import Learner
        from mlpy.tasks import Task, TaskClassif, TaskRegr
        from mlpy.predictions import PredictionClassif, PredictionRegr
        from mlpy.resamplings import ResamplingHoldout, ResamplingCV
        from mlpy.measures import MeasureClassifAccuracy, MeasureRegrMSE
        print("   [OK] Core imports successful")
        
        # Model imports
        from mlpy.learners.baseline import LearnerClassifFeatureless, LearnerRegrFeatureless
        from mlpy.learners.ensemble import LearnerVoting, LearnerStacking
        print("   [OK] Model imports successful")
        
        return True
    except Exception as e:
        print(f"   [FAIL] Import failed: {e}")
        return False


def test_basic_classification():
    """Test basic classification workflow."""
    print("\n2. Testing Basic Classification...")
    try:
        import pandas as pd
        import numpy as np
        from mlpy.tasks import TaskClassif
        from mlpy.learners.baseline import LearnerClassifFeatureless
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'target': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Create task
        task = TaskClassif(data=df, target='target')
        print(f"   [OK] Created task with {task.nrow} samples")
        
        # Train model
        learner = LearnerClassifFeatureless()
        learner.train(task)
        print(f"   [OK] Trained baseline classifier")
        
        # Make predictions
        predictions = learner.predict(task)
        print(f"   [OK] Made {len(predictions.response)} predictions")
        
        return True
    except Exception as e:
        print(f"   [FAIL] Classification test failed: {e}")
        traceback.print_exc()
        return False


def test_basic_regression():
    """Test basic regression workflow."""
    print("\n3. Testing Basic Regression...")
    try:
        import pandas as pd
        import numpy as np
        from mlpy.tasks import TaskRegr
        from mlpy.learners.baseline import LearnerRegrFeatureless
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(100) * 0.1
        
        df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
        df['target'] = y
        
        # Create task
        task = TaskRegr(data=df, target='target')
        print(f"   [OK] Created regression task")
        
        # Train model
        learner = LearnerRegrFeatureless()
        learner.train(task)
        print(f"   [OK] Trained baseline regressor")
        
        # Make predictions
        predictions = learner.predict(task)
        print(f"   [OK] Made {len(predictions.response)} predictions")
        
        return True
    except Exception as e:
        print(f"   [FAIL] Regression test failed: {e}")
        traceback.print_exc()
        return False


def test_ensemble():
    """Test ensemble functionality."""
    print("\n4. Testing Ensemble...")
    try:
        import pandas as pd
        import numpy as np
        from mlpy.tasks import TaskClassif
        from mlpy.learners.baseline import LearnerClassifFeatureless
        from mlpy.learners.ensemble import LearnerVoting
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_1': np.random.randn(50),
            'feature_2': np.random.randn(50),
            'target': np.random.choice(['A', 'B'], 50)
        })
        
        # Create task
        task = TaskClassif(data=df, target='target')
        
        # Create ensemble
        base_learners = [
            LearnerClassifFeatureless(),
            LearnerClassifFeatureless()
        ]
        
        ensemble = LearnerVoting(base_learners=base_learners, voting='hard')
        print(f"   [OK] Created voting ensemble with {len(base_learners)} learners")
        
        # Train
        ensemble.train(task)
        print(f"   [OK] Trained ensemble")
        
        # Predict
        predictions = ensemble.predict(task)
        print(f"   [OK] Ensemble predictions: {len(predictions.response)} samples")
        
        return True
    except Exception as e:
        print(f"   [FAIL] Ensemble test failed: {e}")
        traceback.print_exc()
        return False


def test_resampling():
    """Test resampling strategies."""
    print("\n5. Testing Resampling...")
    try:
        import pandas as pd
        import numpy as np
        from mlpy.tasks import TaskClassif
        from mlpy.resamplings import ResamplingHoldout, ResamplingCV
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'target': np.random.choice(['A', 'B'], 100)
        })
        
        task = TaskClassif(data=df, target='target')
        
        # Test Holdout
        holdout = ResamplingHoldout(ratio=0.3)
        holdout_instance = holdout.instantiate(task)
        train_idx = holdout_instance.train_set(0)
        test_idx = holdout_instance.test_set(0)
        print(f"   [OK] Holdout: {len(train_idx)} train, {len(test_idx)} test")
        
        # Test CV
        cv = ResamplingCV(folds=5)
        cv_instance = cv.instantiate(task)
        print(f"   [OK] Cross-validation: {cv.folds} folds created")
        
        return True
    except Exception as e:
        print(f"   [FAIL] Resampling test failed: {e}")
        traceback.print_exc()
        return False


def test_measures():
    """Test performance measures."""
    print("\n6. Testing Measures...")
    try:
        import numpy as np
        from mlpy.measures import MeasureClassifAccuracy, MeasureRegrMSE
        
        # Classification measure
        y_true = ['A', 'B', 'A', 'B', 'A']
        y_pred = ['A', 'B', 'A', 'A', 'A']
        
        measure = MeasureClassifAccuracy()
        accuracy = measure.score(y_true, y_pred)
        print(f"   [OK] Classification accuracy: {accuracy:.2f}")
        
        # Regression measure
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        measure = MeasureRegrMSE()
        mse = measure.score(y_true, y_pred)
        print(f"   [OK] Regression MSE: {mse:.4f}")
        
        return True
    except Exception as e:
        print(f"   [FAIL] Measures test failed: {e}")
        traceback.print_exc()
        return False


def test_model_registry():
    """Test model registry functionality."""
    print("\n7. Testing Model Registry...")
    try:
        from mlpy.model_registry.registry import (
            ModelRegistry, ModelMetadata, ModelCategory, 
            TaskType, Complexity
        )
        
        # Create registry
        registry = ModelRegistry()
        
        # Register a model
        metadata = ModelMetadata(
            name="test_model",
            display_name="Test Model",
            description="A test model",
            category=ModelCategory.TRADITIONAL_ML,
            class_path="mlpy.test.TestModel",
            task_types=[TaskType.CLASSIFICATION],
            complexity=Complexity.LOW
        )
        
        registry.register(metadata)
        print(f"   [OK] Model registered successfully")
        
        # Search models
        models = registry.search(task_type=TaskType.CLASSIFICATION)
        print(f"   [OK] Found {len(models)} classification models")
        
        return True
    except Exception as e:
        print(f"   [FAIL] Model registry test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("="*60)
    print("MLPY FRAMEWORK VALIDATION")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Classification", test_basic_classification),
        ("Regression", test_basic_regression),
        ("Ensemble", test_ensemble),
        ("Resampling", test_resampling),
        ("Measures", test_measures),
        ("Model Registry", test_model_registry)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n[FAIL] {name} test crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "[OK] PASSED" if result else "[FAIL] FAILED"
        print(f"{name:20} {status}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n*** ALL VALIDATION TESTS PASSED!")
        print("MLPY framework is working correctly!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} tests failed")
        print("Some components need fixing")
        return 1


if __name__ == "__main__":
    sys.exit(main())