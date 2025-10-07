"""
Simple tests to increase coverage for MLPY framework.
Focus on modules that work correctly.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os


class TestCoreFunctionality:
    """Test core MLPY functionality."""
    
    @pytest.mark.unit
    def test_basic_classification_workflow(self):
        """Test complete classification workflow."""
        from mlpy.tasks import TaskClassif
        from mlpy.learners.baseline import LearnerClassifFeatureless
        from mlpy.predictions import PredictionClassif
        from mlpy.measures import MeasureClassifAccuracy
        from mlpy.resamplings import ResamplingHoldout
        
        # Create data
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Create task
        task = TaskClassif(data=data, target='target')
        assert task.task_type == 'classif'
        assert task.nrow == 100
        assert task.ncol == 3
        
        # Create resampling
        resampling = ResamplingHoldout(ratio=0.3)
        resampling_instance = resampling.instantiate(task)
        
        train_idx = resampling_instance.train_set(0)
        test_idx = resampling_instance.test_set(0)
        
        assert len(train_idx) + len(test_idx) == 100
        
        # Train learner
        learner = LearnerClassifFeatureless()
        train_task = task.filter(train_idx)
        learner.train(train_task)
        assert learner.is_trained
        
        # Predict
        test_task = task.filter(test_idx)
        predictions = learner.predict(test_task)
        assert isinstance(predictions, PredictionClassif)
        assert len(predictions.response) == len(test_idx)
        
        # Measure performance
        measure = MeasureClassifAccuracy()
        score = measure.score(predictions.truth, predictions.response)
        assert 0 <= score <= 1
    
    @pytest.mark.unit
    def test_basic_regression_workflow(self):
        """Test complete regression workflow."""
        from mlpy.tasks import TaskRegr
        from mlpy.learners.baseline import LearnerRegrFeatureless
        from mlpy.predictions import PredictionRegr
        from mlpy.measures import MeasureRegrMSE, MeasureRegrMAE
        from mlpy.resamplings import ResamplingCV
        
        # Create data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(100) * 0.1
        
        data = pd.DataFrame(X, columns=['feature1', 'feature2'])
        data['target'] = y
        
        # Create task
        task = TaskRegr(data=data, target='target')
        assert task.task_type == 'regr'
        
        # Create CV resampling
        cv = ResamplingCV(folds=5)
        cv_instance = cv.instantiate(task)
        
        scores = []
        for fold in range(5):
            train_idx = cv_instance.train_set(fold)
            test_idx = cv_instance.test_set(fold)
            
            # Train
            learner = LearnerRegrFeatureless()
            train_task = task.filter(train_idx)
            learner.train(train_task)
            
            # Predict
            test_task = task.filter(test_idx)
            predictions = learner.predict(test_task)
            
            # Measure
            measure = MeasureRegrMSE()
            score = measure.score(predictions.truth, predictions.response)
            scores.append(score)
        
        assert len(scores) == 5
        assert all(s >= 0 for s in scores)
    
    @pytest.mark.unit
    def test_ensemble_workflow(self):
        """Test ensemble learning workflow."""
        from mlpy.tasks import TaskClassif
        from mlpy.learners.baseline import LearnerClassifFeatureless
        from mlpy.learners.ensemble import LearnerVoting
        
        # Create data
        data = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'target': np.random.choice(['A', 'B'], 50)
        })
        
        task = TaskClassif(data=data, target='target')
        
        # Create ensemble
        base_learners = [
            LearnerClassifFeatureless(id="learner1"),
            LearnerClassifFeatureless(id="learner2")
        ]
        
        ensemble = LearnerVoting(base_learners=base_learners, voting='hard')
        
        # Train ensemble
        ensemble.train(task)
        assert ensemble.is_trained
        
        # Predict
        predictions = ensemble.predict(task)
        assert len(predictions.response) == 50
    
    @pytest.mark.unit
    def test_model_registry(self):
        """Test model registry functionality."""
        from mlpy.model_registry.registry import ModelRegistry, ModelMetadata
        from mlpy.model_registry.registry import ModelCategory, TaskType, Complexity
        
        registry = ModelRegistry()
        
        # Register a model
        metadata = ModelMetadata(
            name="test_model",
            display_name="Test Model",
            description="A test model",
            category=ModelCategory.TRADITIONAL_ML,
            class_path="mlpy.learners.baseline.LearnerClassifFeatureless",
            task_types=[TaskType.CLASSIFICATION],
            complexity=Complexity.LOW
        )
        
        success = registry.register(metadata)
        assert success
        
        # Get model
        retrieved = registry.get("test_model")
        assert retrieved == metadata
        
        # Search models
        models = registry.search(task_type=TaskType.CLASSIFICATION)
        assert any(m.name == "test_model" for m in models)
    
    @pytest.mark.unit
    def test_validation_functions(self):
        """Test validation functions."""
        from mlpy.validation.validators import validate_task_data, validate_model_params
        
        # Test data validation
        good_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice(['A', 'B'], 100)
        })
        
        result = validate_task_data(good_data, target='target', task_type='classification')
        assert result['valid'] == True
        assert result['stats']['n_samples'] == 100
        
        # Test with missing values
        bad_data = good_data.copy()
        bad_data.iloc[0:10, 0] = np.nan
        
        result = validate_task_data(bad_data, target='target')
        assert result['valid'] == True  # Still valid but with warnings
        assert len(result['warnings']) > 0
        
        # Test parameter validation
        params = {'n_estimators': 100, 'max_depth': 5}
        result = validate_model_params('RandomForest', params)
        assert result['valid'] == True
        
        bad_params = {'n_estimators': -10}
        result = validate_model_params('RandomForest', bad_params)
        assert result['valid'] == False
    
    @pytest.mark.unit
    def test_sklearn_wrapper(self):
        """Test sklearn wrapper functionality."""
        try:
            from mlpy.learners.sklearn import LearnerRandomForestClassifier
            from mlpy.tasks import TaskClassif
            
            # Create data
            data = pd.DataFrame({
                'feature1': np.random.randn(50),
                'feature2': np.random.randn(50),
                'target': np.random.choice([0, 1], 50)
            })
            
            task = TaskClassif(data=data, target='target')
            
            # Create and train model
            learner = LearnerRandomForestClassifier(n_estimators=10, random_state=42)
            learner.train(task)
            assert learner.is_trained
            
            # Predict
            predictions = learner.predict(task)
            assert len(predictions.response) == 50
            
        except ImportError:
            pytest.skip("scikit-learn not installed")
    
    @pytest.mark.unit
    def test_benchmark_functionality(self):
        """Test benchmarking functionality."""
        from mlpy.benchmark import Benchmark, BenchmarkResult
        from mlpy.tasks import TaskClassif
        from mlpy.learners.baseline import LearnerClassifFeatureless
        from mlpy.resamplings import ResamplingHoldout
        from mlpy.measures import MeasureClassifAccuracy
        
        # Create simple benchmark
        data = pd.DataFrame({
            'feature1': np.random.randn(50),
            'target': np.random.choice(['A', 'B'], 50)
        })
        
        task = TaskClassif(data=data, target='target')
        learner = LearnerClassifFeatureless()
        resampling = ResamplingHoldout(ratio=0.3)
        measure = MeasureClassifAccuracy()
        
        # Create benchmark
        benchmark = Benchmark(
            tasks=[task],
            learners=[learner],
            resamplings=[resampling],
            measures=[measure]
        )
        
        # Run benchmark (simplified)
        assert benchmark is not None
        assert len(benchmark.tasks) == 1
        assert len(benchmark.learners) == 1
    
    @pytest.mark.unit
    def test_model_factory(self):
        """Test model factory functionality."""
        from mlpy.model_registry.factory import ModelFactory, create_ensemble
        
        # Test creating a learner
        learner = ModelFactory.create_learner(
            class_path="mlpy.learners.baseline.LearnerClassifFeatureless",
            init_params={"id": "test"}
        )
        assert learner.id == "test"
        
        # Test creating ensemble
        ensemble = create_ensemble(
            base_learners=[
                "mlpy.learners.baseline.LearnerClassifFeatureless",
                "mlpy.learners.baseline.LearnerClassifFeatureless"
            ],
            ensemble_type="voting"
        )
        assert len(ensemble.base_learners) == 2
    
    @pytest.mark.unit
    def test_serialization(self):
        """Test model serialization."""
        import pickle
        from mlpy.learners.baseline import LearnerClassifFeatureless
        from mlpy.tasks import TaskClassif
        
        # Create and train model
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': ['A', 'B', 'A', 'B', 'A']
        })
        
        task = TaskClassif(data=data, target='target')
        learner = LearnerClassifFeatureless()
        learner.train(task)
        
        # Serialize
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            pickle.dump(learner, f)
            temp_path = f.name
        
        # Deserialize
        with open(temp_path, 'rb') as f:
            loaded_learner = pickle.load(f)
        
        assert loaded_learner.is_trained
        
        # Clean up
        os.unlink(temp_path)
    
    @pytest.mark.unit 
    def test_measure_aggregation(self):
        """Test measure aggregation methods."""
        from mlpy.measures import MeasureClassifAccuracy
        
        measure = MeasureClassifAccuracy()
        
        # Test aggregation
        scores = [0.8, 0.85, 0.9, 0.75, 0.82]
        agg = measure.aggregate(scores)
        
        assert 'mean' in agg
        assert 'std' in agg
        assert 'min' in agg
        assert 'max' in agg
        assert 'median' in agg
        
        assert agg['min'] == 0.75
        assert agg['max'] == 0.9
        assert 0.82 <= agg['mean'] <= 0.83
    
    @pytest.mark.unit
    def test_utils_registry(self):
        """Test utils registry functionality."""
        from mlpy.utils.registry import mlpy_learners, mlpy_measures, mlpy_resamplings
        
        # Check registries are dictionaries
        assert isinstance(mlpy_learners, dict)
        assert isinstance(mlpy_measures, dict)
        assert isinstance(mlpy_resamplings, dict)
        
        # Check some items exist
        assert len(mlpy_measures) > 0
        assert len(mlpy_resamplings) > 0


# Run with: pytest tests/unit/test_framework_coverage.py -v