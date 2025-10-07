"""
Comprehensive tests to increase coverage for core MLPY modules.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


class TestMLPYBase:
    """Test base MLPY classes."""
    
    @pytest.mark.unit
    def test_mlpy_object_creation(self):
        """Test MLPYObject basic functionality."""
        from mlpy.base import MLPYObject
        
        obj = MLPYObject(id="test_obj")
        assert obj.id == "test_obj"
        assert str(obj) == "<MLPYObject:test_obj>"
        assert repr(obj) == "<MLPYObject:test_obj>"
        
        # Test hash
        obj2 = MLPYObject(id="test_obj")
        obj3 = MLPYObject(id="different")
        assert hash(obj) == hash(obj2)
        assert hash(obj) != hash(obj3)
        
        # Test equality
        assert obj == obj2
        assert obj != obj3
    
    @pytest.mark.unit
    def test_mlpy_object_properties(self):
        """Test MLPYObject properties."""
        from mlpy.base import MLPYObject
        
        obj = MLPYObject(id="test", label="Test Object")
        assert obj.label == "Test Object"
        
        # Test man property
        assert hasattr(obj, 'man')
        
        # Test print method
        obj.print()  # Should not raise
        
        # Test clone
        cloned = obj.clone()
        assert cloned.id == obj.id
        assert cloned is not obj


class TestTasks:
    """Test Task classes."""
    
    @pytest.mark.unit
    def test_task_classif_creation(self):
        """Test TaskClassif creation."""
        from mlpy.tasks import TaskClassif
        
        # Create sample data
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        task = TaskClassif(data=data, target='target')
        assert task.task_type == 'classif'
        assert len(task.data) == 100
        assert task.target == 'target'
        
        # Test properties
        assert task.nrow == 100
        assert task.ncol == 3
        assert task.feature_names == ['feature1', 'feature2']
        assert set(task.class_names) == {'A', 'B', 'C'}
        assert task.n_classes == 3
    
    @pytest.mark.unit  
    def test_task_regr_creation(self):
        """Test TaskRegr creation."""
        from mlpy.tasks import TaskRegr
        
        # Create sample data
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        })
        
        task = TaskRegr(data=data, target='target')
        assert task.task_type == 'regr'
        assert len(task.data) == 100
        assert task.target == 'target'
        
        # Test properties
        assert task.nrow == 100
        assert task.ncol == 3
        assert task.feature_names == ['feature1', 'feature2']
    
    @pytest.mark.unit
    def test_task_filter(self):
        """Test task filtering."""
        from mlpy.tasks import TaskClassif
        
        data = pd.DataFrame({
            'feature1': range(100),
            'target': ['A'] * 50 + ['B'] * 50
        })
        
        task = TaskClassif(data=data, target='target')
        
        # Filter by indices
        filtered = task.filter([0, 1, 2])
        assert len(filtered.data) == 3
        assert list(filtered.data['feature1']) == [0, 1, 2]
    
    @pytest.mark.unit
    def test_task_head_tail(self):
        """Test task head and tail methods."""
        from mlpy.tasks import TaskRegr
        
        data = pd.DataFrame({
            'feature1': range(100),
            'target': range(100)
        })
        
        task = TaskRegr(data=data, target='target')
        
        # Test head
        head_task = task.head(5)
        assert len(head_task.data) == 5
        assert list(head_task.data['feature1']) == [0, 1, 2, 3, 4]
        
        # Test tail
        tail_task = task.tail(5)
        assert len(tail_task.data) == 5
        assert list(tail_task.data['feature1']) == [95, 96, 97, 98, 99]


class TestPredictions:
    """Test Prediction classes."""
    
    @pytest.mark.unit
    def test_prediction_classif(self):
        """Test PredictionClassif."""
        from mlpy.predictions import PredictionClassif
        
        truth = ['A', 'B', 'A', 'B', 'A']
        response = ['A', 'B', 'B', 'B', 'A']
        
        pred = PredictionClassif(
            task=None,
            learner_id="test_learner",
            row_ids=[0, 1, 2, 3, 4],
            truth=truth,
            response=response
        )
        
        assert pred.learner_id == "test_learner"
        assert len(pred.response) == 5
        assert pred.predict_type == "response"
        assert pred.n == 5
        
        # Test confusion matrix
        cm = pred.confusion_matrix
        assert cm is not None
    
    @pytest.mark.unit
    def test_prediction_regr(self):
        """Test PredictionRegr."""
        from mlpy.predictions import PredictionRegr
        
        truth = [1.0, 2.0, 3.0, 4.0, 5.0]
        response = [1.1, 2.1, 2.9, 4.2, 4.8]
        
        pred = PredictionRegr(
            task=None,
            learner_id="test_learner",
            row_ids=[0, 1, 2, 3, 4],
            truth=truth,
            response=response
        )
        
        assert pred.learner_id == "test_learner"
        assert len(pred.response) == 5
        assert pred.predict_type == "response"
        
        # Test residuals
        residuals = pred.residuals
        assert len(residuals) == 5
        assert np.allclose(residuals[0], 0.1, atol=0.01)


class TestMeasures:
    """Test Measure classes."""
    
    @pytest.mark.unit
    def test_measure_accuracy(self):
        """Test accuracy measure."""
        from mlpy.measures import MeasureClassifAccuracy
        
        measure = MeasureClassifAccuracy()
        assert measure.id == 'classif.acc'
        assert measure.task_type == 'classif'
        assert measure.minimize == False
        
        # Test scoring
        y_true = ['A', 'B', 'A', 'B', 'A']
        y_pred = ['A', 'B', 'A', 'B', 'A']
        
        score = measure.score(y_true, y_pred)
        assert score == 1.0
        
        y_pred2 = ['B', 'A', 'B', 'A', 'B']
        score2 = measure.score(y_true, y_pred2)
        assert score2 == 0.0
    
    @pytest.mark.unit
    def test_measure_mse(self):
        """Test MSE measure."""
        from mlpy.measures import MeasureRegrMSE
        
        measure = MeasureRegrMSE()
        assert measure.id == 'regr.mse'
        assert measure.task_type == 'regr'
        assert measure.minimize == True
        
        # Test scoring
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        score = measure.score(y_true, y_pred)
        assert score == 0.0
        
        y_pred2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        score2 = measure.score(y_true, y_pred2)
        assert np.allclose(score2, 0.01, atol=0.001)
    
    @pytest.mark.unit
    def test_measure_aggregation(self):
        """Test measure aggregation."""
        from mlpy.measures import MeasureClassifAccuracy
        
        measure = MeasureClassifAccuracy()
        scores = [0.8, 0.85, 0.9, 0.75, 0.82]
        
        agg = measure.aggregate(scores)
        assert 'mean' in agg
        assert 'std' in agg
        assert 'min' in agg
        assert 'max' in agg
        assert 'median' in agg
        
        assert np.allclose(agg['mean'], 0.824, atol=0.001)
        assert agg['min'] == 0.75
        assert agg['max'] == 0.9


class TestResamplings:
    """Test Resampling classes."""
    
    @pytest.mark.unit
    def test_resampling_holdout(self):
        """Test holdout resampling."""
        from mlpy.resamplings import ResamplingHoldout
        from mlpy.tasks import TaskClassif
        
        # Create task
        data = pd.DataFrame({
            'feature1': range(100),
            'target': ['A'] * 50 + ['B'] * 50
        })
        task = TaskClassif(data=data, target='target')
        
        # Create holdout
        holdout = ResamplingHoldout(ratio=0.3, stratify=True)
        assert holdout.ratio == 0.3
        assert holdout.stratify == True
        
        # Instantiate
        instance = holdout.instantiate(task)
        assert instance.n_iters == 1
        
        # Get train/test sets
        train_idx = instance.train_set(0)
        test_idx = instance.test_set(0)
        
        assert len(train_idx) == 70
        assert len(test_idx) == 30
        assert len(set(train_idx) & set(test_idx)) == 0  # No overlap
    
    @pytest.mark.unit
    def test_resampling_cv(self):
        """Test cross-validation resampling."""
        from mlpy.resamplings import ResamplingCV
        from mlpy.tasks import TaskRegr
        
        # Create task
        data = pd.DataFrame({
            'feature1': range(100),
            'target': np.random.randn(100)
        })
        task = TaskRegr(data=data, target='target')
        
        # Create CV
        cv = ResamplingCV(folds=5)
        assert cv.folds == 5
        
        # Instantiate
        instance = cv.instantiate(task)
        assert instance.n_iters == 5
        
        # Check all folds
        all_test_idx = []
        for i in range(5):
            train_idx = instance.train_set(i)
            test_idx = instance.test_set(i)
            
            assert len(train_idx) == 80
            assert len(test_idx) == 20
            assert len(set(train_idx) & set(test_idx)) == 0
            all_test_idx.extend(test_idx)
        
        # All samples should be in test set exactly once
        assert sorted(all_test_idx) == list(range(100))
    
    @pytest.mark.unit
    def test_resampling_bootstrap(self):
        """Test bootstrap resampling."""
        from mlpy.resamplings import ResamplingBootstrap
        from mlpy.tasks import TaskClassif
        
        # Create task
        data = pd.DataFrame({
            'feature1': range(50),
            'target': ['A'] * 25 + ['B'] * 25
        })
        task = TaskClassif(data=data, target='target')
        
        # Create bootstrap
        bootstrap = ResamplingBootstrap(n_iters=10, ratio=0.632)
        assert bootstrap.n_iters == 10
        
        # Instantiate
        instance = bootstrap.instantiate(task)
        assert instance.n_iters == 10
        
        # Check one iteration
        train_idx = instance.train_set(0)
        test_idx = instance.test_set(0)
        
        # Bootstrap samples with replacement
        assert len(train_idx) == 50  # Same size as original
        assert len(test_idx) > 0  # Out-of-bag samples


class TestLearners:
    """Test Learner classes."""
    
    @pytest.mark.unit
    def test_learner_baseline_classif(self):
        """Test baseline classification learner."""
        from mlpy.learners.baseline import LearnerClassifFeatureless
        from mlpy.tasks import TaskClassif
        
        # Create task
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'target': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2])
        })
        task = TaskClassif(data=data, target='target')
        
        # Create and train learner
        learner = LearnerClassifFeatureless()
        assert not learner.is_trained
        
        learner.train(task)
        assert learner.is_trained
        
        # Predict
        pred = learner.predict(task)
        assert len(pred.response) == 100
        # Should predict most frequent class
        assert all(p == pred.response[0] for p in pred.response)
    
    @pytest.mark.unit
    def test_learner_baseline_regr(self):
        """Test baseline regression learner."""
        from mlpy.learners.baseline import LearnerRegrFeatureless
        from mlpy.tasks import TaskRegr
        
        # Create task
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'target': np.random.randn(100) + 5.0  # Mean around 5
        })
        task = TaskRegr(data=data, target='target')
        
        # Create and train learner
        learner = LearnerRegrFeatureless()
        learner.train(task)
        
        # Predict
        pred = learner.predict(task)
        assert len(pred.response) == 100
        # Should predict mean value
        expected = np.mean(data['target'])
        assert np.allclose(pred.response[0], expected, atol=0.1)
    
    @pytest.mark.unit
    def test_learner_properties(self):
        """Test learner properties."""
        from mlpy.learners.baseline import LearnerClassifFeatureless
        
        learner = LearnerClassifFeatureless(id="my_learner", label="My Learner")
        assert learner.id == "my_learner"
        assert learner.label == "My Learner"
        assert learner.task_type == "classif"
        assert not learner.is_trained
        
        # Test string representation
        assert "my_learner" in str(learner)


class TestEnsemble:
    """Test ensemble learners."""
    
    @pytest.mark.unit
    def test_voting_ensemble(self):
        """Test voting ensemble."""
        from mlpy.learners.ensemble import LearnerVoting
        from mlpy.learners.baseline import LearnerClassifFeatureless
        from mlpy.tasks import TaskClassif
        
        # Create task
        data = pd.DataFrame({
            'feature1': np.random.randn(50),
            'target': np.random.choice(['A', 'B'], 50)
        })
        task = TaskClassif(data=data, target='target')
        
        # Create ensemble
        base_learners = [
            LearnerClassifFeatureless(id="learner1"),
            LearnerClassifFeatureless(id="learner2"),
            LearnerClassifFeatureless(id="learner3")
        ]
        
        ensemble = LearnerVoting(
            base_learners=base_learners,
            voting='hard',
            weights=[0.5, 0.3, 0.2]
        )
        
        assert len(ensemble.base_learners) == 3
        assert ensemble.voting == 'hard'
        assert sum(ensemble.weights) == pytest.approx(1.0)
        
        # Train
        ensemble.train(task)
        assert ensemble.is_trained
        
        # Predict
        pred = ensemble.predict(task)
        assert len(pred.response) == 50


class TestValidation:
    """Test validation functionality."""
    
    @pytest.mark.unit
    def test_validate_task_data(self):
        """Test task data validation."""
        from mlpy.validation.validators import validate_task_data
        
        # Good data
        good_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice(['A', 'B'], 100)
        })
        
        result = validate_task_data(good_data, target='target', task_type='classification')
        assert result['valid'] == True
        assert len(result['errors']) == 0
        assert result['stats']['n_samples'] == 100
        assert result['stats']['n_features'] == 3
        
        # Bad data - too few samples
        bad_data = pd.DataFrame({
            'feature1': [1, 2],
            'target': ['A', 'B']
        })
        
        result = validate_task_data(bad_data, target='target')
        assert result['valid'] == False
        assert len(result['errors']) > 0
        assert 'Insufficient samples' in result['errors'][0]
    
    @pytest.mark.unit
    def test_validate_model_params(self):
        """Test model parameter validation."""
        from mlpy.validation.validators import validate_model_params
        
        # Valid params
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1
        }
        
        result = validate_model_params('RandomForest', params)
        assert result['valid'] == True
        assert len(result['errors']) == 0
        
        # Invalid params
        bad_params = {
            'n_estimators': -10,
            'learning_rate': 2.0
        }
        
        result = validate_model_params('GradientBoosting', bad_params)
        assert result['valid'] == False
        assert len(result['errors']) > 0


class TestUtils:
    """Test utility functions."""
    
    @pytest.mark.unit
    def test_registry(self):
        """Test model registry utilities."""
        from mlpy.utils.registry import (
            mlpy_learners, mlpy_measures, mlpy_resamplings,
            register_learner, register_measure
        )
        
        # Check registries exist
        assert isinstance(mlpy_learners, dict)
        assert isinstance(mlpy_measures, dict)
        assert isinstance(mlpy_resamplings, dict)
        
        # Check some items are registered
        assert len(mlpy_learners) > 0
        assert len(mlpy_measures) > 0
        assert len(mlpy_resamplings) > 0
    
    @pytest.mark.unit
    def test_logging(self):
        """Test logging utilities."""
        from mlpy.utils.logging import get_logger, configure_logging
        
        # Get logger
        logger = get_logger("test")
        assert logger is not None
        
        # Configure logging
        configure_logging(level="DEBUG")
        
        # Test logging (should not raise)
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")


# Run with: pytest tests/unit/test_core_coverage.py -v