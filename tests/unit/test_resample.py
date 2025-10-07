"""Unit tests for resample function and ResampleResult."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock

from mlpy.resample import resample, ResampleResult
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.resamplings import ResamplingHoldout, ResamplingCV
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifCE, MeasureRegrMSE
from mlpy.predictions import PredictionClassif, PredictionRegr
from mlpy.learners import Learner


class MockLearnerClassif(Learner):
    """Mock classification learner for testing."""
    
    def __init__(self, predict_proba=False, **kwargs):
        super().__init__(id='mock_classif', **kwargs)
        self.predict_proba = predict_proba
        self.train_calls = []
        self.predict_calls = []
        self._model = None
        
    @property
    def task_type(self) -> str:
        return 'classif'
        
    def train(self, task, row_ids=None):
        """Mock training - just stores the call."""
        self.train_calls.append((task, row_ids))
        self._model = "trained"
        return self
        
    def predict(self, task, row_ids=None):
        """Mock prediction - returns random predictions."""
        self.predict_calls.append((task, row_ids))
        
        if self._model is None:
            raise RuntimeError("Model not trained")
            
        # Get actual row_ids to predict
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
        else:
            row_ids = sorted(row_ids)
            
        n = len(row_ids)
        
        # Get class names from task
        classes = task.class_names
        
        # Generate random predictions
        response = np.random.choice(classes, size=n)
        truth = task.truth(rows=row_ids)
        
        # Generate probabilities if needed
        prob = None
        if self.predict_proba:
            # Random probabilities that sum to 1
            prob = np.random.dirichlet(np.ones(len(classes)), size=n)
            prob = pd.DataFrame(prob, columns=classes)
            
        return PredictionClassif(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            truth=truth,
            response=response,
            prob=prob
        )
        
    def clone(self, deep=True):
        """Clone the learner."""
        return MockLearnerClassif(predict_proba=self.predict_proba)


class MockLearnerRegr(Learner):
    """Mock regression learner for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(id='mock_regr', **kwargs)
        self.train_calls = []
        self.predict_calls = []
        self._model = None
        
    @property
    def task_type(self) -> str:
        return 'regr'
        
    def train(self, task, row_ids=None):
        """Mock training."""
        self.train_calls.append((task, row_ids))
        self._model = "trained"
        return self
        
    def predict(self, task, row_ids=None):
        """Mock prediction - returns random predictions."""
        self.predict_calls.append((task, row_ids))
        
        if self._model is None:
            raise RuntimeError("Model not trained")
            
        # Get actual row_ids
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
        else:
            row_ids = sorted(row_ids)
            
        n = len(row_ids)
        
        # Generate random predictions
        response = np.random.randn(n) * 10 + 50  # Mean 50, std 10
        truth = task.truth(rows=row_ids)
        
        return PredictionRegr(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            truth=truth,
            response=response
        )
        
    def clone(self, deep=True):
        """Clone the learner."""
        return MockLearnerRegr()


class TestResampleResult:
    """Test ResampleResult class."""
    
    @pytest.fixture
    def simple_task(self):
        """Create a simple classification task."""
        df = pd.DataFrame({
            'x1': range(10),
            'x2': range(10, 20),
            'y': ['A', 'B'] * 5
        })
        return TaskClassif(data=df, target='y', id='test_task')
        
    @pytest.fixture
    def mock_learner(self):
        """Create a mock learner."""
        return MockLearnerClassif()
        
    @pytest.fixture
    def measures(self):
        """Create test measures."""
        return [MeasureClassifAccuracy(), MeasureClassifCE()]
        
    def test_result_initialization(self, simple_task, mock_learner, measures):
        """Test ResampleResult initialization."""
        resampling = ResamplingCV(folds=3)
        
        result = ResampleResult(
            task=simple_task,
            learner=mock_learner,
            resampling=resampling,
            measures=measures
        )
        
        assert result.task is simple_task
        assert result.learner is mock_learner
        assert result.resampling is resampling
        assert result.measures == measures
        
        # Check storage initialization
        assert result.iterations == []
        assert result.predictions == []
        assert result.train_times == []
        assert result.predict_times == []
        assert result.errors == []
        assert 'classif.acc' in result.scores
        assert 'classif.ce' in result.scores
        
    def test_add_iteration_success(self, simple_task, mock_learner, measures):
        """Test adding successful iteration."""
        result = ResampleResult(
            task=simple_task,
            learner=mock_learner,
            resampling=ResamplingCV(folds=3),
            measures=measures
        )
        
        # Mock prediction
        prediction = Mock(spec=PredictionClassif)
        scores = {'classif.acc': 0.85, 'classif.ce': 0.15}
        
        result.add_iteration(
            iteration=0,
            prediction=prediction,
            scores=scores,
            train_time=1.23,
            predict_time=0.45
        )
        
        assert result.n_iters == 1
        assert result.n_errors == 0
        assert result.predictions[0] is prediction
        assert result.scores['classif.acc'] == [0.85]
        assert result.scores['classif.ce'] == [0.15]
        assert result.train_times == [1.23]
        assert result.predict_times == [0.45]
        
    def test_add_iteration_error(self, simple_task, mock_learner, measures):
        """Test adding failed iteration."""
        result = ResampleResult(
            task=simple_task,
            learner=mock_learner,
            resampling=ResamplingCV(folds=3),
            measures=measures
        )
        
        error = RuntimeError("Training failed")
        
        result.add_iteration(
            iteration=0,
            prediction=None,
            scores=None,
            train_time=0.0,
            predict_time=0.0,
            error=error
        )
        
        assert result.n_iters == 1
        assert result.n_errors == 1
        assert result.predictions[0] is None
        assert np.isnan(result.scores['classif.acc'][0])
        assert np.isnan(result.scores['classif.ce'][0])
        assert result.errors[0] is error
        
    def test_aggregate(self, simple_task, mock_learner, measures):
        """Test score aggregation."""
        result = ResampleResult(
            task=simple_task,
            learner=mock_learner,
            resampling=ResamplingCV(folds=3),
            measures=measures
        )
        
        # Add some iterations
        scores_list = [
            {'classif.acc': 0.80, 'classif.ce': 0.20},
            {'classif.acc': 0.85, 'classif.ce': 0.15},
            {'classif.acc': 0.90, 'classif.ce': 0.10}
        ]
        
        for i, scores in enumerate(scores_list):
            result.add_iteration(
                iteration=i,
                prediction=Mock(),
                scores=scores,
                train_time=1.0,
                predict_time=0.5
            )
            
        # Test aggregation
        agg_df = result.aggregate()
        
        assert len(agg_df) == 2  # Two measures
        
        # Check accuracy aggregation
        acc_row = agg_df[agg_df['measure'] == 'classif.acc'].iloc[0]
        assert acc_row['mean'] == pytest.approx(0.85, rel=1e-6)
        # std of [0.80, 0.85, 0.90] is approximately 0.0408
        assert acc_row['std'] == pytest.approx(0.04082482904638629, rel=1e-6)
        assert acc_row['min'] == 0.80
        assert acc_row['max'] == 0.90
        
    def test_score_method(self, simple_task, mock_learner):
        """Test getting single aggregated score."""
        measures = [MeasureClassifAccuracy()]
        result = ResampleResult(
            task=simple_task,
            learner=mock_learner,
            resampling=ResamplingCV(folds=3),
            measures=measures
        )
        
        # Add iterations
        for i, acc in enumerate([0.80, 0.85, 0.90]):
            result.add_iteration(
                iteration=i,
                prediction=Mock(),
                scores={'classif.acc': acc},
                train_time=1.0,
                predict_time=0.5
            )
            
        # Test default (mean)
        assert result.score() == pytest.approx(0.85, rel=1e-6)
        
        # Test specific aggregation
        assert result.score(average='max') == 0.90
        assert result.score(average='min') == 0.80
        
        # Test error for invalid measure
        with pytest.raises(ValueError, match="Measure 'invalid' not found"):
            result.score(measure_id='invalid')
            
        # Test error for invalid aggregation
        with pytest.raises(ValueError, match="Aggregation 'invalid' not available"):
            result.score(average='invalid')


class TestResampleFunction:
    """Test resample function."""
    
    @pytest.fixture
    def classif_task(self):
        """Create classification task."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100),
            'y': np.random.choice(['A', 'B', 'C'], 100)
        })
        return TaskClassif(data=df, target='y', id='test_classif')
        
    @pytest.fixture
    def regr_task(self):
        """Create regression task."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'y': np.random.randn(100) * 10 + 50
        })
        return TaskRegr(data=df, target='y', id='test_regr')
        
    def test_resample_holdout_classification(self, classif_task):
        """Test resample with holdout on classification."""
        learner = MockLearnerClassif()
        resampling = ResamplingHoldout(ratio=0.7, seed=42)
        measures = [MeasureClassifAccuracy(), MeasureClassifCE()]
        
        result = resample(
            task=classif_task,
            learner=learner,
            resampling=resampling,
            measures=measures
        )
        
        # Check result
        assert isinstance(result, ResampleResult)
        assert result.n_iters == 1
        assert result.n_errors == 0
        
        # Check that resampling worked (learner was encapsulated by default)
        # So the original learner was not modified
        assert len(learner.train_calls) == 0
        assert len(learner.predict_calls) == 0
        
        # Check scores exist
        assert 'classif.acc' in result.scores
        assert 'classif.ce' in result.scores
        assert len(result.scores['classif.acc']) == 1
        
    def test_resample_cv_regression(self, regr_task):
        """Test resample with CV on regression."""
        learner = MockLearnerRegr()
        resampling = ResamplingCV(folds=5, seed=42)
        measures = MeasureRegrMSE()
        
        result = resample(
            task=regr_task,
            learner=learner,
            resampling=resampling,
            measures=measures
        )
        
        # Check result
        assert result.n_iters == 5
        assert result.n_errors == 0
        
        # Check that original learner was not modified (encapsulated by default)
        assert len(learner.train_calls) == 0
        assert len(learner.predict_calls) == 0
        
        # Check scores
        assert len(result.scores['regr.mse']) == 5
        assert all(isinstance(s, float) for s in result.scores['regr.mse'])
        
    def test_resample_with_single_measure(self, classif_task):
        """Test resample with single measure (not list)."""
        learner = MockLearnerClassif()
        resampling = ResamplingHoldout(ratio=0.8)
        measure = MeasureClassifAccuracy()  # Single measure
        
        result = resample(
            task=classif_task,
            learner=learner,
            resampling=resampling,
            measures=measure  # Not a list
        )
        
        assert result.n_iters == 1
        assert 'classif.acc' in result.scores
        
    def test_resample_encapsulation(self, classif_task):
        """Test learner encapsulation."""
        learner = MockLearnerClassif()
        resampling = ResamplingHoldout()
        measures = MeasureClassifAccuracy()
        
        # With encapsulation (default)
        result1 = resample(
            task=classif_task,
            learner=learner,
            resampling=resampling,
            measures=measures,
            encapsulate=True
        )
        
        # Original learner should not be modified
        assert len(learner.train_calls) == 0
        assert len(learner.predict_calls) == 0
        
        # Without encapsulation
        result2 = resample(
            task=classif_task,
            learner=learner,
            resampling=resampling,
            measures=measures,
            encapsulate=False
        )
        
        # Original learner should be modified
        assert len(learner.train_calls) == 1
        assert len(learner.predict_calls) == 1
        
    def test_resample_instantiated_resampling(self, classif_task):
        """Test with already instantiated resampling."""
        learner = MockLearnerClassif()
        resampling = ResamplingCV(folds=3)
        resampling.instantiate(classif_task)
        measures = MeasureClassifAccuracy()
        
        result = resample(
            task=classif_task,
            learner=learner,
            resampling=resampling,
            measures=measures
        )
        
        assert result.n_iters == 3
        
    def test_resample_error_handling(self, classif_task):
        """Test error handling during resampling."""
        # Create learner that fails on second iteration
        learner = MockLearnerClassif()
        original_train = learner.train
        
        call_count = [0]  # Use a mutable object to track calls
        
        def failing_train(task, row_ids=None):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail only on second call
                raise RuntimeError("Training failed")
            return original_train(task, row_ids)
            
        learner.train = failing_train
        
        resampling = ResamplingCV(folds=3)
        measures = MeasureClassifAccuracy()
        
        result = resample(
            task=classif_task,
            learner=learner,
            resampling=resampling,
            measures=measures,
            encapsulate=False  # Don't encapsulate so our modified train method is used
        )
        
        # Should complete all iterations
        assert result.n_iters == 3
        assert result.n_errors == 1
        
        # Second iteration should have NaN score
        assert not np.isnan(result.scores['classif.acc'][0])
        assert np.isnan(result.scores['classif.acc'][1])
        assert not np.isnan(result.scores['classif.acc'][2])
        
    def test_resample_incompatible_measure(self, classif_task, regr_task):
        """Test error when measure incompatible with task."""
        learner = MockLearnerClassif()
        resampling = ResamplingHoldout()
        
        # Classification measure on regression task
        with pytest.raises(ValueError, match="not applicable"):
            resample(
                task=regr_task,
                learner=learner,
                resampling=resampling,
                measures=MeasureClassifAccuracy()
            )
            
        # Regression measure on classification task  
        learner = MockLearnerRegr()
        with pytest.raises(ValueError, match="not applicable"):
            resample(
                task=classif_task,
                learner=learner,
                resampling=resampling,
                measures=MeasureRegrMSE()
            )
            
    def test_resample_no_measures(self, classif_task):
        """Test error when no measures provided."""
        learner = MockLearnerClassif()
        resampling = ResamplingHoldout()
        
        with pytest.raises(ValueError, match="At least one measure"):
            resample(
                task=classif_task,
                learner=learner,
                resampling=resampling,
                measures=[]
            )
            
    def test_resample_integration(self, classif_task):
        """Integration test with multiple components."""
        learner = MockLearnerClassif(predict_proba=True)
        resampling = ResamplingCV(folds=5, stratify=True, seed=42)
        measures = [
            MeasureClassifAccuracy(),
            MeasureClassifCE(),
        ]
        
        result = resample(
            task=classif_task,
            learner=learner,
            resampling=resampling,
            measures=measures
        )
        
        # Check complete execution
        assert result.n_iters == 5
        assert result.n_errors == 0
        
        # Check aggregation works
        agg_df = result.aggregate()
        assert len(agg_df) == 2  # Two measures
        assert all(col in agg_df.columns for col in ['measure', 'mean', 'std'])
        
        # Check can get scores
        acc_score = result.score('classif.acc', 'mean')
        assert 0 <= acc_score <= 1
        
        # Check timing recorded
        assert all(t > 0 for t in result.train_times)
        assert all(t > 0 for t in result.predict_times)